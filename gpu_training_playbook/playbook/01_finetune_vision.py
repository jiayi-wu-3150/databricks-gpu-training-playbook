"""Fine-tune a vision transformer on Databricks GPU compute.

Runs unchanged on AI Runtime (serverless GPU) and on classic Databricks Runtime ML.
Every knob is an env var so the workload can be retargeted without editing code.

Signals logged to MLflow: params, per-epoch loss/accuracy, throughput (img/s),
wall time by phase, GPU name/memory, and the final model registered to Unity Catalog.
"""
import os
import sys

# Accept --KEY=value args and fold them into the environment, so the same script
# runs under AIR (env_variables in YAML) and classic (spark_python_task parameters).
for _arg in sys.argv[1:]:
    if _arg.startswith("--") and "=" in _arg:
        _k, _v = _arg[2:].split("=", 1)
        os.environ[_k.upper().replace("-", "_")] = _v
import time
import json

CATALOG = os.environ.get("CATALOG", "classic_stable_qxxm2v_catalog")
SCHEMA = os.environ.get("SCHEMA", "dl_training")
DATASET = os.environ.get("DATASET", "beans")
BASE_MODEL = os.environ.get("BASE_MODEL", "google/vit-base-patch16-224")
EPOCHS = float(os.environ.get("EPOCHS", "3"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
LR = float(os.environ.get("LR", "5e-5"))
BF16 = os.environ.get("BF16", "true").lower() == "true"
REGISTER = os.environ.get("REGISTER_MODEL", "true").lower() == "true"
MODEL_NAME = os.environ.get("MODEL_NAME", f"{CATALOG}.{SCHEMA}.vit_{DATASET}")
RUNTIME_TAG = os.environ.get("RUNTIME_TAG", "unknown")
# Classic .py tasks have NO implicit MLflow experiment (AIR sets one from the
# YAML experiment_name; notebooks on classic inherit one). Set it explicitly.
EXPERIMENT = os.environ.get("EXPERIMENT", "")

DATA_VOL = f"/Volumes/{CATALOG}/{SCHEMA}/data/{DATASET}"
CKPT_VOL = f"/Volumes/{CATALOG}/{SCHEMA}/checkpoints/vit_{DATASET}_{RUNTIME_TAG}"

t0 = time.time()
import torch
import numpy as np
import mlflow
from datasets import load_dataset
from transformers import (AutoImageProcessor, AutoModelForImageClassification,
                          TrainingArguments, Trainer)

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
gpu_mem_gb = (round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
              if torch.cuda.is_available() else 0)
print(f"device={gpu_name} mem={gpu_mem_gb}GB torch={torch.__version__} "
      f"cuda={torch.version.cuda}")

# --- data: read parquet staged in the UC Volume (the documented AIR data path) ---
t_data = time.time()
files = {split: f"{DATA_VOL}/{split}.parquet" for split in ("train", "validation")}
ds = load_dataset("parquet", data_files=files)
labels_path = f"{DATA_VOL}/labels.json"
if os.path.exists(labels_path):
    with open(labels_path) as fh:
        label_names = json.load(fh)
else:
    label_names = [str(i) for i in sorted({int(x) for x in ds["train"]["labels"]})]
num_labels = len(label_names)
id2label = {i: n for i, n in enumerate(label_names)}
label2id = {n: i for i, n in enumerate(label_names)}
print(f"loaded train={len(ds['train'])} val={len(ds['validation'])} labels={num_labels}")

processor = AutoImageProcessor.from_pretrained(BASE_MODEL)


def preprocess(batch):
    images = [img.convert("RGB") for img in batch["image"]]
    out = processor(images, return_tensors="pt")
    out["labels"] = batch["labels"]
    return out


ds = ds.with_transform(preprocess)
data_seconds = time.time() - t_data

model = AutoModelForImageClassification.from_pretrained(
    BASE_MODEL, num_labels=num_labels, id2label=id2label, label2id=label2id,
    ignore_mismatched_sizes=True)


def collate(examples):
    return {"pixel_values": torch.stack([e["pixel_values"] for e in examples]),
            "labels": torch.tensor([e["labels"] for e in examples])}


def metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=1)
    return {"accuracy": float((preds == eval_pred.label_ids).mean())}


args = TrainingArguments(
    output_dir=CKPT_VOL,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    bf16=BF16 and torch.cuda.is_available(),
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=10,
    report_to=["mlflow"],          # MLflow autologging for HF Trainer
    remove_unused_columns=False,
    dataloader_num_workers=4,
)

trainer = Trainer(model=model, args=args, train_dataset=ds["train"],
                  eval_dataset=ds["validation"], data_collator=collate,
                  compute_metrics=metrics)

mlflow.set_registry_uri("databricks-uc")
if EXPERIMENT:
    mlflow.set_experiment(EXPERIMENT)
with mlflow.start_run(run_name=f"vit-{DATASET}-{RUNTIME_TAG}", nested=bool(os.environ.get("MLFLOW_RUN_ID"))):
    mlflow.set_tags({"runtime": RUNTIME_TAG, "gpu": gpu_name, "workload": "vision-finetune"})
    mlflow.log_params({"base_model": BASE_MODEL, "dataset": DATASET, "epochs": EPOCHS,
                       "batch_size": BATCH_SIZE, "lr": LR, "bf16": BF16,
                       "torch": torch.__version__, "cuda": torch.version.cuda,
                       "gpu_mem_gb": gpu_mem_gb, "num_labels": num_labels})

    t_train = time.time()
    train_out = trainer.train()
    train_seconds = time.time() - t_train
    eval_out = trainer.evaluate()

    n_seen = len(ds["train"]) * EPOCHS
    mlflow.log_metrics({
        "train_seconds": train_seconds,
        "data_load_seconds": data_seconds,
        "import_seconds": t_data - t0,
        "total_seconds": time.time() - t0,
        "throughput_img_per_s": n_seen / train_seconds,
        "final_train_loss": train_out.training_loss,
        "final_eval_accuracy": eval_out.get("eval_accuracy", float("nan")),
        "peak_gpu_mem_gb": (torch.cuda.max_memory_allocated() / 1e9
                            if torch.cuda.is_available() else 0),
    })

    if REGISTER:
        # Unwrap the Accelerate/AMP-prepared model and cast to fp32: a prepared
        # AMP model cannot be pickled, and bf16 weights break signature inference.
        to_log = trainer.accelerator.unwrap_model(trainer.model)
        to_log = to_log.to("cpu").to(torch.float32).eval()
        mlflow.transformers.log_model(
            transformers_model={"model": to_log, "image_processor": processor},
            task="image-classification",
            name="model",
            registered_model_name=MODEL_NAME)
        print(f"registered {MODEL_NAME}")

    summary = {"runtime": RUNTIME_TAG, "gpu": gpu_name,
               "train_seconds": round(train_seconds, 1),
               "throughput_img_per_s": round(n_seen / train_seconds, 2),
               "eval_accuracy": eval_out.get("eval_accuracy")}
    print("SUMMARY " + json.dumps(summary))
