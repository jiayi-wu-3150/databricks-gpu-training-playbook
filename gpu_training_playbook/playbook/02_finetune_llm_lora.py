"""LoRA supervised fine-tune of a small open LLM on Databricks GPU compute.

Runs unchanged on AI Runtime (serverless GPU) and classic Databricks Runtime ML.
Every knob is an env var, so retargeting needs no code edits.

Logs to MLflow: params, train/eval loss, tokens/s, wall time by phase, peak GPU
memory, trainable-parameter count and ratio, and registers the adapter to UC.
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
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B")
DATASET_ID = os.environ.get("DATASET_ID", "databricks/databricks-dolly-15k")
N_TRAIN = int(os.environ.get("N_TRAIN", "2000"))
N_EVAL = int(os.environ.get("N_EVAL", "200"))
EPOCHS = float(os.environ.get("EPOCHS", "1"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "512"))
LR = float(os.environ.get("LR", "2e-4"))
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))
BF16 = os.environ.get("BF16", "true").lower() == "true"
REGISTER = os.environ.get("REGISTER_MODEL", "true").lower() == "true"
MERGE = os.environ.get("MERGE_ADAPTER", "true").lower() == "true"
RUNTIME_TAG = os.environ.get("RUNTIME_TAG", "unknown")
# Classic .py tasks have NO implicit MLflow experiment (AIR sets one from the
# YAML experiment_name; notebooks on classic inherit one). Set it explicitly.
EXPERIMENT = os.environ.get("EXPERIMENT", "")
MODEL_NAME = os.environ.get(
    "MODEL_NAME", f"{CATALOG}.{SCHEMA}.llm_lora_{BASE_MODEL.split('/')[-1].replace('.', '_').replace('-', '_').lower()}")

CKPT_VOL = f"/Volumes/{CATALOG}/{SCHEMA}/checkpoints/lora_{RUNTIME_TAG}"

t0 = time.time()
import torch
import mlflow
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
gpu_mem_gb = (round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
              if torch.cuda.is_available() else 0)
print(f"device={gpu_name} mem={gpu_mem_gb}GB torch={torch.__version__} cuda={torch.version.cuda}")
import transformers, peft, trl
print(f"transformers={transformers.__version__} peft={peft.__version__} trl={trl.__version__}")

# --- data: instruction-tuning set formatted into a single text field ---
t_data = time.time()
raw = load_dataset(DATASET_ID, split="train")
raw = raw.shuffle(seed=42)
train_raw = raw.select(range(min(N_TRAIN, len(raw))))
eval_raw = raw.select(range(N_TRAIN, min(N_TRAIN + N_EVAL, len(raw))))


def to_text(ex):
    ctx = (ex.get("context") or "").strip()
    instruction = ex["instruction"].strip()
    prompt = (f"### Instruction:\n{instruction}\n\n### Context:\n{ctx}\n\n### Response:\n"
              if ctx else f"### Instruction:\n{instruction}\n\n### Response:\n")
    return {"text": prompt + ex["response"].strip()}


train_ds = train_raw.map(to_text, remove_columns=train_raw.column_names)
eval_ds = eval_raw.map(to_text, remove_columns=eval_raw.column_names)
data_seconds = time.time() - t_data
print(f"train={len(train_ds)} eval={len(eval_ds)}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, dtype=torch.bfloat16 if (BF16 and torch.cuda.is_available()) else torch.float32)
model.config.use_cache = False

lora = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                  bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"])
peft_model = get_peft_model(model, lora)
trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
total = sum(p.numel() for p in peft_model.parameters())
print(f"trainable={trainable:,} / total={total:,} ({100*trainable/total:.3f}%)")

cfg = SFTConfig(
    output_dir=CKPT_VOL,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    bf16=BF16 and torch.cuda.is_available(),
    max_length=MAX_SEQ_LEN,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    report_to=["mlflow"],
    dataset_text_field="text",
)

trainer = SFTTrainer(model=peft_model, args=cfg,
                     train_dataset=train_ds, eval_dataset=eval_ds,
                     processing_class=tokenizer)

mlflow.set_registry_uri("databricks-uc")
if EXPERIMENT:
    mlflow.set_experiment(EXPERIMENT)
with mlflow.start_run(run_name=f"lora-{BASE_MODEL.split('/')[-1]}-{RUNTIME_TAG}",
                      nested=bool(os.environ.get("MLFLOW_RUN_ID"))):
    mlflow.set_tags({"runtime": RUNTIME_TAG, "gpu": gpu_name, "workload": "llm-lora-sft"})
    mlflow.log_params({
        "base_model": BASE_MODEL, "dataset": DATASET_ID, "n_train": len(train_ds),
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
        "effective_batch": BATCH_SIZE * GRAD_ACCUM, "max_seq_len": MAX_SEQ_LEN, "lr": LR,
        "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
        "bf16": BF16, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu_mem_gb": gpu_mem_gb, "trainable_params": trainable, "total_params": total,
        "trainable_pct": round(100 * trainable / total, 4),
    })

    t_train = time.time()
    out = trainer.train()
    train_seconds = time.time() - t_train
    ev = trainer.evaluate()

    tokens = out.metrics.get("train_tokens", None)
    metrics = {
        "train_seconds": train_seconds,
        "data_load_seconds": data_seconds,
        "import_seconds": t_data - t0,
        "total_seconds": time.time() - t0,
        "final_train_loss": out.training_loss,
        "final_eval_loss": ev.get("eval_loss", float("nan")),
        "samples_per_second": len(train_ds) * EPOCHS / train_seconds,
        "peak_gpu_mem_gb": (torch.cuda.max_memory_allocated() / 1e9
                            if torch.cuda.is_available() else 0),
    }
    if tokens:
        metrics["tokens_per_second"] = tokens / train_seconds
    mlflow.log_metrics(metrics)

    if REGISTER:
        adapter_dir = f"{CKPT_VOL}/adapter_final"
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        print(f"adapter saved to {adapter_dir}")
        if MERGE:
            # Merged fp32 weights are the portable artifact: loadable anywhere with
            # plain transformers, no PEFT dependency. Matters for off-platform serving.
            merged = trainer.accelerator.unwrap_model(trainer.model)
            merged = merged.merge_and_unload().to("cpu").to(torch.float32).eval()
            mlflow.transformers.log_model(
                transformers_model={"model": merged, "tokenizer": tokenizer},
                task="text-generation", name="model",
                registered_model_name=MODEL_NAME)
            print(f"registered merged model {MODEL_NAME}")

    summary = {"runtime": RUNTIME_TAG, "gpu": gpu_name,
               "train_seconds": round(train_seconds, 1),
               "samples_per_s": round(len(train_ds) * EPOCHS / train_seconds, 2),
               "train_loss": round(out.training_loss, 4),
               "eval_loss": round(ev.get("eval_loss", float("nan")), 4),
               "trainable_pct": round(100 * trainable / total, 4),
               "peak_gpu_mem_gb": round(metrics["peak_gpu_mem_gb"], 2)}
    print("SUMMARY " + json.dumps(summary))
