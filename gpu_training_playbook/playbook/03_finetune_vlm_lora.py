"""LoRA fine-tune of a small vision-language model (VLM) on Databricks GPU compute.

Runs unchanged on AI Runtime (serverless GPU) and classic Databricks Runtime ML.

WHY THIS TEMPLATE EXISTS, separately from 01 (vision) and 02 (text):
multimodal data loading is genuinely different from both. A VLM sample is an
interleaved image+text conversation, so you must
  1. build chat-template messages containing an image placeholder,
  2. run text and images through ONE processor together,
  3. write a custom collator (the stock text collator will not work),
  4. mask the prompt tokens so loss is computed on the answer only,
  5. budget sequence length for image tokens, which dominate the text.
It also freezes the vision tower and LoRA-adapts only the language projections,
which is the usual recipe for small-data VLM tuning.

Demonstrates turning an existing image dataset already staged in a UC Volume
into a VLM instruction set, so no new data staging is required.
"""
import os
import sys

# Accept --KEY=value args and fold them into the environment, so the same script
# runs under AIR (env_variables in YAML) and classic (spark_python_task parameters).
for _arg in sys.argv[1:]:
    if _arg.startswith("--") and "=" in _arg:
        _k, _v = _arg[2:].split("=", 1)
        os.environ[_k.upper().replace("-", "_")] = _v

import json
import time

CATALOG = os.environ.get("CATALOG", "classic_stable_qxxm2v_catalog")
SCHEMA = os.environ.get("SCHEMA", "dl_training")
DATASET = os.environ.get("DATASET", "beans")
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
N_TRAIN = int(os.environ.get("N_TRAIN", "240"))
N_EVAL = int(os.environ.get("N_EVAL", "40"))
EPOCHS = float(os.environ.get("EPOCHS", "1"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))
LR = float(os.environ.get("LR", "1e-4"))
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", str(336 * 336)))
BF16 = os.environ.get("BF16", "true").lower() == "true"
RUNTIME_TAG = os.environ.get("RUNTIME_TAG", "unknown")
EXPERIMENT = os.environ.get("EXPERIMENT", "")
REGISTER = os.environ.get("REGISTER_MODEL", "false").lower() == "true"
MODEL_NAME = os.environ.get("MODEL_NAME", f"{CATALOG}.{SCHEMA}.vlm_lora_{DATASET}")
QUESTION = os.environ.get(
    "QUESTION", "This is a photo of a bean leaf. Which condition does it show?")

DATA_VOL = f"/Volumes/{CATALOG}/{SCHEMA}/data/{DATASET}"
CKPT_VOL = f"/Volumes/{CATALOG}/{SCHEMA}/checkpoints/vlm_{RUNTIME_TAG}"

t0 = time.time()
import torch
import mlflow
from datasets import load_dataset
from transformers import AutoProcessor, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
gpu_mem_gb = (round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
              if torch.cuda.is_available() else 0)
import transformers, peft
print(f"device={gpu_name} mem={gpu_mem_gb}GB torch={torch.__version__} cuda={torch.version.cuda}")
print(f"transformers={transformers.__version__} peft={peft.__version__}")

# The VLM model class moved across transformers major versions. Try both.
try:
    from transformers import Qwen2VLForConditionalGeneration as VLMClass
    print("model class: Qwen2VLForConditionalGeneration")
except ImportError:
    from transformers import AutoModelForVision2Seq as VLMClass
    print("model class: AutoModelForVision2Seq (fallback)")

# ---------------------------------------------------------------- data
t_data = time.time()
with open(f"{DATA_VOL}/labels.json") as fh:
    label_names = json.load(fh)
pretty = {i: n.replace("_", " ") for i, n in enumerate(label_names)}
print(f"labels: {pretty}")

ds = load_dataset("parquet", data_files={
    "train": f"{DATA_VOL}/train.parquet",
    "validation": f"{DATA_VOL}/validation.parquet"})
train_raw = ds["train"].shuffle(seed=42).select(range(min(N_TRAIN, len(ds["train"]))))
eval_raw = ds["validation"].select(range(min(N_EVAL, len(ds["validation"]))))
data_seconds = time.time() - t_data
print(f"train={len(train_raw)} eval={len(eval_raw)}")

processor = AutoProcessor.from_pretrained(BASE_MODEL, max_pixels=MAX_PIXELS)
tok = processor.tokenizer


def to_messages(example):
    """One interleaved image+text conversation. This is the multimodal-specific part."""
    answer = pretty[int(example["labels"])]
    return [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]


def collate(examples):
    """Custom collator: text and images go through ONE processor call together.

    The stock text collator cannot do this, which is the core difference from
    the text-only LoRA template (02).
    """
    texts, images = [], []
    for ex in examples:
        msgs = to_messages(ex)
        texts.append(processor.apply_chat_template(msgs, tokenize=False,
                                                  add_generation_prompt=False))
        images.append([ex["image"].convert("RGB")])
    batch = processor(text=texts, images=images, return_tensors="pt",
                      padding=True, truncation=True)
    labels = batch["input_ids"].clone()
    labels[labels == tok.pad_token_id] = -100
    # Do not train on the image placeholder tokens.
    for name in ("image_token_id", "video_token_id"):
        tid = getattr(processor, name, None) or getattr(
            getattr(processor, "tokenizer", None), name, None)
        if isinstance(tid, int):
            labels[labels == tid] = -100
    batch["labels"] = labels
    return batch


model = VLMClass.from_pretrained(
    BASE_MODEL, dtype=torch.bfloat16 if (BF16 and torch.cuda.is_available()) else torch.float32)
model.config.use_cache = False

# Freeze the vision tower; adapt only the language-side projections.
frozen = 0
for n, p in model.named_parameters():
    if "visual" in n or "vision" in n:
        p.requires_grad = False
        frozen += p.numel()
print(f"frozen vision params: {frozen:,}")

lora = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05, bias="none",
                  task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
model = get_peft_model(model, lora)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable={trainable:,} / total={total:,} ({100 * trainable / total:.3f}%)")

args = TrainingArguments(
    output_dir=CKPT_VOL,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    bf16=BF16 and torch.cuda.is_available(),
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="no",
    report_to=["mlflow"],
    remove_unused_columns=False,   # required: we need the raw image column
    dataloader_num_workers=2,
    label_names=["labels"],
)

trainer = Trainer(model=model, args=args, train_dataset=train_raw,
                  eval_dataset=eval_raw, data_collator=collate)

mlflow.set_registry_uri("databricks-uc")
if EXPERIMENT:
    mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run(run_name=f"vlm-{BASE_MODEL.split('/')[-1]}-{RUNTIME_TAG}",
                      nested=bool(os.environ.get("MLFLOW_RUN_ID"))):
    mlflow.set_tags({"runtime": RUNTIME_TAG, "gpu": gpu_name, "workload": "vlm-lora-sft"})
    mlflow.log_params({
        "base_model": BASE_MODEL, "dataset": DATASET, "n_train": len(train_raw),
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
        "effective_batch": BATCH_SIZE * GRAD_ACCUM, "lr": LR, "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA, "max_pixels": MAX_PIXELS, "bf16": BF16,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": transformers.__version__, "gpu_mem_gb": gpu_mem_gb,
        "trainable_params": trainable, "total_params": total,
        "frozen_vision_params": frozen,
        "trainable_pct": round(100 * trainable / total, 4)})

    t_train = time.time()
    out = trainer.train()
    train_seconds = time.time() - t_train
    ev = trainer.evaluate()

    metrics = {
        "train_seconds": train_seconds,
        "data_load_seconds": data_seconds,
        "import_seconds": t_data - t0,
        "total_seconds": time.time() - t0,
        "final_train_loss": out.training_loss,
        "final_eval_loss": ev.get("eval_loss", float("nan")),
        "samples_per_second": len(train_raw) * EPOCHS / train_seconds,
        "peak_gpu_mem_gb": (torch.cuda.max_memory_allocated() / 1e9
                            if torch.cuda.is_available() else 0)}
    mlflow.log_metrics(metrics)

    # Qualitative check: does the tuned model answer in the label vocabulary?
    try:
        model.eval()
        ex = eval_raw[0]
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": QUESTION}]}]
        prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[[ex["image"].convert("RGB")]],
                           return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=12, do_sample=False)
        answer = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:],
                                        skip_special_tokens=True)[0].strip()
        expected = pretty[int(ex["labels"])]
        print(f"SAMPLE expected='{expected}' got='{answer}'")
        mlflow.log_dict({"expected": expected, "generated": answer}, "sample_generation.json")
    except Exception as e:
        print(f"sample generation skipped: {type(e).__name__}: {e}")

    if REGISTER:
        adapter_dir = f"{CKPT_VOL}/adapter_final"
        trainer.model.save_pretrained(adapter_dir)
        processor.save_pretrained(adapter_dir)
        print(f"adapter + processor saved to {adapter_dir}")

    summary = {"runtime": RUNTIME_TAG, "gpu": gpu_name,
               "train_seconds": round(train_seconds, 1),
               "samples_per_s": round(len(train_raw) * EPOCHS / train_seconds, 3),
               "train_loss": round(out.training_loss, 4),
               "eval_loss": round(ev.get("eval_loss", float("nan")), 4),
               "trainable_pct": round(100 * trainable / total, 4),
               "peak_gpu_mem_gb": round(metrics["peak_gpu_mem_gb"], 2),
               "transformers": transformers.__version__}
    print("SUMMARY " + json.dumps(summary))
