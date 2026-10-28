# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Classic GPU: `DeepspeedTorchDistributor`
# MAGIC
# MAGIC **Compute:** all-purpose cluster `gpu-demo-4xa10g` — `g5.12xlarge` (4× A10G), DBR 18.x ML GPU.
# MAGIC
# MAGIC The third documented classic option, and the one the docs recommend when the model is
# MAGIC **memory**-constrained rather than compute-constrained ("low GPU memory", "large model
# MAGIC training"). It is built on top of `TorchDistributor` and needs **DBR 14.0 ML or above**.
# MAGIC
# MAGIC Watch the argument names — `DeepspeedTorchDistributor` is **camelCase**, unlike
# MAGIC `TorchDistributor`'s snake_case:
# MAGIC
# MAGIC ```python
# MAGIC DeepspeedTorchDistributor(numGpus=1, nnodes=1, localMode=True, useGpu=True, deepspeedConfig=None)
# MAGIC ```
# MAGIC
# MAGIC `numGpus` is **per node**, and internally `num_processes = numGpus * nnodes`.
# MAGIC `deepspeedConfig` accepts a path **or** a dict (a dict is written out for you).

# COMMAND ----------

# MAGIC %md
# MAGIC **No pip install needed.** DeepSpeed is preinstalled on DBR 18.x ML GPU — the import
# MAGIC resolves to `/databricks/python/lib/python3.12/site-packages/deepspeed`. (Verified from a
# MAGIC traceback: it loaded from the DBR system path, not an ephemeral pip env.)

# COMMAND ----------

CATALOG = "classic_stable_qxxm2v_catalog"
SCHEMA = "dl_training"
SHARD_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/data/food101_shards"
LABELS = f"/Volumes/{CATALOG}/{SCHEMA}/data/food101_labels.json"
BASE_MODEL = "google/vit-base-patch16-224"
EXPERIMENT = "/Users/jiayi.wu@databricks.com/gpu-demo-classic"
NUM_GPUS = 4
MAX_SHARDS_PER_RANK = 1
MICRO_BATCH = 32

# COMMAND ----------

# ZeRO stage 2: optimizer state + gradients sharded across the 4 ranks. This is the knob that
# makes DeepSpeed worth reaching for over plain DDP — DDP replicates optimizer state on every GPU.
ds_config = {
    "train_micro_batch_size_per_gpu": MICRO_BATCH,
    "gradient_accumulation_steps": 1,
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "contiguous_gradients": True,
        "overlap_comm": True,
    },
    # `torch_adam: True` is REQUIRED here. Without it DeepSpeed builds **FusedAdam**, which
    # JIT-compiles `fused_adam.so` at init — and that fails on DBR 18 ML GPU with
    # `ImportError: .../py312_cu129/fused_adam/fused_adam.so: cannot open shared object file`
    # (all 4 ranks race to compile into the same torch_extensions cache). The flag makes
    # DeepSpeed use `torch.optim.AdamW` instead, which needs no compiled op.
    "optimizer": {"type": "AdamW", "params": {"lr": 1e-4, "torch_adam": True}},
    "gradient_clipping": 1.0,
    "steps_per_print": 20,
}
ds_config

# COMMAND ----------

import mlflow
mlflow.set_experiment(EXPERIMENT)

# COMMAND ----------


def train_ds(num_epochs, batch_size, shard_dir, labels_path, base_model, max_shards, config):
    """Runs once per process under the DeepSpeed launcher. All imports inside."""
    import os, io, json, time, glob
    import torch
    import torch.distributed as dist
    import deepspeed
    import pyarrow.parquet as pq
    from PIL import Image
    import torchvision.transforms as T
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    deepspeed.init_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank, world = dist.get_rank(), dist.get_world_size()
    is_main = rank == 0
    if is_main:
        print(f"world_size={world} gpu={torch.cuda.get_device_name(local_rank)} "
              f"deepspeed={deepspeed.__version__} torch={torch.__version__}", flush=True)

    n_classes = len(json.load(open(labels_path)))
    processor = AutoImageProcessor.from_pretrained(base_model)
    size = processor.size.get("height", 224)
    mean = torch.tensor(processor.image_mean).view(3, 1, 1)
    std = torch.tensor(processor.image_std).view(3, 1, 1)
    tf = T.Compose([T.Resize((size, size)), T.ToTensor()])

    def to_pil(v):
        if isinstance(v, dict):
            v = v.get("bytes") or v.get("path")
        if isinstance(v, (bytes, bytearray)):
            return Image.open(io.BytesIO(v)).convert("RGB")
        return v.convert("RGB")

    all_shards = sorted(glob.glob(f"{shard_dir}/*.parquet"))
    my_shards = all_shards[rank::world][:max_shards]

    model = AutoModelForImageClassification.from_pretrained(
        base_model, num_labels=n_classes, ignore_mismatched_sizes=True)

    # DeepSpeed owns the optimizer (from `config`) and wraps the model in an engine.
    engine, _, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), config=config)
    lossf = torch.nn.CrossEntropyLoss()

    seen, step, last_loss = 0, 0, float("nan")
    t0 = time.time()
    engine.train()
    for epoch in range(num_epochs):
        for shard_path in my_shards:
            rows_all = pq.read_table(shard_path).to_pylist()
            for i in range(0, len(rows_all) - batch_size + 1, batch_size):
                rows = rows_all[i:i + batch_size]
                xs = torch.stack([(tf(to_pil(r["image"])) - mean) / std
                                  for r in rows]).to(engine.device).to(torch.bfloat16)
                ys = torch.tensor([int(r["labels"]) for r in rows]).to(engine.device)
                loss = lossf(engine(pixel_values=xs).logits.float(), ys)
                engine.backward(loss)   # DeepSpeed, not loss.backward()
                engine.step()           # DeepSpeed, not optimizer.step()
                seen += len(rows); step += 1; last_loss = float(loss.item())
    wall = time.time() - t0

    tot = torch.tensor([seen], dtype=torch.float64, device=local_rank)
    dist.all_reduce(tot, op=dist.ReduceOp.SUM)
    global_seen = float(tot.item())

    if is_main:
        print(f"SUMMARY world={world} imgs={global_seen:.0f} "
              f"{global_seen / wall:.1f} img/s loss={last_loss:.4f} "
              f"zero_stage={config['zero_optimization']['stage']}", flush=True)

    return {"rank": rank, "world_size": world, "images": seen,
            "global_images": global_seen, "train_seconds": round(wall, 2),
            "images_per_second": round(global_seen / wall, 2),
            "final_loss": round(last_loss, 4),
            "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "zero_stage": config["zero_optimization"]["stage"],
            "deepspeed": deepspeed.__version__}

# COMMAND ----------

from pyspark.ml.deepspeed.deepspeed_distributor import DeepspeedTorchDistributor

distributor = DeepspeedTorchDistributor(
    numGpus=NUM_GPUS,        # per node
    nnodes=1,
    localMode=True,          # driver-local; False spreads across Spark workers
    useGpu=True,
    deepspeedConfig=ds_config,
)

with mlflow.start_run(run_name="classic-deepspeed-zero2-4xA10G") as run:
    mlflow.set_tags({"runtime": "classic-dbr18-g5.12xlarge",
                     "launcher": "DeepspeedTorchDistributor",
                     "zero_stage": ds_config["zero_optimization"]["stage"],
                     "world_size": NUM_GPUS})
    output = distributor.run(train_ds, 1, MICRO_BATCH, SHARD_DIR, LABELS,
                             BASE_MODEL, MAX_SHARDS_PER_RANK, ds_config)
    if isinstance(output, dict):
        mlflow.log_metrics({k: v for k, v in output.items() if isinstance(v, (int, float))})

output
