# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Classic GPU: multi-GPU with `TorchDistributor`
# MAGIC
# MAGIC **Compute:** all-purpose cluster `gpu-demo-4xa10g` — `g5.12xlarge` (4× A10G), DBR 18.x ML GPU.
# MAGIC
# MAGIC This follows the official TorchDistributor template — **function mode in a notebook**, which
# MAGIC is the only workflow the docs document. (Function mode fails from a `spark_python_task`,
# MAGIC because the module is re-executed in every subprocess and re-triggers the launcher. In a
# MAGIC notebook there is no module to re-execute, so function mode is the natural fit.)
# MAGIC
# MAGIC Doc requirements honoured:
# MAGIC - **All imports inside the training function** — avoids pickling / object-not-found errors.
# MAGIC - `device_id = int(os.environ["LOCAL_RANK"])`.
# MAGIC - `backend = "nccl" if use_gpu else "gloo"`.
# MAGIC - **`NCCL_P2P_DISABLE=1`** is set as a cluster env var: the docs flag a peer-access error on
# MAGIC   the **G5 suite on AWS**, which is exactly this node type.
# MAGIC
# MAGIC Requirements per the docs: Spark 3.4, DBR 13.0 ML or above.
# MAGIC
# MAGIC **Contrast with serverless (notebook 02):** classic has no `UCVolumeDataset`, so the shard
# MAGIC list is partitioned across ranks by hand below. On AI Runtime that sharding is free.

# COMMAND ----------

CATALOG = "classic_stable_qxxm2v_catalog"
SCHEMA = "dl_training"
SHARD_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/data/food101_shards"
LABELS = f"/Volumes/{CATALOG}/{SCHEMA}/data/food101_labels.json"
BASE_MODEL = "google/vit-base-patch16-224"
EXPERIMENT = "/Users/jiayi.wu@databricks.com/gpu-demo-classic"
NUM_GPUS = 4
MAX_SHARDS_PER_RANK = 1

# COMMAND ----------

import os, subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout)
print("NCCL_P2P_DISABLE =", os.environ.get("NCCL_P2P_DISABLE"))

# COMMAND ----------

import mlflow
mlflow.set_experiment(EXPERIMENT)

# COMMAND ----------


def train(lr, use_gpu, num_epochs, batch_size, shard_dir, labels_path, base_model, max_shards):
    """Runs once per process. Every import is inside, per the docs."""
    import os, io, json, time, glob
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    import pyarrow.parquet as pq
    from PIL import Image
    import torchvision.transforms as T
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    backend = "nccl" if use_gpu else "gloo"
    dist.init_process_group(backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    device = int(os.environ["LOCAL_RANK"]) if use_gpu else "cpu"
    if use_gpu:
        torch.cuda.set_device(device)
    is_main = rank == 0
    if is_main:
        print(f"world_size={world} gpu={torch.cuda.get_device_name(device)} "
              f"torch={torch.__version__} cuda={torch.version.cuda}", flush=True)

    label_names = json.load(open(labels_path))
    n_classes = len(label_names)

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

    # No UCVolumeDataset on classic — partition the shard list across ranks manually.
    all_shards = sorted(glob.glob(f"{shard_dir}/*.parquet"))
    my_shards = all_shards[rank::world][:max_shards]
    if is_main:
        print(f"{len(all_shards)} shards total, {len(my_shards)} per rank", flush=True)

    model = AutoModelForImageClassification.from_pretrained(
        base_model, num_labels=n_classes, ignore_mismatched_sizes=True).to(device)
    model = DDP(model, device_ids=[device] if use_gpu else None)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss()

    seen, step, last_loss = 0, 0, float("nan")
    t0 = time.time()
    model.train()
    for epoch in range(num_epochs):
        for shard_path in my_shards:
            rows_all = pq.read_table(shard_path).to_pylist()
            for i in range(0, len(rows_all) - batch_size + 1, batch_size):
                rows = rows_all[i:i + batch_size]
                xs = torch.stack([(tf(to_pil(r["image"])) - mean) / std for r in rows]).to(device)
                ys = torch.tensor([int(r["labels"]) for r in rows]).to(device)
                opt.zero_grad(set_to_none=True)
                loss = lossf(model(pixel_values=xs).logits, ys)
                loss.backward()
                opt.step()
                seen += len(rows); step += 1; last_loss = float(loss.item())
    wall = time.time() - t0

    tot = torch.tensor([seen], dtype=torch.float64, device=device if use_gpu else "cpu")
    dist.all_reduce(tot, op=dist.ReduceOp.SUM)
    global_seen = float(tot.item())

    if is_main:
        print(f"SUMMARY world={world} imgs={global_seen:.0f} "
              f"{global_seen / wall:.1f} img/s loss={last_loss:.4f}", flush=True)

    out = {"rank": rank, "world_size": world, "images": seen,
           "global_images": global_seen, "train_seconds": round(wall, 2),
           "images_per_second": round(global_seen / wall, 2),
           "final_loss": round(last_loss, 4),
           "gpu": torch.cuda.get_device_name(device) if use_gpu else "cpu",
           "torch": torch.__version__}
    dist.destroy_process_group()
    return out

# COMMAND ----------

from pyspark.ml.torch.distributor import TorchDistributor

# local_mode=True -> all 4 processes on the driver node (single-node multi-GPU).
# For multi-node, use local_mode=False on a cluster with GPU workers.
distributor = TorchDistributor(num_processes=NUM_GPUS, local_mode=True, use_gpu=True)

with mlflow.start_run(run_name="classic-torchdistributor-4xA10G") as run:
    mlflow.set_tags({"runtime": "classic-dbr18-g5.12xlarge", "launcher": "TorchDistributor",
                     "mode": "function-mode", "world_size": NUM_GPUS})
    output = distributor.run(train, 1e-4, True, 1, 64, SHARD_DIR, LABELS,
                             BASE_MODEL, MAX_SHARDS_PER_RANK)
    if isinstance(output, dict):
        mlflow.log_metrics({k: v for k, v in output.items() if isinstance(v, (int, float))})

output

# COMMAND ----------

import json
print(json.dumps(output, indent=2, default=str))
