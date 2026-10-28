# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Serverless GPU (AI Runtime): multi-GPU with `@distributed`
# MAGIC
# MAGIC **Compute:** attach to a serverless GPU accelerator. Per the AI Runtime docs, distributed
# MAGIC training **requires `GPU_8xH100` or `GPU_8xB300`** — `1xA10` / `1xH100` are single-GPU only.
# MAGIC
# MAGIC This notebook follows the official `@distributed` template:
# MAGIC
# MAGIC | Doc requirement | How this notebook complies |
# MAGIC |---|---|
# MAGIC | Decorate, then launch with `.distributed(...)` | `train_vit.distributed(num_epochs=1, ...)` |
# MAGIC | All imports **inside** the decorated function | the function is cloudpickled to each rank |
# MAGIC | Load the dataset inside the function | outside objects are pickled with it and can exceed the pickle limit |
# MAGIC | Use `UCVolumeDataset` for UC volume files | streams shard paths, auto-partitioned per `(rank, worker)` |
# MAGIC | Use `UCVolumeWriter` for distributed checkpoints | torch DCP storage backend, **not** `torch.save` |
# MAGIC | Return values are collected from all ranks | each rank returns its metrics dict |
# MAGIC | MLflow run is created by `.distributed()` | we only `log_metric`, we do **not** open our own run |
# MAGIC
# MAGIC `databricks-sdk-air` is preinstalled in GPU environment **v6+**.

# COMMAND ----------

CATALOG = "classic_stable_qxxm2v_catalog"
SCHEMA = "dl_training"
SHARD_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/data/food101_shards"
LABELS = f"/Volumes/{CATALOG}/{SCHEMA}/data/food101_labels.json"
CKPT_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/checkpoints/air_ddp_8xh100"
BASE_MODEL = "google/vit-base-patch16-224"
MAX_SHARDS_PER_RANK = 1   # keep the demo short; raise for a longer run

# COMMAND ----------

from databricks.air import distributed


@distributed(num_accelerators=8, accelerator_type="GPU_8xH100")
def train_vit(num_epochs: int = 1, batch_size: int = 64, lr: float = 1e-4):
    """Runs once per rank. Every import lives here: the function is pickled to each process."""
    import os, io, json, time
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.nn.parallel import DistributedDataParallel as DDP
    import pyarrow.parquet as pq
    from PIL import Image
    import torchvision.transforms as T
    import mlflow
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    from databricks.air.data import UCVolumeDataset, DataLoader, UCVolumeWriter

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    is_main = rank == 0
    if is_main:
        print(f"world_size={world} gpu={torch.cuda.get_device_name(local_rank)} "
              f"torch={torch.__version__} cuda={torch.version.cuda}", flush=True)

    label_names = json.load(open(LABELS))
    n_classes = len(label_names)

    processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
    size = processor.size.get("height", 224)
    mean = torch.tensor(processor.image_mean).view(3, 1, 1)
    std = torch.tensor(processor.image_std).view(3, 1, 1)
    tf = T.Compose([T.Resize((size, size)), T.ToTensor()])

    def to_pil(v):
        # Parquet image columns arrive as {'bytes': ...}, raw bytes, or already-decoded images.
        if isinstance(v, dict):
            v = v.get("bytes") or v.get("path")
        if isinstance(v, (bytes, bytearray)):
            return Image.open(io.BytesIO(v)).convert("RGB")
        return v.convert("RGB")

    # UCVolumeDataset yields LOCAL paths to the volume's files, already split across
    # (rank, worker) streams — no DistributedSampler needed, the file list is the shard unit.
    ds = UCVolumeDataset(SHARD_DIR)
    loader = DataLoader(ds, batch_size=None, num_workers=2)

    model = AutoModelForImageClassification.from_pretrained(
        BASE_MODEL, num_labels=n_classes, ignore_mismatched_sizes=True).to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss()

    seen, step, last_loss = 0, 0, float("nan")
    t0 = time.time()
    model.train()
    for epoch in range(num_epochs):
        shards_done = 0
        for shard_path in loader:
            if shards_done >= MAX_SHARDS_PER_RANK:
                break
            shards_done += 1
            tbl = pq.read_table(shard_path).to_pylist()
            for i in range(0, len(tbl) - batch_size + 1, batch_size):
                rows = tbl[i:i + batch_size]
                xs = torch.stack([(tf(to_pil(r["image"])) - mean) / std for r in rows]).to(local_rank)
                ys = torch.tensor([int(r["labels"]) for r in rows]).to(local_rank)
                opt.zero_grad(set_to_none=True)
                out = model(pixel_values=xs).logits
                loss = lossf(out, ys)
                loss.backward()
                opt.step()
                seen += len(rows); step += 1; last_loss = float(loss.item())
                if is_main and step % 5 == 0:
                    mlflow.log_metric("loss", last_loss, step=step)
    wall = time.time() - t0

    # Aggregate throughput across ranks so rank 0 can report a global number.
    tot = torch.tensor([seen], dtype=torch.float64, device=local_rank)
    dist.all_reduce(tot, op=dist.ReduceOp.SUM)
    global_seen = float(tot.item())

    # Distributed checkpoint straight to the UC volume. Collective: every rank participates.
    dcp.save({"model": model.state_dict()}, storage_writer=UCVolumeWriter(CKPT_DIR))

    if is_main:
        mlflow.log_metrics({"train_seconds": wall,
                            "images_per_second": global_seen / wall,
                            "global_images": global_seen,
                            "world_size": world})
        print(f"SUMMARY world={world} imgs={global_seen:.0f} "
              f"{global_seen / wall:.1f} img/s loss={last_loss:.4f}", flush=True)

    dist.destroy_process_group()
    return {"rank": rank, "world_size": world, "images": seen,
            "train_seconds": round(wall, 2), "final_loss": round(last_loss, 4),
            "gpu": torch.cuda.get_device_name(local_rank), "torch": torch.__version__}

# COMMAND ----------

# Launch. Return values from ALL ranks come back to the caller.
results = train_vit.distributed(num_epochs=1, batch_size=64, lr=1e-4)
results

# COMMAND ----------

import json
print(json.dumps(results if isinstance(results, list) else [results], indent=2, default=str))
