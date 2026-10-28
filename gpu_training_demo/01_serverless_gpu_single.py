# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Serverless GPU (AI Runtime): single-GPU fine-tune
# MAGIC
# MAGIC **Compute:** serverless GPU accelerator `GPU_1xA10`, GPU environment **v6+**.
# MAGIC
# MAGIC The cheap half of the serverless story: no `@distributed` here, because the docs restrict
# MAGIC distributed training to `GPU_8xH100` / `GPU_8xB300` — `1xA10` and `1xH100` are single-GPU
# MAGIC only. Notebook 02 is the multi-GPU one.
# MAGIC
# MAGIC What this still demonstrates that classic cannot do:
# MAGIC - `UCVolumeDataset` — streams volume files as local cached paths.
# MAGIC - `databricks.air.data.DataLoader` — MLflow-instrumented iterator timing, and since
# MAGIC   `torch.distributed` is **not** initialised here its `num_workers` default is **48**,
# MAGIC   driving ~50-way node parallelism from one process.

# COMMAND ----------

CATALOG = "classic_stable_qxxm2v_catalog"
SCHEMA = "dl_training"
SHARD_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/data/food101_shards"
LABELS = f"/Volumes/{CATALOG}/{SCHEMA}/data/food101_labels.json"
BASE_MODEL = "google/vit-base-patch16-224"
EXPERIMENT = "/Users/jiayi.wu@databricks.com/gpu-demo-serverless"
BATCH_SIZE = 64
MAX_SHARDS = 2

# COMMAND ----------

import io, json, time
import torch
import pyarrow.parquet as pq
from PIL import Image
import torchvision.transforms as T
import mlflow
from transformers import AutoImageProcessor, AutoModelForImageClassification
from databricks.air.data import UCVolumeDataset, DataLoader

print("gpu:", torch.cuda.get_device_name(0), "| torch:", torch.__version__,
      "| cuda:", torch.version.cuda)

# COMMAND ----------

n_classes = len(json.load(open(LABELS)))
processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
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


# COMMAND ----------

# UCVolumeDataset yields local paths to the volume's files. batch_size=None => one path per item.
ds = UCVolumeDataset(SHARD_DIR)
loader = DataLoader(ds, batch_size=None, num_workers=4)
print("dataset:", SHARD_DIR)

# COMMAND ----------

mlflow.set_experiment(EXPERIMENT)

model = AutoModelForImageClassification.from_pretrained(
    BASE_MODEL, num_labels=n_classes, ignore_mismatched_sizes=True).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
lossf = torch.nn.CrossEntropyLoss()

with mlflow.start_run(run_name="serverless-1xA10-single") as run:
    mlflow.set_tags({"runtime": "air-serverless-1xA10", "launcher": "notebook-single-gpu",
                     "data_api": "UCVolumeDataset"})
    seen, step, last_loss = 0, 0, float("nan")
    t0 = time.time()
    model.train()
    shards = 0
    for shard_path in loader:
        if shards >= MAX_SHARDS:
            break
        shards += 1
        rows_all = pq.read_table(shard_path).to_pylist()
        for i in range(0, len(rows_all) - BATCH_SIZE + 1, BATCH_SIZE):
            rows = rows_all[i:i + BATCH_SIZE]
            xs = torch.stack([(tf(to_pil(r["image"])) - mean) / std for r in rows]).cuda()
            ys = torch.tensor([int(r["labels"]) for r in rows]).cuda()
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(pixel_values=xs).logits, ys)
            loss.backward()
            opt.step()
            seen += len(rows); step += 1; last_loss = float(loss.item())
            if step % 5 == 0:
                mlflow.log_metric("loss", last_loss, step=step)
    wall = time.time() - t0
    metrics = {"train_seconds": wall, "images": seen,
               "images_per_second": seen / wall, "final_loss": last_loss,
               "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
    mlflow.log_metrics(metrics)

print(json.dumps({k: round(v, 4) for k, v in metrics.items()}, indent=2))
