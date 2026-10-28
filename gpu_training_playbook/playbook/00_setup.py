"""Stage a small image-classification dataset into a UC Volume as parquet.

Run once per workspace. Subsequent training runs read from the Volume, which is the
data path Databricks documents for AI Runtime (write to Volume, then load parquet).
"""
import os
import sys

CATALOG = os.environ.get("CATALOG", "classic_stable_qxxm2v_catalog")
SCHEMA = os.environ.get("SCHEMA", "dl_training")
DATASET = os.environ.get("DATASET", "beans")
DATASET_ID = os.environ.get("DATASET_ID", "AI-Lab-Makerere/beans")
# HF datasets disagree on column names (image/labels vs img/label). Normalize to
# image/labels so downstream training templates need no per-dataset changes.
IMAGE_COL = os.environ.get("IMAGE_COL", "image")
LABEL_COL = os.environ.get("LABEL_COL", "labels")
N_TRAIN = int(os.environ.get("N_TRAIN", "0"))  # 0 = all
VOL = f"/Volumes/{CATALOG}/{SCHEMA}/data/{DATASET}"

print("=== environment probe ===")
print("python:", sys.version.split()[0])
for mod in ("torch", "torchvision", "transformers", "datasets", "mlflow"):
    try:
        m = __import__(mod)
        print(f"  {mod}: {getattr(m, '__version__', 'unknown')}")
    except ImportError:
        print(f"  {mod}: NOT INSTALLED")

print(f"=== staging {DATASET_ID} -> {VOL} ===")
from datasets import load_dataset

ds = load_dataset(DATASET_ID)
print("raw splits:", {k: len(v) for k, v in ds.items()})
label_feature = ds[list(ds.keys())[0]].features[LABEL_COL]
rename = {}
if IMAGE_COL != "image":
    rename[IMAGE_COL] = "image"
if LABEL_COL != "labels":
    rename[LABEL_COL] = "labels"
if rename:
    ds = ds.rename_columns(rename)
    print("renamed:", rename)
if N_TRAIN and "train" in ds:
    ds["train"] = ds["train"].shuffle(seed=42).select(range(min(N_TRAIN, len(ds["train"]))))
    print(f"subset train -> {len(ds['train'])}")
print("splits:", {k: len(v) for k, v in ds.items()})
os.makedirs(VOL, exist_ok=True)
for split, d in ds.items():
    out = f"{VOL}/{split}.parquet"
    d.to_parquet(out)
    print(f"  wrote {out} ({len(d)} rows)")

# Persist label names: parquet loses the ClassLabel feature, so training needs this.
import json as _json
names = label_feature.names
with open(f"{VOL}/labels.json", "w") as fh:
    _json.dump(names, fh)
print("labels:", names, "-> labels.json")
print("=== setup complete ===")
