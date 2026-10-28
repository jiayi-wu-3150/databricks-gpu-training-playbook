# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Stage the dataset as Parquet (alternative to the per-sample `.npz` layout)
# MAGIC
# MAGIC Third format for the *same* data. Motivation: all three AI Runtime sources say
# MAGIC **consolidate into large files** (docs: ~1-10 GiB each; Slack field guidance:
# MAGIC "consolidate data into large files… monitor data-loader throughput"). The `.npz`
# MAGIC layout from `01_` is one file per sample — the opposite — and it also tripped
# MAGIC `OSError: [Errno 5]` because a `.npz` is a ZIP needing seek-writes.
# MAGIC
# MAGIC Parquet fixes both: many samples per file, and columnar sequential writes that a
# MAGIC volume FUSE mount handles natively.
# MAGIC
# MAGIC | Format | Loader | Files | FUSE write |
# MAGIC |---|---|---|---|
# MAGIC | MDS (`00_`) | `StreamingDataset` (classic) | shards | fine |
# MAGIC | `.npz` (`01_`) | `UCVolumeDataset` (serverless) | 1/sample | needed in-memory zip workaround |
# MAGIC | **Parquet (this)** | `load_dataset("parquet", …)` | few, large | fine |
# MAGIC
# MAGIC The three images per sample stay in **one row** (`input_jpg` / `target_jpg` /
# MAGIC `mask_png` binary columns), which is what keeps them pixel-aligned — the same reason
# MAGIC the `.npz` packed them into one file.
# MAGIC
# MAGIC Run on the classic GPU cluster (it has `mosaicml-streaming` to read the MDS side).

# COMMAND ----------

# MAGIC %pip install mosaicml-streaming~=0.13.0

# COMMAND ----------

CATALOG = "classic_stable_qxxm2v_catalog"
SCHEMA = "dl_training"
MDS_IN = f"/Volumes/{CATALOG}/{SCHEMA}/data/OTR_like_MDS"
PQ_OUT = f"/Volumes/{CATALOG}/{SCHEMA}/data/OTR_like_parquet"
SPLITS = ["train", "OTR_easy", "OTR_hard"]
ROWS_PER_FILE = 64   # tiny here (320 samples total); for real OTR target ~1-10 GiB/file

# COMMAND ----------

import os, io, shutil, glob
import pyarrow as pa
import pyarrow.parquet as pq
from streaming import StreamingDataset

SCHEMA_ARROW = pa.schema([
    ("id", pa.string()),
    ("input_jpg", pa.binary()),
    ("target_jpg", pa.binary()),
    ("mask_png", pa.binary()),
])


def encode(img, mode, fmt):
    b = io.BytesIO()
    img.convert(mode).save(b, format=fmt)
    return b.getvalue()


for split in SPLITS:
    src, dst = f"{MDS_IN}/{split}", f"{PQ_OUT}/{split}"
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst, exist_ok=True)
    ds = StreamingDataset(local=src, shuffle=False)

    rows, file_idx, written = [], 0, 0
    def flush(rows, file_idx):
        if not rows:
            return 0
        tbl = pa.Table.from_pylist(rows, schema=SCHEMA_ARROW)
        # Sequential columnar write — no seek games, so this works directly on /Volumes.
        pq.write_table(tbl, f"{dst}/part_{file_idx:04d}.parquet", compression="zstd")
        return len(rows)

    for i in range(len(ds)):
        s = ds[i]
        rows.append({
            "id": s.get("id", f"{split}_{i:06d}"),
            "input_jpg": encode(s["input"], "RGB", "JPEG"),
            "target_jpg": encode(s["target"], "RGB", "JPEG"),
            "mask_png": encode(s["mask"], "L", "PNG"),
        })
        if len(rows) >= ROWS_PER_FILE:
            written += flush(rows, file_idx); rows, file_idx = [], file_idx + 1
    written += flush(rows, file_idx)

    files = sorted(glob.glob(f"{dst}/*.parquet"))
    mb = sum(os.path.getsize(f) for f in files) / 1e6
    print(f"{split:10s} {written:5d} rows -> {len(files)} parquet file(s), {mb:.1f} MB total")

# COMMAND ----------

# Verify by reading back the way the serverless notebook will (docs' parquet path).
from PIL import Image

for split in SPLITS:
    files = sorted(glob.glob(f"{PQ_OUT}/{split}/*.parquet"))
    tbl = pq.read_table(files[0])
    r = tbl.slice(0, 1).to_pylist()[0]
    inp = Image.open(io.BytesIO(r["input_jpg"]))
    msk = Image.open(io.BytesIO(r["mask_png"]))
    print(f"{split:10s} files={len(files)} rows_in_first={tbl.num_rows:4d} cols={tbl.column_names} "
          f"input={inp.size}{inp.mode} mask={msk.size}{msk.mode} extrema={msk.getextrema()}")
