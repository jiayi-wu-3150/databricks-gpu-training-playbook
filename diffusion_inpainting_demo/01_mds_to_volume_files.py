# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Stage the dataset as volume files for `UCVolumeDataset`
# MAGIC
# MAGIC **Same model, same data, different format.** Per the AI Runtime data-loading docs
# MAGIC (`ai-runtime/develop-workloads/dataloading`), unstructured data in a UC Volume should be
# MAGIC read with **`UCVolumeDataset` + `databricks.air.data.DataLoader`** — the documented
# MAGIC optimized path. `UCVolumeDataset` yields *cached local file paths*, and you wrap it in a
# MAGIC second `IterableDataset` that decodes. Mosaic Streaming is not mentioned in those docs at
# MAGIC all; it is the classic-side recommendation.
# MAGIC
# MAGIC An inpainting sample is three pixel-aligned images (`input`/`target`/`mask`), so each
# MAGIC sample becomes **one `.npz`** holding the three encoded buffers. One file per sample is
# MAGIC what makes `UCVolumeDataset`'s per-`(rank, worker)` partitioning meaningful — and keeping
# MAGIC the triplet in a single file is what keeps the three images aligned, since separate
# MAGIC `UCVolumeDataset`s would shard independently.
# MAGIC
# MAGIC | | Classic | Serverless (AI Runtime) |
# MAGIC |---|---|---|
# MAGIC | Format | MDS shards | one `.npz` per sample |
# MAGIC | Loader | `StreamingDataset` | `UCVolumeDataset` + `air DataLoader` |
# MAGIC | Needs `mosaicml-streaming` | yes → `transformers<5` → `diffusers<=0.39` | **no** → `transformers 5.x` + `diffusers 0.42` |
# MAGIC | Model / data | **identical** | **identical** |
# MAGIC
# MAGIC > **Docs caveat we are knowingly ignoring at this scale:** the page advises consolidating
# MAGIC > into files of ~1-10 GiB and warns that many small files read straight from a volume are
# MAGIC > slower. Our smoke set is 320 files of ~1 MB. Fine for validating the path; for a real
# MAGIC > OTR run (74k samples) you would batch many samples per file or pre-copy to local disk.
# MAGIC
# MAGIC Run on the classic GPU cluster (it has `mosaicml-streaming` to read the MDS side).

# COMMAND ----------

# MAGIC %pip install mosaicml-streaming~=0.13.0

# COMMAND ----------

CATALOG = "classic_stable_qxxm2v_catalog"
SCHEMA = "dl_training"
MDS_IN = f"/Volumes/{CATALOG}/{SCHEMA}/data/OTR_like_MDS"
FILES_OUT = f"/Volumes/{CATALOG}/{SCHEMA}/data/OTR_like_files"
SPLITS = ["train", "OTR_easy", "OTR_hard"]

# COMMAND ----------

import os, io, shutil, glob
import numpy as np
from streaming import StreamingDataset


def encode(img, mode, fmt):
    b = io.BytesIO()
    img.convert(mode).save(b, format=fmt)
    return np.frombuffer(b.getvalue(), dtype=np.uint8)


for split in SPLITS:
    src, dst = f"{MDS_IN}/{split}", f"{FILES_OUT}/{split}"
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst, exist_ok=True)
    ds = StreamingDataset(local=src, shuffle=False)
    for i in range(len(ds)):
        s = ds[i]
        # np.savez writes a ZIP, which needs random-access/seek writes. Writing one
        # directly to a UC Volume FUSE path fails with `OSError: [Errno 5] Input/output
        # error`. Build it in memory and commit with a SINGLE sequential write instead.
        _buf = io.BytesIO()
        np.savez(
            _buf,
            input=encode(s["input"], "RGB", "JPEG"),
            target=encode(s["target"], "RGB", "JPEG"),
            mask=encode(s["mask"], "L", "PNG"),
        )
        with open(os.path.join(dst, f"{split}_{i:06d}.npz"), "wb") as _f:
            _f.write(_buf.getvalue())
    print(f"{split:10s} {len(ds):5d} samples -> {dst}")

# COMMAND ----------

# Verify by decoding one sample exactly as the training wrapper will.
from PIL import Image

for split in SPLITS:
    files = sorted(glob.glob(f"{FILES_OUT}/{split}/*.npz"))
    with np.load(files[0]) as z:
        inp = Image.open(io.BytesIO(z["input"].tobytes()))
        tgt = Image.open(io.BytesIO(z["target"].tobytes()))
        msk = Image.open(io.BytesIO(z["mask"].tobytes()))
    print(f"{split:10s} files={len(files):5d} input={inp.size}{inp.mode} "
          f"target={tgt.size}{tgt.mode} mask={msk.size}{msk.mode} extrema={msk.getextrema()}")
