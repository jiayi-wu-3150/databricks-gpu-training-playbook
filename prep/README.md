# `prep/` — staging the source images, before step 00

Step `00_prepare_data.py` does **not** invent its images. It reads real photographs from a UC
volume and draws synthetic text overlays onto them. These two scripts put those photographs there.

Run once per workspace. After that, steps 00–06 are self-contained.

```
ethz/food101  (Hugging Face, 20,000 train images)
     │
     │  prep/stage_food101.py          driven by prep/stage_food101.yaml (AIR job, GPU_1xA10)
     ▼
/Volumes/{catalog}/dl_training/data/food101/
     ├── train.parquet
     ├── validation.parquet
     └── labels.json
     │
     │  prep/reshard_food101.py        one monolithic parquet cannot be partitioned
     ▼
/Volumes/{catalog}/dl_training/data/food101_shards/
     └── shard_0000.parquet ... shard_0026.parquet
     │
     │  00_prepare_data.py             draws text overlays + masks
     ▼
/Volumes/{catalog}/sd2_playbook/data/{raw,parquet}/
```

## What step 00 actually receives

Verified against `shard_0000.parquet` (768 rows, 3 row groups):

```
image:  struct<bytes: binary, path: string>
labels: int64
```

One row: `image.bytes` = 33,871 bytes of JPEG, `image.path` = `'833480.jpg'`, `labels` = `74`.
The Hugging Face `Image` feature survives as that `{bytes, path}` struct, which is why step 00
unwraps a dict before decoding:

```python
v = rec.get("image")
if isinstance(v, dict):
    v = v.get("bytes")
yield Image.open(io.BytesIO(v))
```

**`labels` is never read.** food101 is a classification dataset, but inpainting has no class
target — step 00 wants pixels only. So the input contract is just *"a parquet column named `image`
holding JPEG/PNG bytes, either raw or as a `{bytes, path}` struct."* Any corpus matching that
works; point the `source_images` widget at it.

## Why the re-shard step exists

`stage_food101.py` writes **one** `train.parquet`. The re-shard exists because this corpus was
originally used for distributed **ViT** training, which read these files directly — and a single
parquet file cannot be split across `(rank, worker)` streams, so every rank would read the whole
file. `reshard_food101.py` streams row batches into `N_SHARDS = 32` files; because it flushes on
batch boundaries it produced **27** in practice.

**For this playbook the sharding is incidental.** Step 00 reads these files single-process with
`pyarrow` and consumes only the first 384 images (256 train + 64 val + 64 test) out of 20,000 —
so it never opens past `shard_0000`, which holds 768 rows on its own. Any single parquet file with
≥384 usable images would do.

The `world_size × num_workers` shard floor belongs one stage later and applies to *different*
files: step 00 **writes** 16 train shards to `{catalog}.sd2_playbook` `data/parquet/train/`, and
those are what `UCVolumeDataset` shards across streams in 02a. Do not confuse the 27 source shards
here with those 16 output shards.

## Notes

- `stage_food101.py` is dataset-agnostic, driven by env vars (`DATASET_ID`, `IMAGE_COL`,
  `LABEL_COL`, `N_TRAIN`). The YAML pins it to `ethz/food101` with `N_TRAIN=20000`.
- It normalises HF's inconsistent column naming (`image`/`img`, `labels`/`label`) to
  `image`/`labels`, and persists `labels.json` separately because parquet drops the `ClassLabel`
  feature.
- These scripts stage into the **`dl_training`** schema, deliberately separate from
  `sd2_playbook`, so the source corpus survives a `DROP SCHEMA ... sd2_playbook CASCADE` teardown.
- `food101` images are licensed for research use. Check the
  [dataset card](https://huggingface.co/datasets/ethz/food101) before using them for anything else.
