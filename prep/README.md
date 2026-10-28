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

## Why the re-shard step exists

`stage_food101.py` writes **one** `train.parquet`. A single parquet file cannot be split across
`(rank, worker)` streams, so distributed training would read the whole file on every rank.
`reshard_food101.py` streams row batches into `N_SHARDS = 32` files; because it flushes on batch
boundaries it produced **27** in practice, which is what the measured runs used.

27 shards comfortably exceeds the `world_size × num_workers` floor (8 × 2 = 16) that
`UCVolumeDataset` needs — see the shard-count note in the top-level README.

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
