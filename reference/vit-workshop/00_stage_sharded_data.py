# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Stage sharded Food-101 data
# MAGIC
# MAGIC Re-shards the existing single `train.parquet` into **32 parquet shards** under a new volume
# MAGIC directory. This matters for the serverless notebooks: `databricks.air.data.UCVolumeDataset`
# MAGIC is an `IterableDataset` that streams *files* and automatically partitions the file list
# MAGIC across `(rank, worker)` pairs. One monolithic parquet file cannot be partitioned, so a
# MAGIC sharded layout is what actually exercises the API.
# MAGIC
# MAGIC Run once, on any compute with UC access (the classic GPU cluster is fine).

# COMMAND ----------

CATALOG = "classic_stable_qxxm2v_catalog"
SCHEMA = "dl_training"
SRC = f"/Volumes/{CATALOG}/{SCHEMA}/data/food101"
DST = f"/Volumes/{CATALOG}/{SCHEMA}/data/food101_shards"
N_SHARDS = 32

# COMMAND ----------

import os, json, shutil
import pyarrow.parquet as pq
import pyarrow as pa

os.makedirs(DST, exist_ok=True)

pf = pq.ParquetFile(f"{SRC}/train.parquet")
rows = pf.metadata.num_rows
print("source rows:", rows, "columns:", pf.schema_arrow.names)

# COMMAND ----------

# Stream row batches into N_SHARDS files so we never hold the whole dataset in memory.
per = (rows + N_SHARDS - 1) // N_SHARDS
writer, written, cur, idx = None, [], 0, 0

for batch in pf.iter_batches(batch_size=256):
    if writer is None:
        path = f"{DST}/shard_{idx:04d}.parquet"
        writer = pq.ParquetWriter(path, pf.schema_arrow)
        cur = 0
    writer.write_batch(batch)
    cur += batch.num_rows
    if cur >= per:
        writer.close()
        written.append((path, cur))
        writer, idx = None, idx + 1

if writer is not None:
    writer.close()
    written.append((path, cur))

print(f"wrote {len(written)} shards to {DST}")
for pth, n in written[:3]:
    print("  ", pth, n, "rows")
print("   ...")

# COMMAND ----------

# Labels travel separately — copy alongside so the training notebooks need only one path.
shutil.copyfile(f"{SRC}/labels.json", f"{DST}/../food101_labels.json")
labels = json.load(open(f"{SRC}/labels.json"))
print("n_classes:", len(labels))
print("shard files visible:", len([f for f in os.listdir(DST) if f.endswith(".parquet")]))
