# Databricks notebook source
# MAGIC %md
# MAGIC # 05b — Batch inference at scale (**CLASSIC GPU**, Spark)
# MAGIC
# MAGIC Scores the `test` split from the **UC registered model** (`@champion`) using
# MAGIC **`mlflow.pyfunc.spark_udf`** — the standard Databricks batch-scoring pattern, distributed
# MAGIC over a Spark DataFrame.
# MAGIC
# MAGIC ## Why this differs from serverless
# MAGIC
# MAGIC Classic **has Spark on the GPU node**, so the model becomes a UDF and Spark handles
# MAGIC partitioning, retries and spill. The AI Runtime worker has no Spark, which is why `05a`
# MAGIC shards manually under `@distributed` instead.
# MAGIC
# MAGIC | | Classic (`05b`) | Serverless (`05a`) |
# MAGIC |---|---|---|
# MAGIC | Mechanism | `mlflow.pyfunc.spark_udf` | `@distributed` + `rows[rank::world]` |
# MAGIC | Parallelism | Spark partitions | one process per GPU |
# MAGIC | Retries / spill | Spark handles it | you handle it |
# MAGIC
# MAGIC > **Single-node caveat.** This cluster is single-node (`local[*]`), so Spark parallelism is
# MAGIC > CPU-thread based and all tasks share the 4 GPUs. `spark_udf` is the right *shape* and is
# MAGIC > what scales on a multi-node GPU cluster; on one node, set
# MAGIC > `spark.task.resource.gpu.amount` and size partitions deliberately or tasks will contend.

# COMMAND ----------

# MAGIC %pip install 'diffusers==0.39.0' accelerate

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text("catalog", "classic_stable_qxxm2v_catalog", "Catalog")
dbutils.widgets.text("schema", "sd2_playbook", "Schema")
dbutils.widgets.text("uc_model", "sd2_inpaint_classic", "UC model")
dbutils.widgets.text("alias", "champion", "Alias")
dbutils.widgets.dropdown("split", "test", ["test", "val"], "Split")
dbutils.widgets.text("num_inference_steps", "20", "Denoising steps")
dbutils.widgets.text("limit", "16", "Max rows (0 = all)")
dbutils.widgets.text("partitions", "4", "Spark partitions")

C = dbutils.widgets.get("catalog"); S = dbutils.widgets.get("schema")
UC_MODEL = f"{C}.{S}.{dbutils.widgets.get('uc_model')}"
ALIAS = dbutils.widgets.get("alias")
SPLIT = dbutils.widgets.get("split")
STEPS = int(dbutils.widgets.get("num_inference_steps"))
LIMIT = int(dbutils.widgets.get("limit"))
PARTS = int(dbutils.widgets.get("partitions"))

PQ = f"/Volumes/{C}/{S}/data/parquet/{SPLIT}"
OUT = f"/Volumes/{C}/{S}/data/predictions/{SPLIT}_classic"
PROMPT = "remove text from image, clean background"

import os, io, glob, json, time, base64
import mlflow, pandas as pd
os.makedirs(OUT, exist_ok=True)
print(f"model : {UC_MODEL}@{ALIAS}\nsplit : {PQ}\nout   : {OUT}")

# COMMAND ----------

# Build the input DataFrame: the PyFunc signature takes base64 strings.
import pyarrow.parquet as pq
rows = []
for f in sorted(glob.glob(f"{PQ}/*.parquet")):
    rows.extend(pq.read_table(f).to_pylist())
if LIMIT:
    rows = rows[:LIMIT]
pdf = pd.DataFrame([{"id": r["id"],
                     "image": base64.b64encode(r["input_jpg"]).decode(),
                     "mask": base64.b64encode(r["mask_png"]).decode(),
                     "prompt": PROMPT} for r in rows])
sdf = spark.createDataFrame(pdf).repartition(PARTS)
print(f"{len(pdf)} rows across {sdf.rdd.getNumPartitions()} partitions")

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Users/{spark.sql('select current_user()').first()[0]}/sd2_playbook_batch")

with mlflow.start_run(run_name=f"batch-classic-{SPLIT}") as run:
    mlflow.log_params({"uc_model": f"{UC_MODEL}@{ALIAS}", "split": SPLIT, "steps": STEPS,
                       "limit": LIMIT, "partitions": PARTS, "mechanism": "mlflow.pyfunc.spark_udf"})
    t0 = time.time()
    # spark_udf rebuilds the model env from the version's pip_requirements -- which is exactly why
    # 04b pinned diffusers==0.39.0 there rather than leaving it to chance.
    predict = mlflow.pyfunc.spark_udf(
        spark, model_uri=f"models:/{UC_MODEL}@{ALIAS}", result_type="string",
        params={"num_inference_steps": STEPS, "guidance_scale": 7.5})
    scored = sdf.withColumn("pred_b64", predict("image", "mask", "prompt"))
    collected = scored.select("id", "pred_b64").collect()
    wall = round(time.time() - t0, 2)

    manifest = []
    for row in collected:
        p = f"{OUT}/{row['id']}_pred.png"
        with open(p, "wb") as fh:
            fh.write(base64.b64decode(row["pred_b64"]))
        manifest.append({"id": row["id"], "path": p})

    summary = {"uc_model": f"{UC_MODEL}@{ALIAS}", "split": SPLIT, "steps": STEPS,
               "mechanism": "mlflow.pyfunc.spark_udf", "partitions": PARTS,
               "rows_scored": len(manifest), "total_wall_seconds": wall,
               "throughput_images_per_second": round(len(manifest) / wall, 3) if wall else None,
               "output_dir": OUT}
    mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float))})

with open(f"{OUT}/_manifest.json", "w") as fh:
    json.dump({"summary": summary, "rows": manifest}, fh, indent=2)
with open(f"/Volumes/{C}/{S}/data/batch_classic_{SPLIT}.json", "w") as fh:
    json.dump(summary, fh, indent=2)
print(json.dumps(summary, indent=2))
