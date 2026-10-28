# Databricks notebook source
# MAGIC %md
# MAGIC # 05a — Batch inference at scale (**SERVERLESS / AI Runtime**)
# MAGIC
# MAGIC Scores the whole `test` split from the **UC registered model** (`@champion`), sharded across
# MAGIC GPUs with `@distributed`.
# MAGIC
# MAGIC ## Why this differs from classic
# MAGIC
# MAGIC **The AI Runtime GPU worker has no Spark**, so `mlflow.pyfunc.spark_udf` — the standard
# MAGIC Databricks batch-scoring pattern — is not available. The options here are:
# MAGIC
# MAGIC | Approach | Notes |
# MAGIC |---|---|
# MAGIC | **`@distributed` + manual shard** (this notebook) | each rank takes `rows[rank::world]`; no extra dependency |
# MAGIC | **Ray Data** | what the AIR cookbook's `ray/data/batch-inference-*` examples use; better for very large inputs and heterogeneous work |
# MAGIC | vLLM | **not applicable** — it serves LLM/VLM text generation. Diffusion has no KV cache for PagedAttention to page. |
# MAGIC
# MAGIC Writes predictions as PNGs to a volume plus a per-row manifest, so `03` metrics could be
# MAGIC recomputed over the full split rather than a sample.

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text("catalog", "classic_stable_qxxm2v_catalog", "Catalog")
dbutils.widgets.text("schema", "sd2_playbook", "Schema")
dbutils.widgets.text("uc_model", "sd2_inpaint_serverless", "UC model")
dbutils.widgets.text("alias", "champion", "Alias")
dbutils.widgets.dropdown("accelerator", "GPU_8xH100", ["GPU_1xA10", "GPU_8xH100"], "Accelerator")
dbutils.widgets.text("num_accelerators", "8", "num_accelerators (TOTAL GPUs)")
dbutils.widgets.dropdown("split", "test", ["test", "val"], "Split")
dbutils.widgets.text("num_inference_steps", "20", "Denoising steps")
dbutils.widgets.text("limit", "64", "Max rows (0 = all)")

C = dbutils.widgets.get("catalog"); S = dbutils.widgets.get("schema")
UC_MODEL = f"{C}.{S}.{dbutils.widgets.get('uc_model')}"
ALIAS = dbutils.widgets.get("alias")
ACCEL = dbutils.widgets.get("accelerator")
NGPU = int(dbutils.widgets.get("num_accelerators"))
SPLIT = dbutils.widgets.get("split")
STEPS = int(dbutils.widgets.get("num_inference_steps"))
LIMIT = int(dbutils.widgets.get("limit"))

PQ = f"/Volumes/{C}/{S}/data/parquet/{SPLIT}"
OUT = f"/Volumes/{C}/{S}/data/predictions/{SPLIT}_serverless"
DEBUG = f"/Volumes/{C}/{S}/checkpoints/_debug"
PROMPT = "remove text from image, clean background"

import os, io, glob, json, time
import mlflow
print(f"model : {UC_MODEL}@{ALIAS}\nsplit : {PQ}\nout   : {OUT}")

# COMMAND ----------

# Resolve the UC model to a local artifact dir ONCE on the driver; remote=False keeps every rank
# on this node, so they all read the same staged copy instead of each pulling it from UC.
mlflow.set_registry_uri("databricks-uc")
t0 = time.time()
_root = mlflow.artifacts.download_artifacts(f"models:/{UC_MODEL}@{ALIAS}")
PIPE_DIR = next((r for r, _, fs in os.walk(_root) if "model_index.json" in fs), None)
assert PIPE_DIR, f"no model_index.json under {_root}"
print(f"resolved UC model -> {PIPE_DIR} ({round(time.time()-t0,1)}s)")
os.makedirs(OUT, exist_ok=True)
os.makedirs(DEBUG, exist_ok=True)

# COMMAND ----------

def score_fn(pipe_dir, pq_root, out_dir, steps, limit, prompt, debug):
    """One rank scores rows[rank::world]. All imports inside (cloudpickled)."""
    import os, io, glob, json, time, traceback
    rank = int(os.environ.get("RANK", 0))
    try:
        import torch, numpy as np
        import pyarrow.parquet as pq_
        from PIL import Image
        from diffusers import StableDiffusionInpaintPipeline
        import torch.distributed as dist

        world = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        rows = []
        for f in sorted(glob.glob(f"{pq_root}/*.parquet")):
            rows.extend(pq_.read_table(f).to_pylist())
        if limit:
            rows = rows[:limit]
        mine = rows[rank::world]          # contiguous-free round-robin shard

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            pipe_dir, torch_dtype=torch.float16).to(f"cuda:{local_rank}")
        pipe.set_progress_bar_config(disable=True)

        manifest, t_start = [], time.time()
        for r in mine:
            inp = Image.open(io.BytesIO(r["input_jpg"])).convert("RGB")
            msk = Image.open(io.BytesIO(r["mask_png"])).convert("L")
            t0 = time.time()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                pred = pipe(prompt=prompt, image=inp, mask_image=msk,
                            num_inference_steps=steps, guidance_scale=7.5).images[0]
            dt = time.time() - t0
            # Sequential single-file write -- safe on volume FUSE (no seek).
            p = f"{out_dir}/{r['id']}_pred.png"
            buf = io.BytesIO(); pred.save(buf, format="PNG")
            with open(p, "wb") as fh:
                fh.write(buf.getvalue())
            manifest.append({"id": r["id"], "rank": rank, "path": p, "seconds": round(dt, 2)})
        wall = time.time() - t_start
        return {"rank": rank, "world_size": world, "rows": len(mine),
                "wall_seconds": round(wall, 2),
                "images_per_second": round(len(mine) / wall, 3) if wall else None,
                "gpu": torch.cuda.get_device_name(local_rank), "manifest": manifest}
    except BaseException:
        try:
            os.makedirs(debug, exist_ok=True)
            with open(f"{debug}/batch_err_rank{rank}.txt", "w") as fh:
                fh.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n" + traceback.format_exc())
        except BaseException:
            pass
        raise

# COMMAND ----------

from databricks.air import distributed

mlflow.set_experiment(f"/Users/{spark.sql('select current_user()').first()[0]}/sd2_playbook_batch")
with mlflow.start_run(run_name=f"batch-serverless-{SPLIT}") as run:
    mlflow.log_params({"uc_model": f"{UC_MODEL}@{ALIAS}", "split": SPLIT, "steps": STEPS,
                       "accelerator": ACCEL, "num_accelerators": NGPU, "limit": LIMIT,
                       "sharding": "rows[rank::world]"})
    t0 = time.time()
    runner = distributed(num_accelerators=NGPU, accelerator_type=ACCEL)(score_fn)
    results = runner.distributed(PIPE_DIR, PQ, OUT, STEPS, LIMIT, PROMPT, DEBUG)
    total_wall = round(time.time() - t0, 2)

    rs = results if isinstance(results, list) else [results]
    assert len(rs) == NGPU, f"expected {NGPU} rank results, got {len(rs)}"
    assert sorted(r["rank"] for r in rs) == list(range(NGPU)), "missing ranks"
    rows_total = sum(r["rows"] for r in rs)
    manifest = [m for r in rs for m in r["manifest"]]
    assert len({m["id"] for m in manifest}) == rows_total, "duplicate ids across ranks"

    summary = {"uc_model": f"{UC_MODEL}@{ALIAS}", "split": SPLIT, "steps": STEPS,
               "world_size": len(rs), "rows_scored": rows_total,
               "total_wall_seconds": total_wall,
               "throughput_images_per_second": round(rows_total / total_wall, 3),
               "per_rank_rows": {str(r["rank"]): r["rows"] for r in rs},
               "per_rank_images_per_second": {str(r["rank"]): r["images_per_second"] for r in rs},
               "gpu": rs[0]["gpu"], "output_dir": OUT}
    mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float))})

with open(f"{OUT}/_manifest.json", "w") as fh:
    json.dump({"summary": summary, "rows": manifest}, fh, indent=2)
with open(f"/Volumes/{C}/{S}/data/batch_serverless_{SPLIT}.json", "w") as fh:
    json.dump(summary, fh, indent=2)
print(json.dumps(summary, indent=2))
