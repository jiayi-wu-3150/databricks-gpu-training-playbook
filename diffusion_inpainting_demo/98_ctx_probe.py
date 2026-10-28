# Databricks notebook source
# MAGIC %md
# MAGIC # 98 — What auth material does a JOB run actually expose?
# MAGIC Reports presence only — never the secret values themselves.

# COMMAND ----------

import json, os

rep = {}
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()


def probe(name, fn):
    try:
        v = fn()
        if v is None:
            rep[name] = "None"
        elif isinstance(v, str):
            # presence + length only; never the value
            rep[name] = f"present(len={len(v)})" if v else "empty-string"
        else:
            rep[name] = f"type={type(v).__name__}"
    except Exception as e:
        rep[name] = f"RAISED {type(e).__name__}: {str(e)[:120]}"


probe("ctx.apiToken().get()", lambda: ctx.apiToken().get())
probe("ctx.browserHostName().get()", lambda: ctx.browserHostName().get())
probe("ctx.apiUrl().get()", lambda: ctx.apiUrl().get())
probe("ctx.notebookPath().get()", lambda: ctx.notebookPath().get())
probe("spark.conf workspaceUrl", lambda: spark.conf.get("spark.databricks.workspaceUrl"))
probe("env DATABRICKS_HOST", lambda: os.environ.get("DATABRICKS_HOST"))
probe("env DATABRICKS_TOKEN", lambda: os.environ.get("DATABRICKS_TOKEN"))
probe("env MLFLOW_TRACKING_URI", lambda: os.environ.get("MLFLOW_TRACKING_URI"))

# Can a plain subprocess reach MLflow without explicit creds?
import subprocess, sys
code = ("import mlflow;from mlflow.tracking import MlflowClient;"
        "mlflow.set_tracking_uri('databricks');"
        "print('SUBPROC_OK', len(MlflowClient().search_experiments(max_results=1)))")
try:
    cp = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=240)
    rep["subprocess_mlflow"] = (cp.stdout.strip()[-200:] or "") + " | ERR:" + (cp.stderr.strip()[-400:] or "")
except Exception as e:
    rep["subprocess_mlflow"] = f"{type(e).__name__}: {e}"

out = "/Volumes/classic_stable_qxxm2v_catalog/dl_training/diffusion/_debug/ctx_probe.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(rep, f, indent=2)
print(json.dumps(rep, indent=2))
