# Databricks notebook source
import importlib.metadata as md, sys, json, os
rep={"python": sys.version.split()[0]}
for p in ["torch","transformers","diffusers","huggingface_hub","accelerate","databricks-sdk-air","numpy","pillow"]:
    try: rep[p]=md.version(p)
    except Exception: rep[p]="absent"
try:
    import databricks.air as a; rep["_air_import"]="OK"
except Exception as e: rep["_air_import"]=f"{type(e).__name__}: {e}"
print(json.dumps(rep, indent=2))
out="/Volumes/classic_stable_qxxm2v_catalog/dl_training/diffusion/_debug/env_v6_probe.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
open(out,"w").write(json.dumps(rep, indent=2))
