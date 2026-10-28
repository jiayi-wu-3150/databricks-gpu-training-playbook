# Databricks notebook source
# MAGIC %md
# MAGIC # 99 — Dependency probe for the diffusion stack
# MAGIC Installs the same pins as the fine-tuning notebooks, **without `-q`**, so pip's
# MAGIC resolver output (downgrades, conflicts, "incompatible" warnings) is visible.

# COMMAND ----------

# MAGIC %pip install 'diffusers==0.42.0' 'transformers>=5' 'huggingface_hub>=1.32,<3' accelerate

# COMMAND ----------

# MAGIC %pip install pillow
# MAGIC # mosaicml-streaming and datasets~=4.8 REMOVED: streaming requires transformers<5,
# MAGIC # which would force the failure this probe is meant to test for.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import importlib.metadata as md
for p in ["huggingface_hub", "diffusers", "transformers", "sentence-transformers", "tokenizers", "evaluate", "accelerate",
          "mosaicml-streaming", "torch", "safetensors", "bitsandbytes"]:
    try:
        print(f"{p:22s} {md.version(p)}")
    except Exception as e:
        print(f"{p:22s} MISSING ({type(e).__name__})")

# COMMAND ----------

import huggingface_hub, inspect
print("hub file:", huggingface_hub.__file__)
print("hub version:", huggingface_hub.__version__)
print("has resolve_revision:", hasattr(huggingface_hub, "resolve_revision"))

# COMMAND ----------

try:
    import diffusers
    print("diffusers OK:", diffusers.__version__)
    from diffusers import StableDiffusionInpaintPipeline
    print("pipeline import OK")
except Exception as e:
    import traceback
    print("DIFFUSERS IMPORT FAILED:", type(e).__name__, e)
    traceback.print_exc()


# COMMAND ----------

# Write everything to the volume — a Job run does not expose notebook stdout anywhere
# readable via the API, so self-persisting the report is the only way to see it headlessly.
import io, json, importlib.metadata as md, traceback, os

report = {}
for pkg in ["huggingface_hub", "diffusers", "transformers", "datasets", "accelerate",
            "mosaicml-streaming", "torch", "safetensors", "bitsandbytes"]:
    try:
        report[pkg] = md.version(pkg)
    except Exception as e:
        report[pkg] = f"MISSING ({type(e).__name__})"

try:
    import huggingface_hub
    report["_hub_file"] = huggingface_hub.__file__
    report["_has_resolve_revision"] = hasattr(huggingface_hub, "resolve_revision")
except Exception as e:
    report["_hub_error"] = f"{type(e).__name__}: {e}"

try:
    import diffusers
    from diffusers import StableDiffusionInpaintPipeline
    report["_diffusers_import"] = "OK " + diffusers.__version__
except Exception as e:
    report["_diffusers_import"] = f"FAILED {type(e).__name__}: {e}"
    report["_diffusers_trace"] = traceback.format_exc()[-2000:]

import subprocess, sys as _sys
# NOTE: the `pip check` LINE MAGIC exits 1 on conflicts, which aborts the whole
# notebook before anything is written. Run it captured instead, so the conflict list
# lands in the report rather than killing the probe.
for label, cmd in [("pip_check", [_sys.executable, "-m", "pip", "check"]),
                   ("pip_show_hub", [_sys.executable, "-m", "pip", "show", "huggingface_hub"])]:
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        report[label] = (cp.stdout + cp.stderr).strip()[:3000]
    except Exception as e:
        report[label] = f"{type(e).__name__}: {e}"

out = "/Volumes/classic_stable_qxxm2v_catalog/dl_training/diffusion/_debug/env_probe_CLASSIC_force42.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
