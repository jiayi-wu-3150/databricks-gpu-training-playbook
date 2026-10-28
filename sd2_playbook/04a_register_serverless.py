# Databricks notebook source
# MAGIC %md
# MAGIC # 04a — Register the fine-tuned model in Unity Catalog (serverless)
# MAGIC
# MAGIC Produces a **versioned, self-contained, aliased** model so that batch inference (`05a`) and
# MAGIC serving (`06`) reference `models:/<cat>.<schema>.<name>@champion` instead of a volume path.
# MAGIC
# MAGIC ## Promotion gate
# MAGIC
# MAGIC This notebook reads the **eval report from `03a`** and refuses to promote a checkpoint that
# MAGIC did not beat the base model — unless you override. That is the point of the ablation: a
# MAGIC registry full of `@champion` aliases that were never compared to the base model is worse
# MAGIC than no registry.
# MAGIC
# MAGIC > On this synthetic 10-step playbook the gate **will** report no improvement. That is the
# MAGIC > honest result; `require_improvement=false` lets the mechanism still be demonstrated.
# MAGIC
# MAGIC Register for **governance, not speed** — UC loading is slower than a local checkpoint
# MAGIC (measured: 50.7 s from UC vs 1.2 s local). What you get: versioning, aliases, lineage,
# MAGIC access control, no HF
# MAGIC egress at inference, and Model Serving compatibility.

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text("catalog", "classic_stable_qxxm2v_catalog", "Catalog")
dbutils.widgets.text("schema", "sd2_playbook", "Schema")
dbutils.widgets.text("base_model", "sd2_inpaint_base", "UC base model")
dbutils.widgets.text("base_alias", "base", "Base alias")
dbutils.widgets.text("target_model", "sd2_inpaint_serverless", "UC model to create")
dbutils.widgets.text("alias", "champion", "Alias to set")
dbutils.widgets.text("checkpoint_dir", "", "Checkpoint dir (blank = newest serverless_*)")
dbutils.widgets.text("runtime_label", "serverless", "Runtime label")
dbutils.widgets.dropdown("require_improvement", "false", ["true", "false"], "Block if eval shows no gain")

C = dbutils.widgets.get("catalog"); S = dbutils.widgets.get("schema")
UC_BASE = f"{C}.{S}.{dbutils.widgets.get('base_model')}"
BASE_ALIAS = dbutils.widgets.get("base_alias")
UC_TARGET = f"{C}.{S}.{dbutils.widgets.get('target_model')}"
ALIAS = dbutils.widgets.get("alias")
CKPT_IN = dbutils.widgets.get("checkpoint_dir").strip()
LABEL = dbutils.widgets.get("runtime_label")
REQUIRE = dbutils.widgets.get("require_improvement") == "true"

import os, io, glob, json, time, base64, shutil
import numpy as np, torch, mlflow, pandas as pd
from PIL import Image, ImageDraw

CKPT_ROOT = f"/Volumes/{C}/{S}/checkpoints"
if not CKPT_IN:
    cands = sorted(d for d in glob.glob(f"{CKPT_ROOT}/{LABEL}_*") if os.path.isdir(d))
    assert cands, f"no {LABEL}_* checkpoint under {CKPT_ROOT}"
    CKPT_IN = cands[-1]
UNET_DIR = f"{CKPT_IN}/best/unet"
assert os.path.isdir(UNET_DIR), f"no unet at {UNET_DIR}"
print(f"checkpoint -> {UNET_DIR}\ntarget     -> {UC_TARGET}@{ALIAS}")

# COMMAND ----------

# DBTITLE 1,Promotion gate: did this checkpoint beat the BASE model?
eval_path = f"/Volumes/{C}/{S}/data/eval_{LABEL}_test.json"
gate = {"eval_report_found": os.path.exists(eval_path)}
if gate["eval_report_found"]:
    rep = json.load(open(eval_path))["report"]
    d_psnr = rep.get("delta_finetuned_vs_base_psnr")
    gate.update({"delta_finetuned_vs_base_psnr": d_psnr,
                 "delta_vs_baseline_psnr": rep.get("delta_vs_baseline_psnr"),
                 "verdict": rep.get("VERDICT"), "improved": bool(d_psnr and d_psnr > 0)})
    print(json.dumps(gate, indent=2))
    if not gate["improved"]:
        msg = (f"GATE: this checkpoint did NOT beat the base model "
               f"(delta_finetuned_vs_base_psnr={d_psnr}). Promoting it to @{ALIAS} would alias a "
               f"model no better than the base.")
        if REQUIRE:
            raise SystemExit(msg + " Set require_improvement=false to register anyway.")
        print("WARNING: " + msg + " Continuing because require_improvement=false.")
else:
    print(f"WARNING: no eval report at {eval_path}. Run 03a first — registering unevaluated.")

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
_root = mlflow.artifacts.download_artifacts(f"models:/{UC_BASE}@{BASE_ALIAS}")
BASE_DIR = next((r for r, _, fs in os.walk(_root) if "model_index.json" in fs), None)
assert BASE_DIR, "base pipeline not found"


def _stage_local(vol_path, label):
    """Volume FUSE is slow for big reads (265.8 s vs 1.1 s local for a 3.3 GB UNet)."""
    local = f"/local_disk0/{label}" if os.path.isdir("/local_disk0") else f"/tmp/{label}"
    if os.path.isdir(local):
        return local
    try:
        t0 = time.time(); shutil.copytree(vol_path, local)
        print(f"staged {label} ({round(time.time()-t0,1)}s)"); return local
    except Exception as e:
        print(f"staging failed ({type(e).__name__}); using volume path"); return vol_path


UNET_LOCAL = _stage_local(UNET_DIR, f"unet_{LABEL}")

from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel
LOCAL_PIPE = f"/tmp/sd2_{LABEL}_full"
t0 = time.time()
pipe = StableDiffusionInpaintPipeline.from_pretrained(BASE_DIR, torch_dtype=torch.float16)
pipe.unet = UNet2DConditionModel.from_pretrained(UNET_LOCAL, torch_dtype=torch.float16)
if os.path.exists(LOCAL_PIPE):
    shutil.rmtree(LOCAL_PIPE)
pipe.save_pretrained(LOCAL_PIPE)
size_gb = round(sum(os.path.getsize(os.path.join(r, f))
                    for r, _, fs in os.walk(LOCAL_PIPE) for f in fs) / 1e9, 2)
print(f"assembled fine-tuned pipeline: {size_gb} GB ({round(time.time()-t0,1)}s)")

# COMMAND ----------

class SD2InpaintModel(mlflow.pyfunc.PythonModel):
    """Self-contained fine-tuned inpainting model. base64 PNG in -> base64 PNG out."""

    def load_context(self, context):
        import torch
        from diffusers import StableDiffusionInpaintPipeline
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dt = torch.float16 if dev == "cuda" else torch.float32
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            context.artifacts["pipeline"], torch_dtype=dt).to(dev)
        self.pipe.set_progress_bar_config(disable=True)

    def predict(self, context, model_input, params=None):
        import base64, io
        from PIL import Image
        p = params or {}
        steps, guid = int(p.get("num_inference_steps", 50)), float(p.get("guidance_scale", 7.5))
        out = []
        for _, row in model_input.iterrows():
            img = Image.open(io.BytesIO(base64.b64decode(row["image"]))).convert("RGB")
            msk = Image.open(io.BytesIO(base64.b64decode(row["mask"]))).convert("L")
            res = self.pipe(prompt=row.get("prompt", "remove text from image, clean background"),
                            image=img, mask_image=msk,
                            num_inference_steps=steps, guidance_scale=guid).images[0]
            b = io.BytesIO(); res.save(b, format="PNG")
            out.append(base64.b64encode(b.getvalue()).decode())
        return out

# COMMAND ----------

from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec, ParamSchema, ParamSpec
from mlflow import MlflowClient

# UC REQUIRES an explicit signature (inputs AND outputs); an input_example alone is not enough.
signature = ModelSignature(
    inputs=Schema([ColSpec("string", "image"), ColSpec("string", "mask"), ColSpec("string", "prompt")]),
    outputs=Schema([ColSpec("string")]),
    params=ParamSchema([ParamSpec("num_inference_steps", "integer", 50),
                        ParamSpec("guidance_scale", "double", 7.5)]),
)
_b = io.BytesIO(); Image.new("RGB", (64, 64), (127, 127, 127)).save(_b, format="PNG")
_m = io.BytesIO(); Image.new("L", (64, 64), 0).save(_m, format="PNG")
example = pd.DataFrame([{"image": base64.b64encode(_b.getvalue()).decode(),
                         "mask": base64.b64encode(_m.getvalue()).decode(),
                         "prompt": "remove text from image, clean background"}])

mlflow.set_experiment(f"/Users/{spark.sql('select current_user()').first()[0]}/sd2_playbook_register")
t0 = time.time()
with mlflow.start_run(run_name=f"register-{LABEL}") as run:
    mlflow.log_params({"runtime": LABEL, "checkpoint": UNET_DIR, "uc_base": f"{UC_BASE}@{BASE_ALIAS}",
                       "pipeline_gb": size_gb, **{f"gate_{k}": v for k, v in gate.items()}})
    info = mlflow.pyfunc.log_model(
        name=f"sd2_inpaint_{LABEL}",
        python_model=SD2InpaintModel(),
        artifacts={"pipeline": LOCAL_PIPE},
        # Must match the stack this was trained and validated on -- MLflow rebuilds the
        # environment from these when serving or loading elsewhere.
        pip_requirements=["diffusers==0.39.0", "transformers<5", "huggingface_hub<2.0,>=0.34",
                          "accelerate", "torch", "pillow"],
        input_example=example, signature=signature,
    )
    ver = mlflow.register_model(info.model_uri, UC_TARGET)
reg_s = round(time.time() - t0, 2)
MlflowClient().set_registered_model_alias(UC_TARGET, ALIAS, ver.version)
print(f"registered {UC_TARGET} v{ver.version} in {reg_s}s; alias @{ALIAS}")

# COMMAND ----------

# Prove the registered artifact works and needs no HF access.
t0 = time.time()
m = mlflow.pyfunc.load_model(f"models:/{UC_TARGET}@{ALIAS}")
load_s = round(time.time() - t0, 2)
img = Image.new("RGB", (512, 512), (120, 120, 120)); msk = Image.new("L", (512, 512), 0)
ImageDraw.Draw(msk).rectangle([150, 200, 360, 260], fill=255)
bi = io.BytesIO(); img.save(bi, format="PNG"); bm = io.BytesIO(); msk.save(bm, format="PNG")
t0 = time.time()
got = m.predict(pd.DataFrame([{"image": base64.b64encode(bi.getvalue()).decode(),
                               "mask": base64.b64encode(bm.getvalue()).decode(),
                               "prompt": "remove text from image, clean background"}]),
                params={"num_inference_steps": 8})
pred_s = round(time.time() - t0, 2)
out_img = Image.open(io.BytesIO(base64.b64decode(got[0])))

result = {"uc_model": UC_TARGET, "version": str(ver.version), "alias": ALIAS,
          "runtime": LABEL, "checkpoint": UNET_DIR, "pipeline_gb": size_gb,
          "register_seconds": reg_s, "uc_load_seconds": load_s,
          "uc_predict_8steps_seconds": pred_s, "roundtrip": f"OK {list(out_img.size)}",
          "promotion_gate": gate}
with open(f"/Volumes/{C}/{S}/data/register_{LABEL}.json", "w") as fh:
    json.dump(result, fh, indent=2)
print(json.dumps(result, indent=2))
