# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Stage the base model: HF hub (pinned) -> Unity Catalog
# MAGIC
# MAGIC **This is the only step that reaches the public internet.** It downloads SD2 inpainting from
# MAGIC the Hugging Face hub at a **pinned commit**, then registers the complete pipeline as a UC
# MAGIC model. Everything downstream loads from UC, so:
# MAGIC
# MAGIC - no egress to `huggingface.co` during training, evaluation, batch inference or serving —
# MAGIC   which matters in a no-public-egress workspace, where `from_pretrained` otherwise fails
# MAGIC - **reproducible**: `from_pretrained` without `revision=` resolves `main` at whatever commit
# MAGIC   exists today, so your "base model" silently drifts
# MAGIC
# MAGIC Nothing of yours leaves Databricks here — the traffic is an inbound `GET` for public weights.
# MAGIC No training data, no checkpoints, no credentials.
# MAGIC
# MAGIC There is **no `mlflow.diffusers` flavor** (MLflow 3.10 has `transformers`,
# MAGIC `sentence_transformers`, `pyfunc`), so the pipeline is wrapped in a custom PyFunc — the same
# MAGIC approach the AI Runtime cookbook uses for its LLM.

# COMMAND ----------

# MAGIC %pip install 'diffusers==0.39.0' accelerate
# MAGIC # diffusers 0.39 needs huggingface_hub>=0.34,<2.0; DBR ships 0.36.2, so nothing to resolve.
# MAGIC # 0.40+ needs huggingface_hub>=1.0, which transformers<5 forbids -> ImportError on
# MAGIC # 'resolve_revision'. Lifting diffusers alone does NOT work; you must lift transformers too.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text("catalog", "classic_stable_qxxm2v_catalog", "Catalog")
dbutils.widgets.text("schema", "sd2_playbook", "Schema")
dbutils.widgets.text("model_name", "sd2_inpaint_base", "UC model name")
dbutils.widgets.text("hf_repo", "sd2-community/stable-diffusion-2-inpainting", "HF repo")
dbutils.widgets.text("revision", "5f74973cbb64c8568780732c17f43eb269d63a0d", "HF commit SHA (pinned)")
dbutils.widgets.dropdown("verify_roundtrip", "true", ["true", "false"], "Load back from UC and infer")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
UC_NAME = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('model_name')}"
HF_REPO = dbutils.widgets.get("hf_repo")
REVISION = dbutils.widgets.get("revision").strip() or None
VERIFY = dbutils.widgets.get("verify_roundtrip") == "true"
LOCAL = "/tmp/sd2_base_pipeline"

import os
os.environ.setdefault("HF_HOME", "/tmp/hf_home")   # /root/.cache may be read-only
print(f"HF  : {HF_REPO} @ {REVISION or 'main (UNPINNED - not reproducible)'}")
print(f"UC  : {UC_NAME}")

# COMMAND ----------

import io, json, time, base64, shutil
import numpy as np, torch, mlflow, pandas as pd
from PIL import Image, ImageDraw
from diffusers import StableDiffusionInpaintPipeline

t0 = time.time()
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    HF_REPO, revision=REVISION, torch_dtype=torch.float16)
dl_s = round(time.time() - t0, 2)

if os.path.exists(LOCAL):
    shutil.rmtree(LOCAL)
pipe.save_pretrained(LOCAL)
size_gb = round(sum(os.path.getsize(os.path.join(r, f))
                    for r, _, fs in os.walk(LOCAL) for f in fs) / 1e9, 2)
print(f"downloaded in {dl_s}s; full pipeline on disk {size_gb} GB")

# COMMAND ----------

class SD2InpaintModel(mlflow.pyfunc.PythonModel):
    """Self-contained SD2 inpainting. base64 PNG in -> base64 PNG out."""

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
        steps = int(p.get("num_inference_steps", 50))
        guid = float(p.get("guidance_scale", 7.5))
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

mlflow.set_registry_uri("databricks-uc")          # REQUIRED for Unity Catalog
mlflow.set_experiment(f"/Users/{spark.sql('select current_user()').first()[0]}/sd2_playbook_register")

# UC REQUIRES an explicit signature (input AND output types). An input_example alone is not
# enough. Declaring params keeps steps/guidance tunable at serving time.
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

t0 = time.time()
with mlflow.start_run(run_name=f"stage-base-{HF_REPO.split('/')[-1]}") as run:
    mlflow.log_params({"hf_repo": HF_REPO, "hf_revision": REVISION or "main-unpinned",
                       "pipeline_gb": size_gb, "download_seconds": dl_s})
    info = mlflow.pyfunc.log_model(
        name="sd2_inpaint_base",
        python_model=SD2InpaintModel(),
        artifacts={"pipeline": LOCAL},
        # Reproducible on BOTH runtimes: diffusers 0.39 accepts hub>=0.34,<2.0, which is what
        # DBR ships and what AIR Standard resolves to.
        pip_requirements=["diffusers==0.39.0", "transformers<5", "huggingface_hub<2.0,>=0.34",
                          "accelerate", "torch", "pillow"],
        input_example=example,
        signature=signature,
    )
    ver = mlflow.register_model(info.model_uri, UC_NAME)
reg_s = round(time.time() - t0, 2)
MlflowClient().set_registered_model_alias(UC_NAME, "base", ver.version)
print(f"registered {UC_NAME} v{ver.version} in {reg_s}s; alias @base -> v{ver.version}")

# COMMAND ----------

result = {"uc_model": UC_NAME, "version": str(ver.version), "alias": "base",
          "hf_repo": HF_REPO, "hf_revision": REVISION, "pipeline_gb": size_gb,
          "download_seconds": dl_s, "register_seconds": reg_s}

if VERIFY:
    # Prove self-containment: load from UC and infer. No HF access on this path.
    t0 = time.time()
    m = mlflow.pyfunc.load_model(f"models:/{UC_NAME}@base")
    result["uc_load_seconds"] = round(time.time() - t0, 2)
    img = Image.new("RGB", (512, 512), (120, 120, 120))
    msk = Image.new("L", (512, 512), 0)
    ImageDraw.Draw(msk).rectangle([150, 200, 360, 260], fill=255)
    bi = io.BytesIO(); img.save(bi, format="PNG")
    bm = io.BytesIO(); msk.save(bm, format="PNG")
    t0 = time.time()
    got = m.predict(pd.DataFrame([{"image": base64.b64encode(bi.getvalue()).decode(),
                                   "mask": base64.b64encode(bm.getvalue()).decode(),
                                   "prompt": "remove text from image, clean background"}]),
                    params={"num_inference_steps": 8})
    result["uc_predict_8steps_seconds"] = round(time.time() - t0, 2)
    out_img = Image.open(io.BytesIO(base64.b64decode(got[0])))
    result["roundtrip"] = f"OK {list(out_img.size)}"

out_p = f"/Volumes/{CATALOG}/{SCHEMA}/data/stage_base_model.json"
with open(out_p, "w") as fh:
    json.dump(result, fh, indent=2)
print(json.dumps(result, indent=2))
