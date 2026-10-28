# Databricks notebook source
# MAGIC %md
# MAGIC # 04c — Register in Unity Catalog + time load paths — **CLASSIC GPU**
# MAGIC
# MAGIC Classic twin of `04_register_uc_and_time_load.py`. Separate UC model
# MAGIC (`sd2_inpaint_otr_classic`), separate checkpoint, separate experiment,
# MAGIC separate timing file, and `diffusers==0.39.0` throughout.
# MAGIC
# MAGIC **Why a custom PyFunc:** MLflow 3.10 has **no `mlflow.diffusers` flavor** (only
# MAGIC `transformers`, `sentence_transformers`, `pyfunc`). So a diffusion pipeline is registered
# MAGIC the same way the cookbook registers its LLM
# MAGIC (`serving/sft-qwen2.5-0.5b-lora-serving/register_model.py`): wrap it in a
# MAGIC `mlflow.pyfunc.PythonModel`, `log_model`, then `register_model` into UC.
# MAGIC
# MAGIC **The question being tested:** is loading from a UC registered model faster than loading
# MAGIC from the raw volume checkpoint? The two paths are not just different storage:
# MAGIC
# MAGIC | Path | What actually happens |
# MAGIC |---|---|
# MAGIC | **volume checkpoint** | base pipeline pulled from the **HF hub** (network) + fine-tuned UNet read from `/Volumes` over **FUSE** |
# MAGIC | **UC model** | **one self-contained artifact** fetched by the MLflow client — no HF dependency at inference time |
# MAGIC
# MAGIC So UC may win for reasons beyond raw throughput: one parallel download instead of
# MAGIC hub-fetch + FUSE reads, and no external network dependency. It may also lose: the UC
# MAGIC artifact contains the *whole* pipeline (~5 GB) rather than just the UNet.

# COMMAND ----------

# MAGIC %pip install 'diffusers==0.39.0' accelerate
# MAGIC # ^ CLASSIC pin. The AI Runtime twin uses 0.42.0; DBR 18.x ML GPU ships transformers 4.57,
# MAGIC # which caps huggingface_hub<1.0, which caps diffusers<=0.39.

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text("catalog", "classic_stable_qxxm2v_catalog", "Catalog")
dbutils.widgets.text("schema", "dl_training", "Schema")
dbutils.widgets.text("model_name", "sd2_inpaint_otr_classic", "UC model name")
dbutils.widgets.text("checkpoint_run", "20261027_220700_4948459ffbec4d13b721f4286322d8cc", "Checkpoint run dir")
dbutils.widgets.dropdown("checkpoint_tag", "best", ["best", "final"], "Checkpoint")
dbutils.widgets.dropdown("do_register", "true", ["true", "false"], "Register (false = time only)")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
MODEL_NAME = dbutils.widgets.get("model_name")
CKPT_RUN = dbutils.widgets.get("checkpoint_run")
CKPT_TAG = dbutils.widgets.get("checkpoint_tag")
DO_REGISTER = dbutils.widgets.get("do_register") == "true"

UC_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
DIFFUSION_VOL = f"/Volumes/{CATALOG}/{SCHEMA}/diffusion"
CKPT = f"{DIFFUSION_VOL}/checkpoints/{CKPT_RUN}/{CKPT_TAG}"
BASE_MODEL = "sd2-community/stable-diffusion-2-inpainting"
LOCAL_PIPE = "/tmp/sd2_full_pipeline"
print("checkpoint:", CKPT, "\nUC target :", UC_NAME)

# COMMAND ----------

import os, io, time, json, base64, shutil
import numpy as np, torch, mlflow
from PIL import Image

os.environ.setdefault("HF_HOME", "/tmp/hf_home")
timings = {}

# COMMAND ----------

# ── Path A: the CURRENT approach — HF hub for the base + FUSE for the fine-tuned UNet ──
from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel

t0 = time.time()
pipe = StableDiffusionInpaintPipeline.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)
pipe.unet = UNet2DConditionModel.from_pretrained(f"{CKPT}/unet", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
timings["A_volume_plus_hf_seconds"] = round(time.time() - t0, 2)
print("A) volume + HF hub load:", timings["A_volume_plus_hf_seconds"], "s")

# Materialise the COMPLETE pipeline locally so the UC artifact is self-contained.
if os.path.exists(LOCAL_PIPE):
    shutil.rmtree(LOCAL_PIPE)
pipe.save_pretrained(LOCAL_PIPE)
sz = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(LOCAL_PIPE) for f in fs)
print(f"full pipeline on disk: {sz/1e9:.2f} GB")

# COMMAND ----------

class SD2InpaintModel(mlflow.pyfunc.PythonModel):
    """Self-contained inpainting model. predict() takes base64 PNG input+mask, returns base64 PNG."""

    def load_context(self, context):
        import torch
        from diffusers import StableDiffusionInpaintPipeline
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            context.artifacts["pipeline"], torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe.set_progress_bar_config(disable=True)

    def predict(self, context, model_input, params=None):
        import base64, io
        from PIL import Image
        steps = int((params or {}).get("num_inference_steps", 50))
        guidance = float((params or {}).get("guidance_scale", 7.5))
        out = []
        for _, row in model_input.iterrows():
            img = Image.open(io.BytesIO(base64.b64decode(row["image"]))).convert("RGB")
            msk = Image.open(io.BytesIO(base64.b64decode(row["mask"]))).convert("L")
            res = self.pipe(prompt=row.get("prompt", "remove text from image, clean background"),
                            image=img, mask_image=msk,
                            num_inference_steps=steps, guidance_scale=guidance).images[0]
            b = io.BytesIO(); res.save(b, format="PNG")
            out.append(base64.b64encode(b.getvalue()).decode())
        return out

# COMMAND ----------

import pandas as pd

if DO_REGISTER:
    mlflow.set_registry_uri("databricks-uc")   # REQUIRED for Unity Catalog
    # Unity Catalog REQUIRES an explicit signature: "All models in the Unity Catalog must be
    # logged with a model signature containing both input and output type specifications."
    # An input_example of empty strings infers nothing, so declare the schema outright.
    # Declaring params too makes num_inference_steps/guidance_scale tunable at serving time.
    from mlflow.models import ModelSignature
    from mlflow.types.schema import Schema, ColSpec, ParamSchema, ParamSpec

    signature = ModelSignature(
        inputs=Schema([ColSpec("string", "image"),
                       ColSpec("string", "mask"),
                       ColSpec("string", "prompt")]),
        outputs=Schema([ColSpec("string")]),
        params=ParamSchema([ParamSpec("num_inference_steps", "integer", 50),
                            ParamSpec("guidance_scale", "double", 7.5)]),
    )

    # A REAL (tiny) example, so the logged example is loadable rather than a placeholder.
    _b = io.BytesIO(); Image.new("RGB", (64, 64), (127, 127, 127)).save(_b, format="PNG")
    _m = io.BytesIO(); Image.new("L", (64, 64), 0).save(_m, format="PNG")
    example = pd.DataFrame([{
        "image": base64.b64encode(_b.getvalue()).decode(),
        "mask": base64.b64encode(_m.getvalue()).decode(),
        "prompt": "remove text from image, clean background",
    }])

    mlflow.set_experiment("/Users/jiayi.wu@databricks.com/diffusion_uc_register_classic")
    t0 = time.time()
    with mlflow.start_run(run_name=f"register-{MODEL_NAME}-{CKPT_RUN[:15]}") as run:
        mlflow.log_params({"checkpoint_run": CKPT_RUN, "checkpoint_tag": CKPT_TAG,
                           "base_model": BASE_MODEL, "pipeline_gb": round(sz/1e9, 2)})
        info = mlflow.pyfunc.log_model(
            name="sd2_inpaint",
            python_model=SD2InpaintModel(),
            artifacts={"pipeline": LOCAL_PIPE},
            # Pinned to the CLASSIC-compatible stack so anything loading this model
            # version reproduces the environment it was trained and validated on.
            pip_requirements=["diffusers==0.39.0", "transformers<5", "accelerate",
                              "torch", "pillow"],
            input_example=example,
            signature=signature,
        )
        ver = mlflow.register_model(info.model_uri, UC_NAME)
    timings["register_seconds"] = round(time.time() - t0, 2)
    print(f"registered {UC_NAME} version {ver.version} in {timings['register_seconds']}s")

    from mlflow import MlflowClient
    MlflowClient().set_registered_model_alias(UC_NAME, "champion", ver.version)
    print("alias @champion ->", ver.version)
    UC_URI = f"models:/{UC_NAME}@champion"
else:
    UC_URI = f"models:/{UC_NAME}@champion"

# COMMAND ----------

# ── Path B: load from the UC registered model ──
mlflow.set_registry_uri("databricks-uc")
t0 = time.time()
uc_model = mlflow.pyfunc.load_model(UC_URI)
timings["B_uc_model_seconds"] = round(time.time() - t0, 2)
print("B) UC model load:", timings["B_uc_model_seconds"], "s")

timings["delta_seconds_A_minus_B"] = round(
    timings["A_volume_plus_hf_seconds"] - timings["B_uc_model_seconds"], 2)
timings["faster"] = "UC model" if timings["delta_seconds_A_minus_B"] > 0 else "volume + HF"

# COMMAND ----------

# Sanity-check the UC model actually produces an image (8 steps — speed, not quality).
import glob
f = sorted(glob.glob(f"/Volumes/{CATALOG}/{SCHEMA}/data/OTR_like_files/OTR_easy/*.npz"))[0]
with np.load(f) as z:
    inp_b = base64.b64encode(z["input"].tobytes()).decode()
    msk_b = base64.b64encode(z["mask"].tobytes()).decode()

t0 = time.time()
res = uc_model.predict(pd.DataFrame([{"image": inp_b, "mask": msk_b,
                                      "prompt": "remove text from image, clean background"}]),
                       params={"num_inference_steps": 8})
timings["uc_predict_8steps_seconds"] = round(time.time() - t0, 2)
img = Image.open(io.BytesIO(base64.b64decode(res[0])))
print("UC model predict OK ->", img.size, img.mode, f"in {timings['uc_predict_8steps_seconds']}s")

out = f"{DIFFUSION_VOL}/_debug/uc_model_load_timing_CLASSIC.json"
with open(out, "w") as fh:
    json.dump({"uc_model": UC_NAME, "checkpoint": CKPT, "timings": timings}, fh, indent=2)
print(json.dumps(timings, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Caveats on the timing
# MAGIC
# MAGIC Single measurement each, on one node, and **cache state is not controlled** — the HF hub
# MAGIC download in path A may hit a warm `HF_HOME`, and the UC artifact fetch may hit a warm
# MAGIC local cache. Re-run on a fresh cluster for a defensible number. Treat the result as
# MAGIC indicative, not a benchmark.
# MAGIC
# MAGIC The durable arguments for UC registration are **not** speed: versioning, `@champion`
# MAGIC aliases, lineage, access control, no external HF dependency at inference time, and
# MAGIC Model Serving compatibility.
