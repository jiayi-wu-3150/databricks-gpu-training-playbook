# Databricks notebook source
# MAGIC %md
# MAGIC # 95 — Does the FORCED modern stack actually WORK on classic (not just import)?
# MAGIC Loads SD2 inpainting with diffusers 0.42 + transformers 5.x on DBR 18 LTS ML GPU and runs
# MAGIC one real denoising pass. Import success is not functional success.

# COMMAND ----------

# MAGIC %pip install 'diffusers==0.42.0' 'transformers>=5' 'huggingface_hub>=1.32,<3' accelerate pillow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json, io, os, time, traceback
rep = {}
try:
    import importlib.metadata as md
    for p in ("diffusers", "transformers", "huggingface_hub", "torch", "accelerate"):
        try: rep[p] = md.version(p)
        except Exception: rep[p] = "absent"

    import torch
    from PIL import Image
    from diffusers import StableDiffusionInpaintPipeline
    rep["cuda"] = torch.cuda.is_available()
    rep["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    t0 = time.time()
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "sd2-community/stable-diffusion-2-inpainting", torch_dtype=torch.float16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    rep["pipeline_load_seconds"] = round(time.time() - t0, 2)

    img = Image.new("RGB", (512, 512), (120, 120, 120))
    msk = Image.new("L", (512, 512), 0)
    from PIL import ImageDraw
    ImageDraw.Draw(msk).rectangle([150, 200, 360, 260], fill=255)

    t0 = time.time()
    out = pipe(prompt="remove text from image, clean background", image=img, mask_image=msk,
               num_inference_steps=8, guidance_scale=7.5).images[0]
    rep["inference_seconds"] = round(time.time() - t0, 2)
    rep["output_size"] = list(out.size)
    rep["VERDICT"] = "FUNCTIONAL: diffusers 0.42 + transformers 5.x runs on classic DBR 18 LTS"
except Exception as e:
    rep["VERDICT"] = f"FAILED {type(e).__name__}: {e}"
    rep["trace"] = traceback.format_exc()[-1500:]

out_p = "/Volumes/classic_stable_qxxm2v_catalog/dl_training/diffusion/_debug/classic_modern_functional.json"
os.makedirs(os.path.dirname(out_p), exist_ok=True)
open(out_p, "w").write(json.dumps(rep, indent=2))
print(json.dumps({k: v for k, v in rep.items() if k != "trace"}, indent=2))
