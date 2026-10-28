# Databricks notebook source
# MAGIC %md
# MAGIC # 03c — Evaluation / inference job — **CLASSIC GPU**
# MAGIC
# MAGIC Run on the all-purpose GPU cluster against the **classic** checkpoint.
# MAGIC The AI Runtime v6 twin is `03_eval_inference.py` — kept separate: different
# MAGIC diffusers pin, different checkpoint, different MLflow experiment.
# MAGIC
# MAGIC **Why separate.** Training-time `val/loss` is diffusion MSE on noise prediction at a random
# MAGIC timestep — a training proxy, *not* a quality metric. Real evaluation runs the **full
# MAGIC denoising loop** (50 steps here) through the whole pipeline, which is orders of magnitude
# MAGIC more expensive than one training step and needs the VAE + scheduler, not just the UNet.
# MAGIC
# MAGIC **Precision/recall do not apply.** Those are classification metrics; inpainting is
# MAGIC image-to-image generation. What applies:
# MAGIC
# MAGIC | Metric | Meaning |
# MAGIC |---|---|
# MAGIC | `masked_l1`, `masked_psnr`, `masked_ssim` | fidelity **inside the hole** vs ground truth — the core numbers |
# MAGIC | `full_*` equivalents | did generation damage pixels **outside** the mask |
# MAGIC | `baseline_*` | same metrics for `input` vs `target`, i.e. before any inpainting. The model must beat these or it did nothing useful |
# MAGIC | LPIPS / FID / KID | perceptual + distributional; LPIPS needs a net, FID/KID need hundreds of samples. Not computed here |
# MAGIC | OCR residual-text rate | the **task** metric for Overlay Text Removal, and the only place precision/recall legitimately appear (per-word detection on the output). Needs an OCR dep — see the note at the bottom |
# MAGIC
# MAGIC **Compute:** the all-purpose `g5.12xlarge` cluster (only 1 of its 4 GPUs is used —
# MAGIC inference here is single-image).

# COMMAND ----------

# MAGIC %pip install 'diffusers==0.39.0' accelerate
# MAGIC # ^ CLASSIC pin. DBR 18.x ML GPU ships transformers 4.57, which caps huggingface_hub<1.0,
# MAGIC # which caps diffusers<=0.39. (The AI Runtime variant of this notebook uses 0.42.0.)
# MAGIC # `-q` omitted on purpose so pip's resolver output stays visible.

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text("catalog", "classic_stable_qxxm2v_catalog", "Catalog")
dbutils.widgets.text("schema", "dl_training", "Schema")
dbutils.widgets.text("checkpoint_run", "20261027_220700_4948459ffbec4d13b721f4286322d8cc", "Checkpoint run dir (blank = newest)")
dbutils.widgets.dropdown("checkpoint_tag", "best", ["best", "final"], "Checkpoint")
dbutils.widgets.dropdown("split", "OTR_easy", ["OTR_easy", "OTR_hard"], "Eval split")
dbutils.widgets.text("n_samples", "8", "Samples to evaluate")
dbutils.widgets.text("num_inference_steps", "50", "Denoising steps")
dbutils.widgets.text("experiment", "/Users/jiayi.wu@databricks.com/diffusion_inpainting_eval_classic", "MLflow experiment")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
CKPT_RUN = dbutils.widgets.get("checkpoint_run").strip()
CKPT_TAG = dbutils.widgets.get("checkpoint_tag")
SPLIT = dbutils.widgets.get("split")
N = int(dbutils.widgets.get("n_samples"))
STEPS = int(dbutils.widgets.get("num_inference_steps"))
EXPERIMENT = dbutils.widgets.get("experiment")

DIFFUSION_VOL = f"/Volumes/{CATALOG}/{SCHEMA}/diffusion"
FILES = f"/Volumes/{CATALOG}/{SCHEMA}/data/OTR_like_files/{SPLIT}"
BASE_MODEL = "sd2-community/stable-diffusion-2-inpainting"
PROMPT = "remove text from image, clean background"

# COMMAND ----------

import os, io, glob, json, time
import numpy as np
import torch
from PIL import Image
import mlflow

ckpt_root = f"{DIFFUSION_VOL}/checkpoints"
runs = sorted(os.listdir(ckpt_root))
CKPT_RUN = CKPT_RUN or runs[-1]
CKPT = f"{ckpt_root}/{CKPT_RUN}/{CKPT_TAG}"
print("checkpoint:", CKPT)
assert os.path.isdir(f"{CKPT}/unet"), f"no unet/ under {CKPT}"
print("eval data :", FILES)

# COMMAND ----------

from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel

dtype = torch.float16
pipe = StableDiffusionInpaintPipeline.from_pretrained(BASE_MODEL, torch_dtype=dtype)
# Swap in the FINE-TUNED UNet. (Alternative, and the better production shape: register the
# pipeline as a UC model with mlflow.pyfunc.log_model + mlflow.register_model, then load
# `models:/<catalog>.<schema>.<name>@champion` here instead of a raw volume path.)
pipe.unet = UNet2DConditionModel.from_pretrained(f"{CKPT}/unet", torch_dtype=dtype)
pipe = pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)
print("pipeline ready on", torch.cuda.get_device_name(0))

# COMMAND ----------

def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))


def ssim_gray(a, b):
    """Global SSIM on grayscale, standard constants. Avoids a scikit-image dependency."""
    a = a.mean(axis=2).astype(np.float64)
    b = b.mean(axis=2).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - ma) * (b - mb)).mean()
    return float(((2 * ma * mb + C1) * (2 * cov + C2)) / ((ma**2 + mb**2 + C1) * (va + vb + C2)))


def masked_stats(pred, tgt, mask_bool):
    """Metrics computed over masked pixels only."""
    if mask_bool.sum() == 0:
        return {}
    p = pred[mask_bool].astype(np.float64)
    t = tgt[mask_bool].astype(np.float64)
    mse = np.mean((p - t) ** 2)
    return {"l1": float(np.mean(np.abs(p - t))),
            "psnr": float("inf") if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))}

# COMMAND ----------

files = sorted(glob.glob(f"{FILES}/*.npz"))[:N]
assert files, f"no .npz under {FILES}"
print(f"evaluating {len(files)} samples x {STEPS} steps")

mlflow.set_experiment(EXPERIMENT)
rows, grids = [], []

with mlflow.start_run(run_name=f"eval-{CKPT_RUN}-{CKPT_TAG}-{SPLIT}") as run:
    mlflow.log_params({"checkpoint_run": CKPT_RUN, "checkpoint_tag": CKPT_TAG,
                       "split": SPLIT, "n_samples": len(files),
                       "num_inference_steps": STEPS, "base_model": BASE_MODEL,
                       "prompt": PROMPT})
    t_all = time.time()
    for fp in files:
        with np.load(fp) as z:
            inp = Image.open(io.BytesIO(z["input"].tobytes())).convert("RGB")
            tgt = Image.open(io.BytesIO(z["target"].tobytes())).convert("RGB")
            msk = Image.open(io.BytesIO(z["mask"].tobytes())).convert("L")

        t0 = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            pred = pipe(prompt=PROMPT, image=inp, mask_image=msk,
                        num_inference_steps=STEPS, guidance_scale=7.5).images[0]
        dt = time.time() - t0
        pred = pred.resize(tgt.size)

        P, T, I = np.array(pred), np.array(tgt), np.array(inp)
        mb = np.array(msk.resize(tgt.size)) > 127
        mb3 = np.repeat(mb[:, :, None], 3, axis=2)

        mp, mi = masked_stats(P, T, mb3), masked_stats(I, T, mb3)
        rows.append({
            "file": os.path.basename(fp), "seconds": round(dt, 2),
            "mask_coverage": round(float(mb.mean()), 4),
            # model output vs ground truth
            "masked_l1": round(mp["l1"], 3), "masked_psnr": round(mp["psnr"], 2),
            "full_psnr": round(psnr(P, T), 2), "full_ssim": round(ssim_gray(P, T), 4),
            # BASELINE: the un-inpainted input vs ground truth. The model must beat this.
            "baseline_masked_l1": round(mi["l1"], 3),
            "baseline_masked_psnr": round(mi["psnr"], 2),
            "baseline_full_psnr": round(psnr(I, T), 2),
            "baseline_full_ssim": round(ssim_gray(I, T), 4),
        })
        if len(grids) < 4:
            W, H = tgt.size
            g = Image.new("RGB", (W * 4, H), (0, 0, 0))
            for k, im in enumerate([inp, msk.convert("RGB"), pred, tgt]):
                g.paste(im, (k * W, 0))
            grids.append(g)

    wall = time.time() - t_all

    # Aggregate
    keys = [k for k in rows[0] if k != "file"]
    agg = {f"mean_{k}": float(np.mean([r[k] for r in rows])) for k in keys}
    agg["eval_wall_seconds"] = wall
    agg["sec_per_image"] = wall / len(rows)
    mlflow.log_metrics(agg)

    # Deltas: positive = the model improved on doing nothing
    mlflow.log_metrics({
        "delta_masked_psnr": agg["mean_masked_psnr"] - agg["mean_baseline_masked_psnr"],
        "delta_masked_l1": agg["mean_baseline_masked_l1"] - agg["mean_masked_l1"],
        "delta_full_ssim": agg["mean_full_ssim"] - agg["mean_baseline_full_ssim"],
    })

    os.makedirs("/tmp/eval_out", exist_ok=True)
    with open("/tmp/eval_out/per_sample.json", "w") as f:
        json.dump(rows, f, indent=2)
    mlflow.log_artifact("/tmp/eval_out/per_sample.json", artifact_path="eval")
    for k, g in enumerate(grids):
        pth = f"/tmp/eval_out/grid_{k}_input_mask_pred_target.png"
        g.save(pth)
        mlflow.log_artifact(pth, artifact_path="eval")

    # Distinct filename: the AI Runtime twin writes eval_{SPLIT}_{TAG}.json to the same
    # directory, and a shared path would silently overwrite the other runtime's result.
    out = f"{DIFFUSION_VOL}/_debug/eval_CLASSIC_{SPLIT}_{CKPT_TAG}.json"
    with open(out, "w") as f:
        json.dump({"aggregate": agg, "per_sample": rows}, f, indent=2)

print(json.dumps({k: round(v, 4) for k, v in agg.items()}, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reading the numbers
# MAGIC
# MAGIC Compare `masked_*` against `baseline_masked_*`. The baseline is the **un-inpainted input**
# MAGIC scored against ground truth, so it is the "did nothing" control. `delta_masked_psnr > 0`
# MAGIC means the model genuinely improved the masked region; `<= 0` means it made it worse than
# MAGIC leaving the text in place. A 10-step smoke checkpoint will very likely be **negative** —
# MAGIC that is the expected and honest result, not a bug.
# MAGIC
# MAGIC ## Not implemented here: the task metric
# MAGIC
# MAGIC For Overlay Text Removal the decisive measure is **OCR residual text**: run an OCR engine
# MAGIC over `pred` and over `input`, and report how much of the overlaid text survived. That is
# MAGIC where precision/recall genuinely apply (per-word detection). It needs an extra dependency
# MAGIC (`easyocr`, or `pytesseract` + the tesseract system binary) and ground-truth word boxes,
# MAGIC which the synthetic generator could emit but currently does not.
# MAGIC
# MAGIC ## Testing on a separate dataset
# MAGIC
# MAGIC Point the `split` widget at any other directory of `.npz` samples. For the production shape,
# MAGIC register the pipeline into Unity Catalog first (cookbook:
# MAGIC `serving/sft-qwen2.5-0.5b-lora-serving/register_model.py`) —
# MAGIC `mlflow.pyfunc.log_model(...)` then `mlflow.register_model(uri, "cat.schema.name")` — and
# MAGIC load `models:/cat.schema.name@champion` here instead of a volume checkpoint path. Then for
# MAGIC scale: on **classic** use `mlflow.pyfunc.spark_udf` over a DataFrame; on **AI Runtime**
# MAGIC there is no Spark on the worker, so shard with `@distributed`/`torchrun` or use **Ray Data**
# MAGIC (see the cookbook's `ray/data/batch-inference-*` examples).
