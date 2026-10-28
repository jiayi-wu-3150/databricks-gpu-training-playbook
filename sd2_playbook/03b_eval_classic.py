# Databricks notebook source
# MAGIC %md
# MAGIC # 03b — Evaluate on the **TEST** split, with a base-model **ablation** (CLASSIC)
# MAGIC
# MAGIC Classic twin of `03a_eval_serverless.py`. Same metrics, same ablation; separate job,
# MAGIC separate checkpoint, separate output file (`eval_classic_test.json`).
# MAGIC
# MAGIC Evaluation is a **separate job** from training on purpose: training validation runs one
# MAGIC denoising step, real evaluation runs the full 50-step loop through the whole pipeline.
# MAGIC
# MAGIC **`val/loss` from training is not a quality metric** — it is diffusion MSE on noise
# MAGIC prediction at a random timestep. This notebook produces the actual numbers.
# MAGIC
# MAGIC ## Three-way comparison (the point of this notebook)
# MAGIC
# MAGIC | Arm | What it is | Answers |
# MAGIC |---|---|---|
# MAGIC | `baseline` | the **un-inpainted input** scored against ground truth | "did anything help at all?" |
# MAGIC | `base` | the **un-fine-tuned** UC base model | "how much is just Stable Diffusion?" |
# MAGIC | `finetuned` | our checkpoint | "did **our training** add anything?" |
# MAGIC
# MAGIC Beating `baseline` only proves inpainting works — overlaid text pixels are maximally wrong,
# MAGIC so almost any plausible fill scores better. **`finetuned` vs `base` is the comparison that
# MAGIC attributes anything to training.** Without it you are reporting that SD2 can inpaint.
# MAGIC
# MAGIC Precision/recall do not apply — those are classification metrics. Inpainting is a
# MAGIC reconstruction task, so this reports masked-region PSNR/L1 plus full-image SSIM.
# MAGIC
# MAGIC Compute: the all-purpose GPU cluster (uses 1 GPU — inference is single-image).

# COMMAND ----------

# MAGIC %pip install 'diffusers==0.39.0' accelerate
# MAGIC # ^ classic installs deps in-notebook; the serverless twin declares them in the job's
# MAGIC # `environments` block instead (Standard GPU + %pip hits a CUDA fork-safety limit).

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text("catalog", "classic_stable_qxxm2v_catalog", "Catalog")
dbutils.widgets.text("schema", "sd2_playbook", "Schema")
dbutils.widgets.text("base_model", "sd2_inpaint_base", "UC base model")
dbutils.widgets.text("base_alias", "base", "Base alias")
dbutils.widgets.text("checkpoint_dir", "", "Fine-tuned checkpoint dir (blank = newest classic_*)")
dbutils.widgets.text("runtime_label", "classic", "Runtime label (output naming)")
dbutils.widgets.dropdown("split", "test", ["test", "val"], "Split")
dbutils.widgets.text("n_samples", "8", "Samples")
dbutils.widgets.text("num_inference_steps", "50", "Denoising steps")
dbutils.widgets.text("seed", "1234", "Sampling seed (identical across arms)")

C = dbutils.widgets.get("catalog"); S = dbutils.widgets.get("schema")
UC_BASE = f"{C}.{S}.{dbutils.widgets.get('base_model')}"
BASE_ALIAS = dbutils.widgets.get("base_alias")
CKPT_IN = dbutils.widgets.get("checkpoint_dir").strip()
LABEL = dbutils.widgets.get("runtime_label")
SPLIT = dbutils.widgets.get("split")
N = int(dbutils.widgets.get("n_samples"))
STEPS = int(dbutils.widgets.get("num_inference_steps"))
SEED = int(dbutils.widgets.get("seed"))

PQ = f"/Volumes/{C}/{S}/data/parquet/{SPLIT}"
CKPT_ROOT = f"/Volumes/{C}/{S}/checkpoints"
PROMPT = "remove text from image, clean background"

import os, io, glob, json, time
import numpy as np, torch, mlflow, pandas as pd
from PIL import Image, ImageDraw
import pyarrow.parquet as pq

if not CKPT_IN:
    cands = sorted(d for d in glob.glob(f"{CKPT_ROOT}/{LABEL}_*") if os.path.isdir(d))
    assert cands, f"no {LABEL}_* checkpoint under {CKPT_ROOT}"
    CKPT_IN = cands[-1]
UNET_DIR = f"{CKPT_IN}/best/unet"
assert os.path.isdir(UNET_DIR), f"no unet at {UNET_DIR}"
print(f"split     : {SPLIT} ({PQ})\nfine-tuned: {UNET_DIR}\nbase      : {UC_BASE}@{BASE_ALIAS}")

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
_root = mlflow.artifacts.download_artifacts(f"models:/{UC_BASE}@{BASE_ALIAS}")
BASE_DIR = next((r for r, _, fs in os.walk(_root) if "model_index.json" in fs), None)
assert BASE_DIR, "base pipeline not found"
print("base pipeline ->", BASE_DIR)

# COMMAND ----------

def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))


def ssim_gray(a, b):
    """Global SSIM on grayscale with standard constants; avoids a scikit-image dependency."""
    a = a.mean(axis=2).astype(np.float64); b = b.mean(axis=2).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ma, mb, va, vb = a.mean(), b.mean(), a.var(), b.var()
    cov = ((a - ma) * (b - mb)).mean()
    return float(((2 * ma * mb + C1) * (2 * cov + C2)) / ((ma ** 2 + mb ** 2 + C1) * (va + vb + C2)))


def masked(pred, tgt, m3):
    p, t = pred[m3].astype(np.float64), tgt[m3].astype(np.float64)
    mse = np.mean((p - t) ** 2)
    return {"l1": float(np.mean(np.abs(p - t))),
            "psnr": float("inf") if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))}

# COMMAND ----------

rows = []
for f in sorted(glob.glob(f"{PQ}/*.parquet")):
    rows.extend(pq.read_table(f).to_pylist())
    if len(rows) >= N:
        break
rows = rows[:N]
assert rows, f"no rows under {PQ}"
print(f"evaluating {len(rows)} samples x {STEPS} steps, 2 model arms")

# COMMAND ----------

from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel
import shutil

# Reading model weights DIRECTLY from a UC volume over FUSE is very slow: measured 265.8 s for a
# ~3.3 GB UNet (~12 MB/s) versus 1.1 s from a local path -- a 240x difference. Stage to local disk
# first. (The upstream repo had a _cache_to_local helper for exactly this reason.)
def _stage_local(vol_path, label):
    local = f"/local_disk0/{label}" if os.path.isdir("/local_disk0") else f"/tmp/{label}"
    if os.path.isdir(local):
        return local
    t0 = time.time()
    try:
        shutil.copytree(vol_path, local)
        print(f"  staged {label} -> {local} ({round(time.time()-t0,1)}s)")
        return local
    except Exception as e:
        print(f"  staging failed ({type(e).__name__}); reading from the volume directly")
        return vol_path


UNET_LOCAL = _stage_local(UNET_DIR, "sd2_finetuned_unet")


def load(which):
    p = StableDiffusionInpaintPipeline.from_pretrained(BASE_DIR, torch_dtype=torch.float16)
    if which == "finetuned":
        p.unet = UNet2DConditionModel.from_pretrained(UNET_LOCAL, torch_dtype=torch.float16)
    p = p.to("cuda"); p.set_progress_bar_config(disable=True)
    return p

per_sample, grids = [], []
arm_times = {}
for arm in ("base", "finetuned"):
    t0 = time.time()
    pipe = load(arm)
    arm_times[f"{arm}_load_seconds"] = round(time.time() - t0, 2)
    for i, r in enumerate(rows):
        inp = Image.open(io.BytesIO(r["input_jpg"])).convert("RGB")
        tgt = Image.open(io.BytesIO(r["target_jpg"])).convert("RGB")
        msk = Image.open(io.BytesIO(r["mask_png"])).convert("L")
        t0 = time.time()
        # FIXED SEED PER SAMPLE, identical across arms. Without this the pipeline draws fresh
        # noise each call, so `base` and `finetuned` denoise from DIFFERENT starting points and
        # the arm-to-arm delta is dominated by sampling variance rather than the model.
        gen = torch.Generator(device="cuda").manual_seed(SEED + i)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            pred = pipe(prompt=PROMPT, image=inp, mask_image=msk,
                        num_inference_steps=STEPS, guidance_scale=7.5,
                        generator=gen).images[0]
        dt = time.time() - t0
        pred = pred.resize(tgt.size)
        P, T, I = np.array(pred), np.array(tgt), np.array(inp)
        mb = np.array(msk.resize(tgt.size)) > 127
        m3 = np.repeat(mb[:, :, None], 3, axis=2)
        mp, mi = masked(P, T, m3), masked(I, T, m3)
        per_sample.append({
            "arm": arm, "id": r["id"], "seconds": round(dt, 2),
            "mask_coverage": round(float(mb.mean()), 4),
            "masked_l1": round(mp["l1"], 3), "masked_psnr": round(mp["psnr"], 2),
            "full_psnr": round(psnr(P, T), 2), "full_ssim": round(ssim_gray(P, T), 4),
            "baseline_masked_l1": round(mi["l1"], 3),
            "baseline_masked_psnr": round(mi["psnr"], 2),
            "baseline_full_psnr": round(psnr(I, T), 2),
            "baseline_full_ssim": round(ssim_gray(I, T), 4),
        })
        if arm == "finetuned" and len(grids) < 4:
            W, H = tgt.size
            g = Image.new("RGB", (W * 4, H), (0, 0, 0))
            for k, im in enumerate([inp, msk.convert("RGB"), pred, tgt]):
                g.paste(im, (k * W, 0))
            grids.append((r["id"], g))
    del pipe; torch.cuda.empty_cache()
    print(f"arm '{arm}' done")

# COMMAND ----------

def agg(arm):
    sub = [r for r in per_sample if r["arm"] == arm]
    keys = [k for k in sub[0] if isinstance(sub[0][k], (int, float))]
    return {k: float(np.mean([r[k] for r in sub])) for k in keys}

A = {arm: agg(arm) for arm in ("base", "finetuned")}
report = {
    "runtime": LABEL, "split": SPLIT, "n_samples": len(rows), "steps": STEPS,
    "seed": SEED,
    "checkpoint": UNET_DIR, "uc_base": f"{UC_BASE}@{BASE_ALIAS}",
    "baseline_masked_psnr": round(A["base"]["baseline_masked_psnr"], 3),
    "baseline_masked_l1": round(A["base"]["baseline_masked_l1"], 3),
    "base_masked_psnr": round(A["base"]["masked_psnr"], 3),
    "finetuned_masked_psnr": round(A["finetuned"]["masked_psnr"], 3),
    "base_masked_l1": round(A["base"]["masked_l1"], 3),
    "finetuned_masked_l1": round(A["finetuned"]["masked_l1"], 3),
    "base_full_ssim": round(A["base"]["full_ssim"], 4),
    "finetuned_full_ssim": round(A["finetuned"]["full_ssim"], 4),
    # the two deltas that matter, in order of what they prove
    "delta_vs_baseline_psnr": round(A["finetuned"]["masked_psnr"] - A["base"]["baseline_masked_psnr"], 3),
    "delta_finetuned_vs_base_psnr": round(A["finetuned"]["masked_psnr"] - A["base"]["masked_psnr"], 3),
    "delta_finetuned_vs_base_l1": round(A["base"]["masked_l1"] - A["finetuned"]["masked_l1"], 3),
    **arm_times,
    "sec_per_image": round(float(np.mean([r["seconds"] for r in per_sample])), 2),
}
# A bare `> 0` test is far too confident at this sample size. Measured: with the sampler
# UNSEEDED the sign of this delta flipped between runs, and even seeded, both checkpoints land
# within +/-0.17 dB of base on n=8. Require a real margin and say the sample size out loud.
MARGIN_DB = 0.25
_d = report["delta_finetuned_vs_base_psnr"]
_n = len(rows)
if _d > MARGIN_DB:
    report["VERDICT"] = f"fine-tuning IMPROVED on the base model (+{_d:.3f} dB > {MARGIN_DB} dB margin, n={_n})"
elif _d < -MARGIN_DB:
    report["VERDICT"] = f"fine-tuning made it WORSE ({_d:.3f} dB, n={_n})"
else:
    report["VERDICT"] = (f"INCONCLUSIVE: |{_d:.3f}| dB is within the {MARGIN_DB} dB noise margin "
                         f"at n={_n}. Ten optimizer steps are not expected to move an 865M-param "
                         f"UNet; raise steps and n before drawing a conclusion.")
report["margin_db"] = MARGIN_DB

mlflow.set_experiment(f"/Users/{spark.sql('select current_user()').first()[0]}/sd2_playbook_eval")
with mlflow.start_run(run_name=f"eval-{LABEL}-{SPLIT}") as run:
    mlflow.log_params({"runtime": LABEL, "split": SPLIT, "n_samples": len(rows),
                       "num_inference_steps": STEPS, "seed": SEED, "checkpoint": UNET_DIR,
                       "uc_base": f"{UC_BASE}@{BASE_ALIAS}", "prompt": PROMPT})
    mlflow.log_metrics({k: v for k, v in report.items() if isinstance(v, (int, float))})
    os.makedirs("/tmp/ev", exist_ok=True)
    with open("/tmp/ev/per_sample.json", "w") as fh:
        json.dump(per_sample, fh, indent=2)
    mlflow.log_artifact("/tmp/ev/per_sample.json", artifact_path="eval")
    for sid, g in grids:
        pth = f"/tmp/ev/grid_{sid}_input_mask_pred_target.png"
        g.save(pth); mlflow.log_artifact(pth, artifact_path="eval")

out = f"/Volumes/{C}/{S}/data/eval_{LABEL}_{SPLIT}.json"
with open(out, "w") as fh:
    json.dump({"report": report, "per_sample": per_sample}, fh, indent=2)
print(json.dumps(report, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reading the result
# MAGIC
# MAGIC - `delta_vs_baseline_psnr > 0` -> inpainting beat leaving the text in place. Expected, and
# MAGIC   weak evidence: the baseline is a low bar by construction.
# MAGIC - **`delta_finetuned_vs_base_psnr`** is the real number. `> 0` means our training helped;
# MAGIC   `<= 0` means the base model was already as good or better, which is the **expected**
# MAGIC   outcome for a 10-step smoke run and not a bug.
# MAGIC
# MAGIC Not computed here: **OCR residual-text rate**, the task metric for Overlay Text Removal and
# MAGIC the one place precision/recall legitimately apply. `00_prepare_data` already emits the
# MAGIC ground-truth words and boxes (`words` column, `raw/*/meta/*.json`) so it can be added
# MAGIC exactly, without labelling — it needs an OCR dependency (`easyocr`).
