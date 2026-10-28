# Diffusion inpainting — complete run inventory

Workspace `fevm-classic-stable-qxxm2v`. All runs 2026-10-27/28. **All SUCCESS unless noted.**

Legend for the "includes" column: **DL**=download data · **TF**=transform/stage data ·
**TR**=training · **VAL**=in-loop validation · **CKPT**=checkpointing · **INF**=inference/sample
generation · **SYS**=MLflow system metrics (GPU/CPU/disk/net) · **MET**=MLflow metrics ·
**PAR**=MLflow params · **ART**=MLflow artifacts · **UC**=Unity Catalog model registration

---

## 1. Data preparation — runtime-agnostic (executed on the classic cluster)

Ran on classic only because reading MDS needs `mosaicml-streaming`; the *outputs* feed both runtimes.

| Run | Notebook | Includes | What it produced |
|---|---|---|---|
| `971947802046448` | `00_make_smoke_otr_mds` | **TF** | Synthesised 320 OTR-shaped samples (text drawn on Food-101) → **MDS shards**. 256 train / 32 OTR_easy / 32 OTR_hard |
| `386660202679506` | `01_mds_to_volume_files` | **TF** | MDS → **one `.npz` per sample** (256/32/32) for `UCVolumeDataset`. Earlier attempt failed `Errno 5`; fixed with in-memory zip |
| `450695538803619` | `02_mds_to_parquet` | **TF** | MDS → **Parquet** (4 train files @64 rows, 1 per eval split, 21.7 MB). ⚠️ **Staged but no training run consumes it yet** |

**Not run: `download_otr_v0.1.py`** — the real OTR dataset (~80 GB). All data here is synthetic.

---

## 2. Classic GPU (`g5.12xlarge`, 4×A10G, DBR 18.x ML GPU)

| Run | Notebook | Includes | Notes |
|---|---|---|---|
| `274618671165412` | `finetune_sd2_inpainting_classic` | **TR VAL CKPT INF SYS MET ART** | `TorchDistributor` function mode, 4 ranks. 10 steps. `train/loss` 0.147, `val/loss` 0.004544. Artifacts: `checkpoints/`, `run_info/rank_0_log.txt`, 1 sample PNG. **No PAR** |
| `529739611474389` | `03c_eval_inference_classic` | **INF MET PAR ART** | 8 samples × 50 denoising steps. `diffusers==0.39.0` |
| `297531922273926` | `04c_register_uc_and_time_load_classic` | **UC MET PAR** | Registered `…dl_training.sd2_inpaint_otr_classic` @champion. Load timing: volume+HF **2.57 s** vs UC **74.2 s** |

Earlier classic attempts (all failed, each a documented defect): `326610631282563` interactive-only
auth · `790294401329234` same · `80797586037851` unregistered volume `Errno 95` ·
`572766583208641` / `903585843970850` `diffusers`/`huggingface_hub` resolution ·
`343297702963413` worker MLflow host missing.

---

## 3. Serverless / AI Runtime

| Run | Notebook | Compute | Includes | Notes |
|---|---|---|---|---|
| `753433848654637` | `finetune_sd2_inpainting_serverless` | `GPU_8xH100` env v4 | **TR VAL CKPT INF SYS MET ART** | First green serverless run, **MDS** data path, `diffusers 0.39` (capped by Mosaic Streaming) |
| `175620959119063` | same | `GPU_8xH100` env v4 | **TR VAL CKPT INF SYS MET ART** | Switched to **`UCVolumeDataset`** + `air DataLoader` → unlocked `diffusers 0.42` + transformers 5.x |
| `249353804464778` | same | `GPU_8xH100` **env v6** | **TR VAL CKPT INF SYS MET ART** | `train/loss` 0.008790, 0.935 min. 8 ranks via `@distributed` |
| `538335853077946` | `03_eval_inference` | `GPU_1xA10` env v6 | **INF MET PAR ART** | 8 samples × 50 steps, `diffusers 0.42` |
| `188067677918990` | `04_register_uc_and_time_load` | `GPU_1xA10` env v6 | **UC MET PAR** | Registered `…dl_training.sd2_inpaint_otr` @champion. volume+HF **22.55 s** vs UC **43.43 s** |

---

## 4. Diagnostic probes — each produced a finding, not a model

| Run | Notebook | Finding |
|---|---|---|
| `819240757457180` | `99_env_probe` | DBR ships `transformers 4.57.6` + `hub 0.36.2` → `diffusers 0.42` cannot install. Established the `diffusers==0.39.0` ceiling for classic |
| `621024503550778` | `98_ctx_probe` | In a **Job** run `apiToken()`/`apiUrl()`/`spark.databricks.workspaceUrl` all work but **`browserHostName()` raises** — and a bare subprocess cannot reach MLflow without explicit creds |
| `295265882768327` `9460406665489` `65988178047673` `923157926666015` | `97_dep_probe` ×4 | Parallel bisection: each dependency group installs **alone**, so the failure was a *combination* → found `mosaicml-streaming` requires `transformers<5` |
| `617295094266771` | `96_env_version_probe` | `environment_version: "6"` on **CPU** = generic serverless Python, no torch, no `databricks.air` |
| `1074247795829740` | same | `environment_version: "6"` + **GPU** = **Standard v6** (cuda + `databricks-sdk-air` preinstalled, still no torch). **AI v6** needs `base_environment: databricks_ai_v6`, which this workspace rejects |

---

## 5. What is measured where

**During training** (both runtimes, automatic):
- `MET`: `train/loss`, `train/lr`, `val/loss`, `val/best_loss`, `checkpoint/step`,
  `checkpoint/best_step`, `train/steps_per_sec`, `train/total_minutes`
- `SYS`: `system/gpu_N_*` (memory, utilisation, power), CPU, disk, network — MLflow system metrics
- `ART`: `checkpoints/{best,final,checkpoint-N}/unet`, `run_info/rank_N_log.txt`,
  `run_info/node_map.json`, `samples/step_*.png`
- ⚠️ **`val/loss` is diffusion MSE on noise prediction — a training proxy, NOT a quality metric**

**In the separate eval job** (`03` / `03c`):
- `masked_l1`, `masked_psnr` — fidelity **inside the mask** (the number that matters)
- `full_psnr`, `full_ssim` — damage **outside** the mask
- `baseline_*` for all of the above — the **un-inpainted input** vs ground truth, i.e. the
  "did nothing" control
- `delta_masked_psnr`, `delta_masked_l1`, `delta_full_ssim` — improvement over doing nothing
- `mean_seconds`, `sec_per_image`, `eval_wall_seconds`
- `ART`: `eval/per_sample.json`, `eval/grid_*_input_mask_pred_target.png`

Result: classic `masked_psnr` **17.19** · AIR v6 **21.38** · baseline **9.84** (identical baseline
on both → same 8 samples evaluated).

---

## ⚠️ 6. Methodological gaps — read before quoting any of this

1. **There is no held-out test set.** `OTR_easy` is used as the *validation* split during training
   **and** as the *evaluation* split afterwards. Those numbers are therefore not test metrics.
   `OTR_hard` is only used for sample generation. Fix: evaluate on `OTR_hard`, or carve a third split.
2. **No ablation against the un-fine-tuned base model.** Beating the "do nothing" baseline shows
   inpainting works, *not* that our 10 training steps helped — base SD2-inpainting is already
   competent. The missing run is base-model-only vs fine-tuned on the same samples.
3. **Precision/recall are not computed and do not apply** to image inpainting. The task metric for
   Overlay Text Removal is **OCR residual-text rate** (where precision/recall legitimately return);
   it needs an OCR dependency and ground-truth word boxes, neither wired up.
4. **No batch inference at scale.** The eval job loops over 8 images in one process. Real batch
   scoring: `mlflow.pyfunc.spark_udf` on classic, or Ray Data / `@distributed` sharding on AIR
   (no Spark on the AIR worker).
5. **Training notebooks log zero MLflow params** (`param count: 0`), so hyperparameters are not
   recoverable from the run records — only from job specs and notebook presets.
6. **Synthetic data, 10 steps.** Nothing here is a model-quality result.
7. **No Model Serving endpoint** was created from either registered UC model.
8. **Parquet format is staged but unused** by any training notebook.
