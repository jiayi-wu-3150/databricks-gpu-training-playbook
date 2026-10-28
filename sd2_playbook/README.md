# SD2 Inpainting on Databricks GPUs — a validated playbook

Fine-tune Stable Diffusion 2 inpainting on **both** Databricks GPU runtimes, with the
differences between them made explicit rather than papered over.

Every claim here was **measured** in `fevm-classic-stable-qxxm2v` on 2026-10-27/28.
Measured facts are date-stamped, because they rot.

## Run order

| Step | Notebook | Compute | Produces |
|---|---|---|---|
| 00 | `00_prepare_data` | any UC-enabled | raw PNG triplets + **sharded Parquet**, 3 splits |
| 01 | `01_stage_base_model` | any GPU | base SD2 pinned by revision → **UC registered model** |
| 02a | `02a_finetune_serverless` | AIR **env v6**, `GPU_8xH100` | fine-tuned UNet + checkpoints |
| 02b | `02b_finetune_classic` | DBR **18 LTS** ML GPU, 4×A10G | fine-tuned UNet + checkpoints |
| 03 | `03_eval_metrics` | 1 GPU either runtime | **test-split** metrics + base-model ablation |
| 04 | `04_register_finetuned` | 1 GPU either runtime | UC model + `@champion` alias |
| 05a | `05a_batch_inference_serverless` | AIR env v6 | sharded batch scoring (no Spark) |
| 05b | `05b_batch_inference_classic` | DBR 18 LTS | `spark_udf` batch scoring |
| 06 | `06_serving_endpoint` | serverless | GPU endpoint — **created, validated, then deleted** |

Classic and serverless are deliberately **separate files**. They differ in launcher, data loader,
and dependency pins; one file with branches hid those differences and made both harder to read.

## Prerequisites

- Unity Catalog schema + volumes. This playbook uses
  `classic_stable_qxxm2v_catalog.sd2_playbook` with volumes `data` and `checkpoints`.
  **The volumes must exist before any notebook runs** — the first path segment under
  `/Volumes/{cat}/{schema}/` must be a *registered volume*; `os.makedirs` cannot create one and
  fails with the unhelpful `OSError: [Errno 95] Operation not supported`.
- Teardown is one statement: `DROP SCHEMA classic_stable_qxxm2v_catalog.sd2_playbook CASCADE`.
- Classic: an all-purpose GPU cluster. See the compute-setup skill for the exact config
  (single-node profile, `SINGLE_USER` access mode, `NCCL_P2P_DISABLE` on G5).
- Serverless: nothing to provision — but **check the required Preview features**
  first. Running **AI v6** in a Job needs the *workspace base environment for jobs* preview (Beta);
  without it you silently get **Standard v6** instead. This playbook is written to work with no
  previews enabled.

## The data is synthetic, on purpose

`00_prepare_data` draws text overlays onto public images to synthesise an Overlay-Text-Removal
task (`input` = text on image, `target` = clean, `mask` = text region). This exercises every
stage of the pipeline in minutes instead of hours.

**No metric here is a model-quality result.** For a defensible quality claim, swap in the real
[OTR dataset](https://huggingface.co/datasets/cyberagent/OTR) (~80 GB) — the data-transformation
guide documents the switch, which is a path change, not a rewrite.

## Documentation

The authoring guides that accompanied this playbook — GPU compute setup and required preview
features, data-transformation options, UC model registration, inpainting metrics, serving
lifecycle, and the error-string catalogue — are **not in this repo**. They ship separately as
agent skills.

What survives here is the part that matters most: every notebook carries its own rationale in
markdown cells, and the measured results are in the [top-level README](../README.md).

## One pin set, and the validated alternative

Both runtimes run **`diffusers==0.39.0`** with DBR's own `transformers 4.x`. Measured:
classic `274618671165412`, serverless `452988514438708`.

| | Classic (18 LTS) | Serverless (AIR env v6) |
|---|---|---|
| diffusers | `0.39.0` | `0.39.0` |
| transformers | 4.57.x (DBR's, unpinned) | 4.57.x |
| Launcher | `TorchDistributor`, function mode | `@distributed` |
| Data loader | `pyarrow` + `DistributedSampler` | `UCVolumeDataset` + `air DataLoader` |

So the **only** differences between the runtimes are the launcher and the loader. Same data, same
model, same dependency set.

### Why 0.39 and not something current

Not because classic cannot do better — **it can**, and we measured it. On DBR 18 LTS, forcing
`diffusers==0.42.0` + `transformers>=5` + `huggingface_hub>=1.32,<3` is **functional**: run
`1097365734409471` loaded the real pipeline (81.2 s) and ran 8 denoising steps (2.12 s) on an A10G,
with `transformers 5.21.0` / `hub 2.3.0`.

The cost is that it **overrides DBR's own ML stack** and leaves it inconsistent:

```
datasets 4.3.0              requires huggingface-hub<2.0   -> violated
sentence-transformers 5.1.2 requires transformers<5        -> violated
```

Harmless for *this* workload (we use `pyarrow`, never `datasets` or `sentence-transformers`) but a
landmine for anyone who later adds those to the same notebook. And SD2 inpainting gains nothing
from 0.40+ — `StableDiffusionInpaintPipeline` has been stable for years.

So: **`0.39` is the default because it costs nothing and breaks nothing.** If you need current
diffusers, the forced path is validated — see the gotchas catalogue for both failure modes you will
hit on the way (lifting only `hub` does *not* work).
