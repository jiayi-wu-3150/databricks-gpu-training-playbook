# SD2 Inpainting on Databricks GPUs — a validated playbook

Fine-tune Stable Diffusion 2 inpainting on **both** Databricks GPU runtimes — AI Runtime
serverless GPU and classic GPU compute — with the differences between them made explicit rather
than papered over.

Seven steps: prepare data → stage the base model in Unity Catalog → fine-tune → evaluate →
register → batch-score → serve. Every step was **executed** on a real workspace
(`fevm-classic-stable-qxxm2v`, 2026-10-28) and every number below is measured, not estimated.
Measured facts are date-stamped, because they rot.

![Input with synthetic text overlays, the text mask, the fine-tuned prediction, and the clean target](example_eval_grid.png)

*Step 03, one test sample — **input** (synthetic overlays on a real food101 photo) · **mask** ·
**prediction** · **target**. Base SD2 already scores 17.540 dB here, so this shows the pipeline
runs, not that fine-tuning helped.*

## What this is, and what it is not

**This repo demonstrates capability, not performance.** It shows that the whole lifecycle — data
prep, fine-tune, eval, UC registration, batch scoring, serving — runs end to end on both
Databricks GPU runtimes, and what the code for each one looks like. That is the deliverable.

**Do not read the numbers as a runtime comparison.** The two runtimes are sized differently on
purpose, so each is used the way you actually would use it:

| | Serverless | Classic |
|---|---|---|
| GPUs | **8 × H100 80GB** | **4 × A10G 24GB** |
| Global batch | 64 | 32 |
| Rows batch-scored | 64 | 16 |

Different GPU class, different GPU count, different batch size, different row count. The ~3.2×
gap in steps/sec says almost nothing about the two runtimes and almost everything about 8×H100
versus 4×A10G. A like-for-like benchmark would need identical accelerators and identical batch
sizes, and that is deliberately not what this is.

**The task is synthetic; the images are real.** The base photographs are
[food101](https://huggingface.co/datasets/ethz/food101) — `00_prepare_data` draws the text
overlays and masks onto them, so the *inpainting task* is synthesised while the imagery is not.
See [`prep/`](prep/) for how those photographs get staged. The fine-tune is a 10-step smoke run,
not a convergence run, and **no metric here is a model-quality result.**

The one number that *is* meant to match across runtimes is the base-model eval arm — **17.540 dB
on both** — because that arm is the same model, same seed, same 8 samples. That it matches is the
evidence the eval harness is sound. Everything else is intentionally unmatched.

## Run order

| Step | File | Compute | Produces |
|---|---|---|---|
| — | [`prep/`](prep/) | AIR `GPU_1xA10` | **run once per workspace** — stages food101 into a volume |
| 00 | `00_prepare_data.py` | serverless CPU | raw PNG triplets + **sharded Parquet**, 3 splits |
| 01 | `01_stage_base_model.py` | any GPU | base SD2 pinned by revision → **UC registered model** |
| 02a | `02a_finetune_serverless.py` | AIR **env v6**, `GPU_8xH100` | fine-tuned UNet + checkpoints |
| 02b | `02b_finetune_classic.py` | DBR **18 LTS** ML GPU, 4×A10G | fine-tuned UNet + checkpoints |
| 03a | `03a_eval_serverless.py` | 1 GPU, serverless | **test-split** metrics + 3-arm ablation |
| 03b | `03b_eval_classic.py` | 1 GPU, classic | same, on classic |
| 04a | `04a_register_serverless.py` | 1 GPU, serverless | UC model + `@champion`, promotion gate |
| 04b | `04b_register_classic.py` | 1 GPU, classic | same, on classic |
| 05a | `05a_batch_inference_serverless.py` | AIR env v6 | sharded batch scoring (no Spark) |
| 05b | `05b_batch_inference_classic.py` | DBR 18 LTS | `spark_udf` batch scoring |
| 06 | `06_serving_endpoint.py` | serverless | GPU endpoint — created, validated, then **deleted** |

Job specs for all of the above are in [`conf/`](conf/).

Classic and serverless are deliberately **separate files**. They differ in launcher, data loader,
and dependency pins; one file with branches hid those differences and made both harder to read.

## Measured results

Synthetic data, and a 10-step smoke run rather than a convergence run. These are **capability
evidence** — proof each step ran, and roughly what it costs in wall-clock on the stated hardware.
They are not a benchmark, and not comparable across the two columns.

**Data (00)** — 512 px, seed 17: train 256 samples / 16 Parquet shards, val 64 / 4, test 64 / 4.
Raw PNGs are kept alongside the Parquet so a notebook can display samples.

**Base model (01)** — `stable-diffusion-2-inpainting` at pinned revision `5f74973`, 2.58 GB
pipeline. Download 6.2 s, UC register 111.3 s, UC load 76.2 s, 8-step predict 0.68 s, round-trip OK.

**Fine-tune (02)**

| | Serverless | Classic |
|---|---|---|
| Compute | `GPU_8xH100`, AI env v6 | `g5.12xlarge`, DBR 18 LTS ML |
| Launcher | `databricks.air.distributed` | `TorchDistributor` (function mode) |
| Data loader | `UCVolumeDataset` + `air.DataLoader` | `pyarrow` + `DistributedSampler` |
| World size | 8 × H100 80GB | 4 × A10G |
| Global batch | 64 (bs 4 × accum 2) | 32 (bs 1 × accum 8) |
| Train / val loss | 0.009198 / 0.016009 | 0.003951 / 0.003910 |
| Train seconds | 14.47 | 46.34 |
| Steps/sec | 0.691 | 0.216 |
| Peak GPU mem | 22.83 GB | 22.63 GB |

**Not a benchmark.** 8×H100 against 4×A10G is twice the GPUs of a much newer class, at twice the
global batch — the wall-clock gap is dominated by hardware, not by runtime. Both columns are
*correct* configurations for their runtime, and that is the whole point.

**Evaluation (03)** — 8 test samples, 50 denoise steps, **seed 1234**, masked-region PSNR:

| Arm | PSNR (dB) |
|---|---|
| Baseline (masked input, no inpainting) | 10.872 |
| Base SD2 | **17.540** — identical across both runtimes |
| Fine-tuned (classic) | 17.705 → +0.165 vs base |
| Fine-tuned (serverless) | 17.456 → −0.084 vs base |

Both deltas fall inside the **0.25 dB noise margin**, so the verdict is **INCONCLUSIVE** — exactly
what a 10-step run should produce. That the base arm scores 17.540 on *both* runtimes is the
evidence the sampler is properly seeded; unseeded, this number drifts and the ablation ends up
measuring sampling noise instead of the model. ~3.05 s/image.

**Register to UC (04)** — both runtimes, version 1 at `@champion`, 2.58 GB:

| | register | UC load | predict |
|---|---|---|---|
| Serverless | 60.9 s | 50.7 s | 0.65 s |
| Classic | 110.7 s | 74.2 s | 0.68 s |

These two columns ran on different node types, so the delta is not a runtime property either.
Registration is for **governance, not speed** — UC load is slower than a local checkpoint. What it
buys: versioning, aliases, lineage, access control, no HF egress at inference, and Model Serving
compatibility.

**Batch inference (05)**

| | Serverless | Classic |
|---|---|---|
| Mechanism | `@distributed`, `rows[rank::world]` | `mlflow.pyfunc.spark_udf` |
| Rows | 64 across 8 ranks | 16 across 4 partitions |
| Wall | 43.5 s | 116.8 s |
| Throughput | 1.471 img/s | 0.137 img/s |
| Balance | exactly 8 rows/rank, 0.512–0.561 img/s | — |

**Again not a comparison** — 64 rows on 8×H100 against 16 rows on 4×A10G, via different
mechanisms. The useful reading: `spark_udf` is the
right *shape* for classic batch scoring but needs a **multi-node** GPU cluster; on a single node,
Spark tasks contend for the same GPUs and most of the hardware sits idle.

**Serving (06)** — `GPU_MEDIUM`: ready in **1565.8 s (26.1 min)**, invocation 1.79 s, 512×512
prediction validated, endpoint **deleted**. Budget ~26 min for a 2.58 GB diffusion pipeline to go
live, and always tear down in a `finally`.

## Prerequisites

- Unity Catalog schema + volumes. This playbook uses
  `classic_stable_qxxm2v_catalog.sd2_playbook` with volumes `data` and `checkpoints`.
  **The volumes must exist before any step runs** — the first path segment under
  `/Volumes/{cat}/{schema}/` must be a *registered volume*; `os.makedirs` cannot create one and
  fails with the unhelpful `OSError: [Errno 95] Operation not supported`.
- The source images must be staged first — see [`prep/`](prep/). Step 00 reads
  `/Volumes/{catalog}/dl_training/data/food101_shards` and fails if it is empty.
- Teardown is one statement: `DROP SCHEMA <catalog>.sd2_playbook CASCADE`. The `dl_training`
  schema holding the source images is deliberately separate, so teardown does not destroy it.
- Classic: an all-purpose GPU cluster — single-node profile, `SINGLE_USER` access mode,
  `NCCL_P2P_DISABLE=1` on G5.
- Serverless: nothing to provision — but running **AI v6** in a Job needs the *workspace base
  environment for jobs* preview (Beta); without it you silently get **Standard v6**, which has no
  torch. This playbook is written to work with no previews enabled.

## Where the data comes from, and the synthetic task

Two stages, and the distinction matters:

1. **Real photographs.** [`prep/`](prep/) stages 20,000 [food101](https://huggingface.co/datasets/ethz/food101)
   images from Hugging Face into `/Volumes/{catalog}/dl_training/data/food101_shards` as 27
   parquet shards. Run once per workspace; steps 00–06 are self-contained after that.
2. **Synthetic task.** `00_prepare_data` reads those photographs and draws text overlays onto
   them, producing an Overlay-Text-Removal triplet (`input` = text on image, `target` = clean,
   `mask` = dilated text bounding box). The *overlays and masks* are generated; the underlying
   imagery is not.

So "synthetic" here means the **task** is synthesised, not that pixels were invented. That is what
makes it exercise every stage of the pipeline in minutes instead of hours.

For a defensible quality claim, swap in the real
[OTR dataset](https://huggingface.co/datasets/cyberagent/OTR) (~80 GB); it is a path change, not a
rewrite. Nothing downstream of step 00 needs to change.

## One pin set, and the validated alternative

Both runtimes run **`diffusers==0.39.0`** with DBR's own `transformers 4.x`.

| | Classic (18 LTS) | Serverless (AIR env v6) |
|---|---|---|
| diffusers | `0.39.0` | `0.39.0` |
| transformers | 4.57.x (DBR's, unpinned) | 4.57.x |
| Launcher | `TorchDistributor`, function mode | `@distributed` |
| Data loader | `pyarrow` + `DistributedSampler` | `UCVolumeDataset` + `air.DataLoader` |

So the **only** differences between the runtimes are the launcher and the loader. Same data, same
model, same dependency set.

### Why 0.39 and not something current

Not because classic cannot do better — **it can**, and we measured it. On DBR 18 LTS, forcing
`diffusers==0.42.0` + `transformers>=5` + `huggingface_hub>=1.32,<3` is **functional**: it loaded
the real pipeline in 81.2 s and ran 8 denoising steps in 2.12 s on an A10G, with
`transformers 5.21.0` / `hub 2.3.0`.

The cost is that it **overrides DBR's own ML stack** and leaves it inconsistent:

```
datasets 4.3.0              requires huggingface-hub<2.0   -> violated
sentence-transformers 5.1.2 requires transformers<5        -> violated
```

Harmless for *this* workload (we use `pyarrow`, never `datasets` or `sentence-transformers`) but a
landmine for anyone who later adds those to the same notebook. And SD2 inpainting gains nothing
from 0.40+ — `StableDiffusionInpaintPipeline` has been stable for years.

So **`0.39` is the default because it costs nothing and breaks nothing.** Note that lifting
`huggingface_hub` alone does *not* work; you must lift `transformers` too, or you get an
`ImportError` on `resolve_revision`.

## Other constraints worth knowing

**Shard count is not free.** `UCVolumeDataset` shards the **file list** across `(rank, worker)`
streams, so you need `n_files >= world_size × num_workers`. Too few shards and some streams get
nothing, and DDP **hangs** at the next all-reduce rather than erroring. `02a` asserts this up front.

**`@distributed` needs notebook context.** In an `air run` CLI job it calls
`set_databricks_credentials_from_dbutils()` and fails with
`cluster_id is required in the configuration`. Use `torchrun` for CLI jobs — AIR injects
`NUM_NODES` / `NODE_RANK` / `MASTER_ADDR` / `MASTER_PORT`.

**UC registration needs an explicit signature.** There is no `mlflow.diffusers` flavor, so step 04
wraps the pipeline in a custom PyFunc with an explicit `ModelSignature` *and* `ParamSchema`.
Without it, UC rejects the model.

**8 contiguous GPUs on classic is a capacity question, not a billing one.** Serverless pools
across tenants and hands back 8×H100 in seconds; a classic cluster asks for 8 contiguous GPUs in
one AZ and can take 10–14 minutes to find out it cannot have them. Set `"zone_id": "auto"`.


## Notes

- Workspace-specific values (catalog, volume paths, cluster IDs, workspace host) are inline as
  **run** rather than parameterized, so the numbers above reproduce as-is. Change them for your
  own workspace.
- The authoring best-practice guides are intentionally not in this repo; they ship separately as
  agent skills.
