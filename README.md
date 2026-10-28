# Databricks GPU Training Playbook

Working, end-to-end GPU fine-tuning on Databricks across **two runtimes** — AI Runtime
serverless GPU and classic GPU compute — with every step executed on a real workspace and the
measured numbers recorded.

The headline deliverable is [`sd2_playbook/`](sd2_playbook/): Stable Diffusion 2 **inpainting**
fine-tuned, evaluated, registered to Unity Catalog, batch-scored, and served — once per runtime,
as separate jobs.

## What's in here

| Folder | What it is |
|---|---|
| [`sd2_playbook/`](sd2_playbook/) | **Start here.** 7-step SD2-inpainting pipeline, both runtimes, all steps verified |
| [`gpu_training_demo/`](gpu_training_demo/) | ViT image-classification demo + 3-hour workshop agenda. Single-GPU, DDP, TorchDistributor, DeepSpeed |
| [`diffusion_inpainting_demo/`](diffusion_inpainting_demo/) | Earlier inpainting iteration, plus the environment **probes** used to bisect the dependency conflicts |
| [`gpu_training_playbook/`](gpu_training_playbook/) | Vision / LLM-LoRA / VLM-LoRA fine-tuning configs, `FINDINGS.md`, and a slide deck |

## The 7 steps, and where each runs

Each step is a separate job so the two runtimes never share state.

| Step | Serverless (AI Runtime) | Classic GPU |
|---|---|---|
| 00 Prepare data | `00_prepare_data.py` — shared, writes raw PNGs + Parquet shards | ← same |
| 01 Stage base model | `01_stage_base_model.py` — HF → UC, pinned revision | ← same |
| 02 Fine-tune | `02a` `@distributed` + `UCVolumeDataset` | `02b` `TorchDistributor` + `pyarrow` |
| 03 Evaluate | `03a` seeded 3-arm ablation | `03b` same |
| 04 Register to UC | `04a` PyFunc + signature + promotion gate | `04b` same |
| 05 Batch inference | `05a` `@distributed`, manual row sharding | `05b` `mlflow.pyfunc.spark_udf` |
| 06 Serving endpoint | `06_serving_endpoint.py` — create → validate → **always delete** | ← same |

## Measured results

All numbers from a single pass on a Databricks workspace, 2026-10-28. Synthetic data; a
10-step smoke run, not a convergence run.

**Data (step 00)** — 512 px, seed 17: train 256 samples / 16 Parquet shards, val 64 / 4, test 64 / 4.
Raw PNGs kept alongside Parquet so notebooks can display samples.

**Base model (step 01)** — `stable-diffusion-2-inpainting` at pinned revision `5f74973`, 2.58 GB
pipeline. Download 6.2 s, UC register 111.3 s, UC load 76.2 s, 8-step predict 0.68 s, round-trip OK.

**Fine-tune (step 02)**

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

Not a benchmark: different GPUs, batch sizes, and world sizes. Both are *correct* configurations
for their runtime, which is the point.

**Evaluation (step 03)** — 8 test samples, 50 denoise steps, **seed 1234**, masked-region PSNR:

| Arm | PSNR (dB) |
|---|---|
| Baseline (masked input, no inpainting) | 10.872 |
| Base SD2 | **17.540** — identical across both runtimes |
| Fine-tuned (classic) | 17.705 → +0.165 vs base |
| Fine-tuned (serverless) | 17.456 → −0.084 vs base |

Both deltas fall inside the **0.25 dB noise margin**, so the verdict is **INCONCLUSIVE** — exactly
what a 10-step run should produce. That the base arm scores 17.540 on both runtimes is the
evidence the sampler is properly seeded; unseeded, this number drifts and an ablation ends up
measuring sampling noise instead of the model. ~3.05 s/image.

**Register to UC (step 04)** — both runtimes, version 1 at `@champion`, 2.58 GB:

| | register | UC load | predict |
|---|---|---|---|
| Serverless | 60.9 s | 50.7 s | 0.65 s |
| Classic | 110.7 s | 74.2 s | 0.68 s |

**Batch inference (step 05)**

| | Serverless | Classic |
|---|---|---|
| Mechanism | `@distributed`, `rows[rank::world]` | `mlflow.pyfunc.spark_udf` |
| Rows | 64 across 8 ranks | 16 across 4 partitions |
| Wall | 43.5 s | 116.8 s |
| Throughput | 1.471 img/s | 0.137 img/s |
| Balance | exactly 8 rows/rank, 0.512–0.561 img/s | — |

Again not a comparison — different row counts and GPUs. The useful reading: `spark_udf` is the
right *shape* for classic batch scoring but needs a **multi-node** GPU cluster; on one node, Spark
tasks contend for the same GPUs and most of the hardware sits idle.

**Serving (step 06)** — `GPU_MEDIUM`: ready in **1565.8 s (26.1 min)**, invocation 1.79 s,
512×512 prediction validated, endpoint **deleted**. Budget ~26 min for a 2.58 GB diffusion
pipeline to go live, and always tear down in a `finally`.

## Constraints worth knowing before you start

**Pin set.** `diffusers==0.39.0` with `transformers<5`. The chain is
`mosaicml-streaming` → `transformers<5` → `huggingface_hub<1.0` → `diffusers<=0.39`. Classic *can*
run `diffusers 0.42` if you lift `transformers` too — verified — but 0.39 has the smaller blast
radius. The probes that established this are in `diffusion_inpainting_demo/9*_*probe*.py`.

**Shard count is not free.** `UCVolumeDataset` shards the **file list** across
`(rank, worker)` streams, so you need `n_files >= world_size × num_workers`. Too few shards and
some streams get nothing, and DDP **hangs** at the next all-reduce rather than erroring. `02a`
asserts this up front.

**`@distributed` needs notebook context.** In an `air run` CLI job it calls
`set_databricks_credentials_from_dbutils()` and fails with
`cluster_id is required in the configuration`. Use `torchrun` for CLI jobs — AIR injects
`NUM_NODES` / `NODE_RANK` / `MASTER_ADDR` / `MASTER_PORT`.

**UC registration needs an explicit signature.** There is no `mlflow.diffusers` flavor, so these
steps wrap the pipeline in a custom PyFunc with an explicit `ModelSignature` *and* `ParamSchema`.
Without it, UC rejects the model.

**8 contiguous GPUs on classic is a capacity question, not a billing one.** Serverless pools
across tenants and hands back 8×H100 in seconds; a classic cluster asks for 8 contiguous GPUs in
one AZ and can take 10–14 minutes to find out it cannot have them. Set `"zone_id": "auto"`.

## Notes

- Workspace-specific values (catalog, volume paths, cluster IDs, workspace host) are inline as
  **run** rather than parameterized, so the numbers above are reproducible as-is. Change them for
  your own workspace.
- The authoring best-practices docs are intentionally not in this repo; they ship separately as
  agent skills.
