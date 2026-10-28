# SD2 inpainting: Classic GPU vs AI Runtime v6 — side by side

Both runs: SD2 inpainting fine-tune, **same source data** (320 synthetic OTR-shaped samples
derived from Food-101), `smoke_test=true` → 10 optimizer steps. Validated 2026-10-27/28 in
`fevm-classic-stable-qxxm2v`.

## Configuration

| | Classic GPU | AI Runtime v6 |
|---|---|---|
| Compute | all-purpose `g5.12xlarge` | serverless `GPU_8xH100` |
| GPUs | 4 × A10G (24 GB) | 8 × H100 (80 GB) |
| Runtime | DBR 18.x ML GPU | Standard v6 (`environment_version: "6"`) |
| Cluster management | create + wait (~11 min to acquire) | none |
| Launcher | `TorchDistributor(local_mode=True)`, function mode | `@distributed(num_accelerators=8)`, co-located |
| Data format | MDS shards | one `.npz` per sample on a UC Volume |
| Data loader | `StreamingDataset` | `UCVolumeDataset` + `databricks.air.data.DataLoader` |
| diffusers | **0.39.0** (capped) | **0.42.0** |
| transformers | 4.57.6 (DBR's) | 5.x |
| huggingface_hub | 0.36.2 | ≥1.32 |
| torch | 2.9.0+cu129, preinstalled | pip-installed via `databricks-sdk-air[data]` |
| Per-GPU batch × accum | 1 × 8 | 4 × 2 |
| Global batch | 32 | 64 |

## Measured

| | Classic GPU | AI Runtime v6 |
|---|---|---|
| Run | `274618671165412` | `249353804464778` |
| MLflow run | `4948459ffbec4d13b721f4286322d8cc` | `e7e73a5fbca94b98b33a2809b893e674` |
| `train/steps_per_sec` | 0.2464 | **1.5815** |
| `train/total_minutes` | 1.513 | **0.935** |
| `train/loss` @ step 10 | 0.14698 | 0.00879 |
| `val/loss` | 0.004544 | 0.010891 |
| Peak GPU mem / device | ~16.08 GB × 4 | ~17.64 GB |
| Inference samples written | 1 PNG | 4 PNGs |
| Checkpoint | step 10 | step 10 |

## ⚠️ This is NOT a benchmark — read before quoting any number

The two sides differ in **at least six** confounded ways: GPU count (4 vs 8), GPU generation
(A10G vs H100), global batch (32 vs 64), `diffusers` major-minor, `transformers` major version,
and the data loader. No single metric difference has one cause.

- **`train/steps_per_sec` 6.4x** is unsurprising with 2x the GPUs on much faster silicon and 2x the
  global batch. It is not a per-GPU efficiency measurement.
- **`train/loss` 0.147 vs 0.0088** is not a quality signal. Step-10 loss is extremely noisy, the
  global batch differs 2x, and the library versions differ.
- **`val/loss` moves the OPPOSITE way** (0.0045 classic vs 0.0109 AIR). That alone proves the two
  are not comparable: validation sampling differs between the loaders (different file ordering and
  random-access pattern). Do not present either as "better".
- Both are **10 steps on 320 synthetic samples**. Neither is a training result, and the data is
  text drawn on food photos — not the real ~80 GB OTR dataset.

## What the comparison DOES legitimately show

1. **The same model and the same data train end to end on both runtimes** — load, distributed
   init, MLflow tracking, checkpointing, and post-training inference all complete. That was the goal.
2. **The library stacks differ structurally, not incidentally.** Classic is *capped* at
   `diffusers 0.39` by a dependency chain: Mosaic Streaming requires `transformers<5`, which caps
   `huggingface_hub<1.0`, which caps `diffusers<=0.39`. AI Runtime, using `UCVolumeDataset`
   instead, runs `diffusers 0.42` on transformers 5.x. **This is the most portable finding.**
3. **Operationally they are not close.** Classic took ~11 minutes to acquire `g5.12xlarge` before a
   single step ran, and the cluster needs lifecycle management. AI Runtime had no cluster to create.
   For short, iterative work that gap dominates anything in the table above.
4. **The launcher is the only part of the training code that changes.** The training function is
   the same shape on both; `TorchDistributor` function mode vs `@distributed`.

## Honest gaps

- AI Runtime v6 here is **Standard v6**, not **AI v6**. `environment_version: "6"` selects the
  minimal GPU image (cuda + `databricks-sdk-air`); the PyTorch/Transformers-preloaded **AI v6**
  requires `base_environment: databricks_ai_v6`, which this workspace rejects
  ("Only custom base environments ... are currently supported"). torch therefore arrives by pip.
- Neither notebook logs hyperparameters to MLflow (`param count: 0`), so the configuration above
  comes from the job specs and notebook presets, not from the run records.
- Classic startup (~11 min) was measured when the cluster was created, not instrumented in this run.
