# Training on Databricks GPUs — 3-hour workshop

Demo assets: the five notebooks in this folder, all run green on `fevm-classic-stable-qxxm2v`.
Every timing below assumes **compute is pre-warmed** (see Prerequisites).

| Time | Min | Segment | Covers | Asset |
|---|---|---|---|---|
| 0:00 | 15 | **Where training on Databricks stands — and the then→now map** | Overview | slides |
| 0:15 | 20 | **Pick your path: AI Runtime vs classic** | Advantages / choosing | slides |
| 0:35 | 30 | **Hands-on 1 — serverless GPU, single GPU** | Setup knobs, templates, data | `01` |
| 1:05 | 10 | *Break* | | |
| 1:15 | 35 | **Hands-on 2 — multi-GPU on both runtimes** | Setup knobs, templates | `02` + `03` |
| 1:50 | 20 | **Hands-on 3 — DeepSpeed when memory is the limit** | Templates, techniques | `04` |
| 2:10 | 30 | **The unvarnished truths** | Unvarnished truths | `RESULTS.md` |
| 2:40 | 20 | **Techniques + take-home** | Data, packages, Q&A | `README.md` |

---

### 0:00 — Where training on Databricks stands, and the then→now map (15 min)
Open with the deprecation map, because the room will arrive with the old mental model:

| They know | Current | Status of the old thing |
|---|---|---|
| HorovodRunner / Horovod | `TorchDistributor` (classic), `@distributed` (AIR) | deprecated, not preinstalled after 15.4 LTS ML |
| Petastorm | Mosaic Streaming (classic), `UCVolumeDataset` (AIR) | deprecated |
| HyperOpt | Optuna / Ray Tune | unmaintained, removed after 16.4 LTS ML |
| FM Fine-tuning / `databricks_genai` | AI Runtime | end of life |

### 0:15 — Pick your path (20 min)
The decision table, not a benchmark. What Databricks recommends (AI Runtime for GPU deep learning,
fine-tuning, batch inference) **and** the honest reverse case: account separation, Spark in the same
cluster, Ray-on-Spark, a library AIR lacks, GPU types AIR lacks (T4/L4/L40S/A100/V100), or a
compliance profile AIR doesn't cover. State plainly that **neither path has a capacity SLA**.

### 0:35 — Hands-on 1: serverless single GPU (30 min) — `01`
Attach live: Environment panel → **AI v6** → attach `GPU_1xA10`. ~40 s to compute.
Show `UCVolumeDataset`, `databricks.air.data.DataLoader`, MLflow. **Measured: 1,536 imgs, 27.1 s.**

### 1:15 — Hands-on 2: multi-GPU, both runtimes (35 min) — `02` + `03`
Run `02` on `GPU_8xH100` (`@distributed()` → `.distributed(...)`, 8 NCCL ranks, DCP checkpoint via
`UCVolumeWriter`) and `03` on the classic 4×A10G cluster (`TorchDistributor` **function mode**).
The point: `train_fn` is the same shape; **only the launcher differs**. Hard limits to say out loud:
`@distributed` needs **8xH100/8xB300**, is **single-node**, and multi-node means CLI + `torchrun`.

### 1:50 — Hands-on 3: DeepSpeed (20 min) — `04`
`DeepspeedTorchDistributor`, ZeRO-2. Note the **camelCase** args. Why reach for it: memory, not
speed. **Measured: 4.41 GB peak vs 8.81 GB on plain DDP, and the fastest run at 269 img/s.**

### 2:10 — The unvarnished truths (30 min) — the most valuable half-hour
Every item below is a failure we actually hit and fixed, with the error text in `RESULTS.md`:
1. `@distributed` requires **notebook context** — in an `air run` CLI job it dies with
   `cluster_id is required in the configuration`. Works fine in a notebook.
2. `TorchDistributor` **function mode breaks under `spark_python_task`** (module re-execution
   re-triggers the launcher). Notebook function mode is the documented path.
3. DeepSpeed `{"type": "AdamW"}` → **FusedAdam JIT build failure** on DBR 18 ML GPU. Fix:
   `"torch_adam": True`.
4. **The 8×H100 run was 99% data-bound** — 206.8 s of 208.0 s outside the loader. Use it to teach
   that a throughput table can be measuring your decode loop, not the GPU.
5. **`transformers` 4.57 (classic) vs 5.8 (AIR)** — a major version apart. Pin explicitly.
6. Serverless **headless job specs** are full of traps (`compute_type` isn't a field;
   `base_environment` may be gated; `environment_version: 4` has no `transformers`).

### 2:40 — Techniques and take-home (20 min)
Data: S3 vs UC Volumes; `UCVolumeDataset` auto-shards per `(rank, worker)` but needs a **many-file**
layout — one big parquet defeats it (hence `00`). Checkpoints: `UCVolumeWriter` + `dcp.save`, not
`torch.save`. Packages: pin across runtimes. Then hand over the repo and take questions.

---

## Prerequisites — do these before the room fills
- **Start the classic GPU cluster ~20 min early.** `g5.12xlarge` took ~11 min to acquire instances.
- **Run `00_stage_sharded_data` once** — the training notebooks assume `food101_shards` exists.
- **Have the completed MLflow runs open as a fallback.** Serverless GPU has no capacity guarantee,
  and the 8×H100 segment is the one that can't be improvised.
- Decide in advance whether `02` runs live (~3.5 min train + startup) or is walked through.

## If you need to trim
Cut **Hands-on 3** to 10 min (show `ds_config` + the memory number, skip the live run) and
**Pick your path** to 10 min. That frees 20 min, enough to absorb a late start or a capacity miss.
