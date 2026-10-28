# Training Deep Learning & LLMs on Databricks GPUs

Deck source. Organized by the **training lifecycle**, with runtime (AI Runtime vs classic
Databricks Runtime ML) as a swappable parameter rather than two parallel tracks.
Slides 90–99 are **back-pocket appendix** — not walked through unless asked.

---

## 1. Title
**Training Deep Learning & LLMs on Databricks GPUs**
A practitioner's walkthrough — setup, techniques, templates, and the unvarnished truths
Validated end-to-end on 2026-10-25 · all numbers measured, not quoted

---

## 2. Agenda
1. The two GPU paths, and how to choose
2. GPU training concepts — a shared vocabulary
3. The training lifecycle on Databricks
4. Setup walkthrough: which knobs, and which to leave alone
5. Techniques: data loading, checkpointing, tracking, registration
6. What's changed — packages no longer recommended
7. Templates you can start from today
8. Measured results
9. The unvarnished truths
10. Getting the model out

---

## 3. Two paths to a GPU

| | **AI Runtime** (serverless GPU) | **Classic** (Databricks Runtime ML) |
|---|---|---|
| What you manage | Nothing — pick an accelerator | A cluster: nodes, runtime, scaling, termination |
| Compute runs in | Databricks-managed plane | **Your own cloud account** |
| Time to first GPU | **~40 seconds** (measured) | **~10 minutes** (measured) |
| GPUs | A10, H100, B300 | T4, L4, A10G, L40S, V100, A100, H100 |
| Spark in the same job | No (Spark Connect only) | Yes, full cluster |
| Status | **GA Oct 15, 2026** (non-compliance-profile workspaces) | Long-established, LTS runtimes |

**Databricks' own recommendation**, quoted: for GPU deep learning, fine-tuning and batch inference,
*"Databricks recommends AI Runtime … instead of Runtime ML"* — citing lower TCO, a more flexible
developer experience, and multinode scale.

**Default to AI Runtime.** Choose classic when you have a specific reason — see slide 92.

---

## 4. GPU training concepts — the vocabulary
*(Concepts are durable; only the tooling churns.)*

- **Single-node, single-GPU** — where every project should start
- **Single-node, multi-GPU** — one machine, several GPUs, data-parallel
- **Multi-node** — several machines, GPUs synchronizing over the network

**Data parallelism (DDP):** replicate the model on each GPU → each reads a different data shard →
compute gradients → **average gradients across all GPUs** → update → repeat.

**Why the network matters:** that gradient averaging is a collective operation (ring all-reduce)
on every step. Each GPU sends its gradients to a neighbour and receives from another, so the
interconnect — not the GPU — becomes the ceiling at scale. This is why multi-node training is
sensitive to the fabric, and why RDMA-class interconnect is a real differentiator rather than a
bullet point.

**The scaling ladder — climb it in order, don't skip:**
1. Single node, single GPU — establish a correct, reproducible baseline
2. Fix the data pipeline if it's the bottleneck (often is — see slide 82)
3. Single node, multi-GPU
4. Multi-node
5. Hyperparameter search — only once the baseline is stable

---

## 5. The training lifecycle on Databricks

**Data** (UC Volumes / Delta) → **Train** (GPU compute) → **Track** (MLflow) →
**Register** (Unity Catalog) → **Deploy** (Model Serving *or* export off-platform)

Governance runs through the whole chain: the data, the experiment, the model, and the lineage
between them all live in Unity Catalog. That is the part a bare GPU cluster elsewhere doesn't give you.

---

## 6. Setup: AI Runtime — the knobs that exist

Three ways in: **notebook** (environment panel → pick accelerator → Apply), **CLI**
(`databricks air run --file workload.yaml`), **jobs/DABs** (`ai_runtime_task`).

```yaml
experiment_name: my-training-run      # bare name — NOT an absolute path
environment:
  version: 'databricks_ai_v6'         # AI image: torch preinstalled
  dependencies: [peft, trl]           # 15 GB total cap
compute:
  num_accelerators: 1
  accelerator_type: GPU_1xA10         # or GPU_1xH100 / GPU_8xH100 / GPU_8xB300
code_source:
  type: snapshot                      # nested form is required
  snapshot:
    root_path: .
command: python $CODE_SOURCE_PATH/train.py
max_retries: 0                        # set explicitly while developing
timeout_minutes: 60                   # default is 3 hours
```

| Knob | Guidance |
|---|---|
| `accelerator_type` | **1xA10 and 1xH100 are single-GPU only.** Multi-GPU needs 8xH100/8xB300 |
| `environment.version` | **`'databricks_ai_v6'` = AI image with torch. Plain `6` = Standard image, NO torch** |
| `num_accelerators` | Defaults to all GPUs on the node |
| `timeout_minutes` | Default 3 h. Set it — a hung job otherwise bills for hours |
| `max_retries` | **Defaults to retrying**, including deterministic failures. Set `0` while iterating |
| `dependencies` | Installed with `uv`. Notebooks cache the env; CLI/jobs reinstall every run |
| `secrets` | Use for `HF_TOKEN` — HF Hub egress works but is rate-limited unauthenticated |

**Tip:** `databricks air run -h config.<field>` is a built-in schema browser and is more current
than the docs page.

---

## 7. Setup: classic GPU — the knobs that matter

| Knob | Set to | Why |
|---|---|---|
| Policy | **Unrestricted** | The default *"Create with Personal Compute"* button allows exactly **one** GPU type (`g4dn.xlarge`, T4 16 GB). A10G is unreachable that way |
| **Machine learning** | **ON** | Gives DBR ML (torch/CUDA preinstalled) and forces Dedicated access mode, which UC access requires |
| Runtime | **18.3 LTS ML (GPU)** | LTS. Selecting a GPU node filters the runtime list |
| Node type | **g5.xlarge** (A10G 24 GB) | Match the GPU to the job; g6e (L40S 48 GB) for more VRAM; p5 (H100 80 GB) for large LLMs |
| Single node | **Yes**, for one GPU | GPU scheduling isn't enabled on single-node; torch sees the device directly |
| **Auto-terminate** | **20–30 min** | **The single most important cost knob.** Defaults here are 120 min, and Personal Compute defaults to 4320 min (3 days) |
| Photon | **OFF** | Not supported with GPU instance types at all |
| Spot | **OFF** | Spot GPUs are documented as "difficult to retain" |
| `spark.task.resource.gpu.amount` | GPUs per worker | Fractional values allowed for concurrent inference tasks |

**Job vs all-purpose compute:** use **job compute** for recorded runs — cheaper DBUs, terminates
automatically, spec lives in code. Use all-purpose only for the interactive debug loop, where a warm
cluster turns a 6-minute iteration into seconds.

**NCCL escapes worth knowing:** `NCCL_P2P_DISABLE=1` for CUDA peer-access errors on G5;
`NCCL_SOCKET_IFNAME=eth0` for multi-node NCCL failures.

---

## 8. Techniques: getting data to the GPU

| Source | AI Runtime | Classic |
|---|---|---|
| **UC Volumes (files)** | Plain filesystem paths; `databricks.air.data.UCVolumeDataset` + `DataLoader` | Filesystem paths (Dedicated access mode) |
| **Delta / tables** | Spark Connect → `toPandas()`; for large tables write parquet to a Volume, then `load_dataset("parquet", …)` | Spark DataFrame directly |
| **Large cloud datasets** | — | **Mosaic Streaming** (`StreamingDataset`, MDS format) |
| **S3 direct** | Not covered in docs | Not covered in docs |

**Two things to know:**
- **Mosaic Streaming appears nowhere in the AI Runtime docs.** The runtimes genuinely have different
  recommended data paths — this is not the same script with a different compute target.
- **File sizing matters**: aim for 1–10 GiB files. For many small files, copy in parallel first.
  Our measured ViT gap (slide 82) was a data-pipeline effect, not a GPU effect.

**Checkpointing:**
- AI Runtime recommends **PyTorch Distributed Checkpoint** — `dcp.save`/`dcp.load` with
  `UCVolumeWriter`/`UCVolumeReader`, staging through local NVMe. **Not `torch.save`.**
  `dcp.async_save` needs `init_process_group(backend="cpu:gloo,cuda:nccl")`.
- Always checkpoint: there is **no capacity SLA** (slide 11).
- Under `/Volumes/{cat}/{schema}/`, every first-level directory must be a **registered volume** or
  `os.makedirs` fails with `OSError [Errno 95]`.

---

## 9. Techniques: tracking and registration

- Set the experiment explicitly. AI Runtime derives one from `experiment_name`; classic
  **`.py` tasks get none at all** and fail with *"Could not find experiment with ID None."*
- HuggingFace `Trainer`: `report_to=["mlflow"]`. Lightning: `mlflow.pytorch.autolog()`.
- AI Runtime auto-logs **GPU/CPU/memory system metrics** and run logs (60-day retention). Classic
  does not do this automatically.
- Log metrics from **rank 0 only** in distributed runs.
- Register to UC: `mlflow.set_registry_uri("databricks-uc")`, name as `catalog.schema.model`.
  A signature is required.

**The trap we hit on both runtimes:** `mlflow.pytorch.log_model` fails on an Accelerate/AMP-prepared
model — `PicklingError: Cannot pickle a prepared model with automatic mixed precision` — and bf16
weights break signature inference.

```python
m = trainer.accelerator.unwrap_model(trainer.model)
m = m.to("cpu").to(torch.float32).eval()
mlflow.transformers.log_model(transformers_model={"model": m, "tokenizer": tok},
                              task="text-generation", name="model",
                              registered_model_name="cat.schema.my_model")
```

---

## 10. What changed — stop using these

| Was | Now | Status of the old thing |
|---|---|---|
| **HorovodRunner / Horovod** | `TorchDistributor` (classic), `@distributed` (AIR) | **Deprecated**; not preinstalled after 15.4 LTS ML |
| **Petastorm** | **Mosaic Streaming** (MDS) | **Deprecated**, explicitly |
| **HyperOpt** | **Optuna** (single node) / Ray Tune (distributed) | **Unmaintained**; removed from DBR ML after 16.4 LTS |
| **Foundation Model Fine-tuning** / `databricks_genai` | **AI Runtime** | **End of life** — and the DBR ML page still links it as current |
| `serverless_gpu` package | **`databricks.air`** | **Deprecated** (FutureWarning) |
| P3 / V100 instances | G5 / G6 / G6e / P4 / P5 | **Deprecated** |
| TensorFlow as top-tier | PyTorch | Not top-tier from **DBR 18.0 ML** |
| Workspace model registry | **UC registry + aliases** | "(legacy)" |

**Also: pin your versions.** The two runtimes ship a different **major** version of
`transformers` — **5.8.1** on AI Runtime v6 vs **4.57.1** on DBR 18 ML. Plus torch 2.11/CUDA 13.0
vs torch 2.9/CUDA 12.9. Code is not guaranteed to move between them unpinned.

**Watch out for stale docs.** The public `compute/gpu` page still claims CUDA 11.0 while DBR 17.3+
ships CUDA 12.6+; `train-model/pytorch` pins `torch==1.5.0` in its example; `dl-best-practices`
never mentions AI Runtime.

---

## 11. The unvarnished truths

**AI Runtime**
- **No capacity SLA and no startup SLA.** H100 capacity can be sporadic; on a miss the session sits
  in "starting" and fails, with **no durable queue**. Checkpoint and retry with backoff.
- **Cross-region bursting is real and not pinnable.** The docs say AIR "leverages cross-region GPUs
  in certain cases during moments of high demand," with possible egress costs. The workspace Geo
  setting enforces **Geo, not region**.
- **No customer-configurable GPU or spend caps**, and no reserved capacity today.
- **Access control is coarse**: serverless ACLs are per-object (Interactive / Automated), **not
  per-GPU** — you cannot express "GPU training only."
- **No self-service off switch** for AI Runtime post-GA.
- Default quota: **8 nodes × 8×H100 = 64 GPUs** per workspace.
- Notebook `@distributed` is **single-node**; multi-node requires the CLI + `torchrun`.
- `for_each` unsupported on serverless GPU jobs; dynamic `{{}}` refs don't work in `ai_runtime_task`;
  **1xH100 is not a valid serverless GPU *job* config** (use 1xA10 or 8xH100).
- Compute runs in the **Databricks-managed plane** — this cannot satisfy a cloud-account-separation
  requirement. No customer-hosted variant exists.

**Classic**
- **~10 minutes to a usable GPU**, every run.
- Idle clusters bill. Auto-terminate is the only guard.
- Spot GPUs are hard to retain; quota increases often needed.
- Photon doesn't apply to GPU instances.
- No automatic system-metrics capture into MLflow.

---

## 12. Templates to start from

**Official tutorials** (`machine-learning/ai-runtime/examples/tutorials/<name>`):
`sgc-finetune-qwen3.5-0.8b` (end-to-end fine-tune + serve), `sgc-finetune-qwen3-4b`,
`sgc-rl-gemma4-2b` (GRPO), `sgc-yolo11n-detect-coco128` (CV), `sgc-ray-resnet18`, `sgc-tabfm`,
`sgc-raydata-vllm-batch-inference`, `sgc-whisper-batch-inference`

**Cookbook:** `github.com/databricks-solutions/ai-runtime-cookbook`
— `getting-started/air-cli/`, `training/`, `post-training/`, `ray/`, `batch-inference/`, `serving/`

**This playbook** — parameterized, no code edits to retarget, runs on both runtimes:
- `00_setup.py` — stage a dataset to a UC Volume as parquet + persist label names
- `01_finetune_vision.py` — ViT image classification
- `02_finetune_llm_lora.py` — LoRA SFT, merges and registers a portable fp32 model

Both accept env vars **and** `--KEY=value` args, because AI Runtime passes `env_variables` in YAML
while classic `spark_python_task` takes `parameters`.

---

## 13. Measured results

Same GPU on both sides — **NVIDIA A10G 23.7 GB**. Single cold runs, **not benchmarks** (n=1).

**ViT-B/16 fine-tune** — 1034 train / 133 val, 3 classes, 3 epochs
| | AI Runtime | Classic DBR 18 ML |
|---|---|---|
| Compute start | **~40 s** | **621 s** |
| Train loop | 48.4 s | 38.2 s |
| Throughput | 64.1 img/s | 81.3 img/s |
| Eval accuracy | 0.985 | 1.000 |
| **Total wall** | **183 s** | **852 s** |

**LoRA SFT** — Qwen2.5-0.5B, 2000 examples, 1 epoch, seq 512
| | AI Runtime | Classic (warm) |
|---|---|---|
| Train loop | 192.1 s | 187.1 s |
| Throughput | 10.41 samp/s | 10.69 samp/s |
| Train / eval loss | 2.1520 / 2.127 | 2.1519 / 2.126 |
| Trainable params | **1.75%** (8.8M / 502.8M) | identical |
| Peak GPU memory | **4.99 GB** | 6.26 GB |

**Read these three things off the table:**
1. **Serverless is 4.7× faster end to end** on short work — almost entirely cluster provisioning.
2. **GPU-bound training performs the same on both.** ViT showed classic +26.9%, but LoRA only
   +2.7% — so the ViT gap is a host-side data-pipeline effect, not platform compute.
3. **Training is equivalent**: loss 2.1519 vs 2.1520 from the same script across different
   `transformers` major versions.

**Headroom:** a 0.5B model at 512 sequence length peaked under **5 GB on a 24 GB A10**. You do not
need H100s to start.

---

## 14. Getting the model out

Relevant whenever inference has to run somewhere other than Databricks — on-prem, at the edge, or
in your own account.

**The honest framing:** this path is **thinly documented.** The model-lifecycle page presents
`mlflow.artifacts.download_artifacts` as an *advanced/debug* use case and says "in most cases … you
should use MLflow APIs." The Model Serving docs contain no export or on-prem guidance.

**What works:**
```python
mlflow.artifacts.download_artifacts(
    artifact_uri="models:/catalog.schema.model/1", dst_path="/Volumes/.../export")
```
- Use `/Volumes/` destinations; `/dbfs/` paths fail.
- For LoRA, log the **merged fp32** model (`merge_and_unload()`), not the adapter — it then loads
  anywhere with plain `transformers`, no PEFT dependency.
- Set an explicit `ModelSignature`, or downstream loaders have no contract.

**Known risks — to verify before relying on them:**
- An unresolved report of **ONNX artifact download failing** with `OSError [Errno 95]` to
  `/Volumes/` (non-ONNX flavors worked). ONNX is the obvious edge format — **test it for your case.**
- **UC access control and lineage do not follow the exported artifact.** Governance stops at the
  boundary.
- MLflow artifact reads are slow for large checkpoint sets.

---

# APPENDIX — back pocket

## 90. Serverless infrastructure boundary: split the question by clause

When a security or legal team says "no serverless," the useful question is *which requirement*,
because the three answers differ:

| The requirement | Verdict | Why |
|---|---|---|
| **Compute must run in our own cloud account** | **Founded. Not mitigable.** | Serverless runs in the Databricks-managed plane; no customer-hosted variant exists, and per-tenant VPCs for serverless are not architecturally available. If this is the clause, **classic GPU in your account is the answer** |
| **Processing must stay in a named region / off a specific cloud** | **Partly founded** | The workspace setting *"Enforce data processing within workspace Geography"* is a real lever, but it enforces **Geo, not region**, and cross-region placement within a Geo is not pinnable |
| **Our data must not leave our storage** | **Not founded** | Training data, checkpoints and artifacts stay in your UC Volumes. Docs state data is not stored outside the workspace Geo |

Note that this applies to **all serverless compute, not just GPU** — the same plane runs CPU
notebooks and jobs. Mitigations: Geo enforcement, NCC / PrivateLink / egress controls, keeping all
data and checkpoints in UC Volumes.

**Residual risks a reviewer can legitimately still raise:** managed compute-plane ownership;
unpinnable cross-region placement; Geo-level-only residency; transient GPU-host data with no
published wipe SLA; the isolation whitepaper not being public. FedRAMP High and DoD IL5 are
documented as unsupported.

## 91. Capability comparison

| Dimension | AI Runtime | Classic DBR ML |
|---|---|---|
| Accelerators | A10, H100, B300 | T4, L4, A10G, L40S, V100, A100, H100 |
| Multi-GPU | 8xH100 / 8xB300 only | Any multi-GPU node |
| Multi-node | CLI + `torchrun`, or Ray | TorchDistributor, DeepSpeed, Ray-on-Spark |
| Spark | Spark Connect only | Full cluster, Photon on CPU |
| Libraries | Standard/AI env v6, `%uv pip`, custom Docker | Full DBR ML preinstalled |
| Data path | UC Volumes, `UCVolumeDataset`, Spark Connect | Delta, Volumes, Mosaic Streaming |
| Checkpoints | PyTorch DCP → UC Volumes | Volumes |
| MLflow | Auto system metrics + Logs tab | Autologging; no auto system metrics |
| Scheduling | `ai_runtime_task`, DABs, CLI | Standard jobs |
| Compliance | Not FedRAMP High / IL5 | Standard profiles |

## 92. When classic is genuinely the right answer

- **Cloud account separation is required** — the decisive case
- Spark and GPU needed **in the same cluster**
- Ray-on-Spark (not available on serverless)
- A library AI Runtime doesn't carry
- **GPU types AI Runtime lacks** — T4, L4, L40S, A100, V100
- A compliance profile AI Runtime doesn't cover
- Very long training runs where per-step throughput outweighs a 10-minute start (crossover measured
  at roughly **46 minutes** of training-loop time — n=1, directional)

## 93. Gotchas found by running it
See `FINDINGS.md` — 21 items, including the four that cost us the most time:
1. `environment.version: 6` silently gives you an image with **no torch**; use `'databricks_ai_v6'`
2. AI Runtime **auto-retries deterministic failures** — set `max_retries: 0` while developing
3. `mlflow.pytorch.log_model` + AMP/Accelerate = `PicklingError`; unwrap and cast to fp32
4. Classic `.py` tasks have **no implicit MLflow experiment**
