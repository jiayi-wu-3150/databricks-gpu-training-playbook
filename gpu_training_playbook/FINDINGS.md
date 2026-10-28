# Training on Databricks — measured findings

All numbers from runs executed 2026-10-25 in `fevm-classic-stable-qxxm2v` (AWS us-east-1).
**Same GPU on both sides (NVIDIA A10G, 23.7 GB)**, so differences are platform, not hardware.

> **These are single cold runs, not benchmarks.** n=1, one dataset, one model size. Treat the
> throughput numbers as directional. Prior batch-inference work on this account showed cold-vs-warm
> differences of ~2× and rankings that inverted with scale.

## Headline: ViT-B/16 fine-tune, beans dataset (1034 train / 133 val, 3 classes, 3 epochs)

| | AI Runtime (serverless) | Classic DBR 18 ML GPU |
|---|---|---|
| Accelerator | `GPU_1xA10` → A10G 23.7 GB | `g5.xlarge` → A10G 23.7 GB |
| **Compute start** | **~40 s** | **621 s** (10.4 min) |
| Training loop | 48.4 s | **38.2 s** |
| Throughput | 64.06 img/s | **81.27 img/s** |
| Eval accuracy | 0.985 | 1.000 |
| **Total wall** | **183 s** | **852 s** |
| UC model | `dl_training.vit_beans` v1 | `dl_training.vit_beans_classic` v1 |

### Two results that cut in opposite directions
1. **Serverless is 4.7× faster end to end** (183 s vs 852 s) — almost entirely because classic spent
   **10.4 minutes provisioning** a cluster versus 40 seconds.
2. **Classic was 26.9% faster inside the training loop** (81.27 vs 64.06 img/s) on identical
   hardware. This was not expected and the cause is **not established** — plausible contributors are
   the torch version gap (2.9.0+cu129 vs 2.11.0+cu130), `transformers` 4.x vs 5.x overhead, or
   differing vCPU count feeding `dataloader_num_workers=4`. **Do not assert a cause without testing.**

### The crossover
Classic's per-step advantage has to pay back a 581 s startup penalty:

- Break-even ≈ **175,000 images processed**, i.e. ~**46 minutes** of serverless training-loop time.
- **Under ~45 min of training → serverless wins on wall clock.** Over it, classic's throughput
  starts to pay back. Caveat: derived from n=1 on both sides.

This is a more useful framing than "serverless is faster" — it is faster *for iterative, short, and
bursty work*, which is exactly the research-team pattern.

### On the accuracy difference
0.985 vs 1.000 on a **133-image** validation set is **2 images**. Per-epoch on classic was
0.977 → 0.993 → 1.000; on serverless 0.977 → 0.993 → 0.985. That is run-to-run noise from dataloader
ordering and seeding, **not a platform quality difference.** Say so if asked.

## LoRA SFT — Qwen2.5-0.5B on dolly-15k (2000 examples, 1 epoch, seq 512)

| Metric | AI Runtime (serverless) | Classic DBR 18 ML (warm) |
|---|---|---|
| Training loop | 192.1 s | **187.1 s** |
| Throughput | 10.41 samples/s | **10.69 samples/s** (+2.7%) |
| Train loss | 2.1520 | 2.1519 |
| Eval loss | 2.127 | 2.126 |
| Trainable params | 1.75% (8,798,208 / 502,830,976) | identical |
| **Peak GPU memory** | **4.99 GB** | 6.26 GB (**AIR 20% lower**) |
| UC model | `llm_lora_qwen2_5_0_5b` v1 | `llm_lora_qwen2_5_0_5b_classic` v1 |

Classic run used the **warm** all-purpose cluster, so its number excludes compute start. Use the
621 s figure measured on job compute for any end-to-end comparison.

### This result explains the ViT anomaly — and kills the "classic compute is faster" reading

| Workload | Classic advantage |
|---|---|
| ViT fine-tune (image preprocessing per batch) | **+26.9%** |
| LoRA SFT (tokenized text, GPU-bound) | **+2.7%** |

The large ViT gap did **not** reproduce on a GPU-compute-bound workload. ViT spends real CPU per
batch on JPEG decode and image preprocessing through `dataloader_num_workers=4`; LoRA feeds
pre-tokenized text and is bound by GPU matmul. So the 27% is best explained as a **host-side
data-pipeline difference** (vCPU count / dataloader throughput), **not** a GPU or platform compute
difference. Still worth confirming with pinned versions, but the hypothesis now has supporting
evidence and the honest headline is **"GPU-bound training performs the same on both."**

### Training equivalence is confirmed
Train loss 2.1519 vs 2.1520 and eval loss 2.126 vs 2.127 — effectively identical from the same
script, same hyperparameters, different runtimes and different `transformers` major versions. This is
the strongest evidence for the "it is the same script, just retargeted" claim.

**Note on memory:** AIR peaked 20% lower (4.99 vs 6.26 GB) for identical work. Likely allocator or
version differences (torch 2.11/transformers 5.8 vs torch 2.9/transformers 4.57). Minor, but it means
memory headroom estimates are not transferable between runtimes.

### Dependency question answered
`peft==0.19.1` + `trl==0.24.0` installed cleanly on DBR 18 ML's `transformers` 4.57.1 — **no forced
`transformers` upgrade**, no dependency conflict. So pinning the same PEFT/TRL versions across both
runtimes is viable even though the base `transformers` differs by a major version.

**Headroom point for the deck:** a 0.5B model at 512 sequence length used under 5 GB on a 24 GB A10.
An A10 has room for substantially larger models before H100s are required — useful against the
assumption that LLM work demands top-end GPUs.

## VLM LoRA — Qwen2-VL-2B-Instruct on beans-as-instructions (240 train / 40 eval, 1 epoch)

Third workload, added to test whether **multimodal** data loading behaves differently. It does, and
the performance result reverses direction.

| Metric | AI Runtime | Classic DBR 18 ML (job compute) |
|---|---|---|
| `transformers` | **5.8.1** | **4.57.1** |
| Compute start | ~40 s | **621 s** (identical to the ViT job) |
| Training loop | **38.3 s** | 47.6 s |
| Throughput | **6.259 samp/s** | 5.041 samp/s |
| Train loss | 2.5580 | 2.5483 |
| Eval loss | 1.7946 | 1.7865 |
| Trainable params | 0.197% (4,358,144 / 2,213,343,744) | identical |
| Frozen vision params | 665,271,296 | identical |
| Peak GPU memory | 6.34 GB | **6.34 GB** (identical) |
| Sample generation | `bean leaf spot` | `bean leaf spot` (identical) |
| Total wall | **152 s** | **869 s** |

### ⚠️ The per-step comparison has NO consistent winner

| Workload | Nature | Faster per-step |
|---|---|---|
| ViT fine-tune | Host-heavy JPEG decode + preprocessing | **Classic +26.9%** |
| LoRA SFT (text) | GPU-matmul-bound, pre-tokenized | Classic +2.7% (noise) |
| **VLM LoRA** | Image + text through one processor | **AI Runtime +24.2%** |

**This retires the "classic compute is faster per step" reading entirely.** The direction flips by
workload, and the most likely driver is the **library version gap, not the platform**: AI Runtime
runs `transformers` 5.8.1 / torch 2.11, classic runs `transformers` 4.57.1 / torch 2.9. Qwen2-VL
support improved substantially across that major version, so AIR wins where the newer stack helps
and loses where host-side data loading dominates.

**The honest statement for the deck:** per-step throughput depends on your workload and your library
versions, not on serverless-vs-classic. Pin versions and measure your own workload. End-to-end,
serverless wins on anything short because of the 621 s provisioning.

### Reproducibility across a major version boundary
Train loss 2.5580 vs 2.5483, eval loss 1.7946 vs 1.7865, **identical** peak memory (6.34 GB), and
the **same generated sample text** on both runtimes — across different `transformers` major
versions. Combined with the LoRA text result, this is strong evidence the templates are genuinely
portable.

### Multimodal data loading really is different
Confirmed by building it. A VLM sample needs:
1. Chat-template messages carrying an **image placeholder**
2. Text and images through **one processor call** together
3. A **custom collator** — the stock text collator cannot do this
4. **Masking of image placeholder tokens** (and pad tokens) out of the loss
5. **Sequence budgeting for image tokens**, which dominate the text; controlled via `max_pixels`
6. `remove_unused_columns=False`, or `Trainer` strips the raw image column before the collator sees it

Freezing the vision tower and LoRA-adapting only the language projections took trainable parameters
to **0.197%** — an order of magnitude below the text-only LoRA's 1.75%.

**Portability note:** the VLM model class moved across `transformers` major versions. The template
tries `Qwen2VLForConditionalGeneration` and falls back to `AutoModelForVision2Seq`. On both runtimes
the first import succeeded, but the fallback is why the same file ran unmodified on 4.57.1 and 5.8.1.

## Distributed (DDP) — and an accidental finding that beats the benchmark

Workload: ViT-B/16 on **Food-101** (20,000 real photos, 101 classes, 1 epoch), per-GPU batch 64.
Chosen deliberately over beans/CIFAR: real JPEGs keep the host decode path realistic, and 20k images
is enough that scale-out can actually help.

### AI Runtime, 8×H100 — SUCCESS
| Metric | Value |
|---|---|
| World size | 8 (NCCL, torchrun) |
| GPU | NVIDIA H100 80GB HBM3 |
| Training loop | **3.9 s** |
| Throughput | **5,061 img/s** (632.7 per GPU) |
| Global batch | 512 |
| Loss | 4.77 → **1.91** (random for 101 classes ≈ 4.615, so this is real learning) |
| Peak GPU memory | 6.58 GB |
| Total wall | 92 s |

### Classic, 8 GPUs — COULD NOT OBTAIN THE HARDWARE
| Attempt | Node | Outcome |
|---|---|---|
| 1 | `p5.48xlarge` (8×H100) | **`AWS_INSUFFICIENT_INSTANCE_CAPACITY`** after 501 s |
| 2 | `p4d.24xlarge` (8×A100) | **`AWS_INSUFFICIENT_INSTANCE_CAPACITY`** after 862 s |

**Twice confirmed, two different GPU generations, same region (us-east-1), same day.**

### What this actually demonstrates
AI Runtime served **8×H100 in ~40 seconds**. Classic could not acquire 8 GPUs at all across ~23
minutes of attempts on two node types. This is the "procuring GPUs is painful" pitch — except
measured, in one workspace, on one afternoon, rather than asserted.

**State it with the honest flip side:** AI Runtime has **no capacity SLA either**. We got H100s
because capacity happened to be there; on a miss the session sits in "starting" and then fails, with
no durable queue. The real difference is not that AIR guarantees capacity — it is that AIR pools
capacity across many tenants and hands it back in seconds, whereas a classic cluster asks EC2 for
8 contiguous GPUs in one AZ and takes ~10-14 minutes to find out it cannot have them.

**Consequence for the comparison:** there is **no classic distributed number**. Any per-GPU
throughput claim between runtimes at 8-GPU scale is unsupported. Do not invent one.

### Distributed launcher mechanics — three failures worth knowing
1. **`@distributed` does NOT work in an `air run` CLI job.** It calls
   `set_databricks_credentials_from_dbutils()` and dies with
   `cluster_id is required in the configuration`. It needs **notebook** context. In the CLI, launch
   with `torchrun` instead; AIR injects `NUM_NODES`/`NODE_RANK`/`MASTER_ADDR`/`MASTER_PORT`.
2. **`TorchDistributor` function mode fails from a `spark_python_task`** —
   `RuntimeError: TorchDistributor failed during training`. It pickles the function and re-executes
   the module in each subprocess, re-triggering the launcher. Per the official docs its entire
   documented workflow is **notebook**-based ("Launch distributed training in a notebook cell"); the
   `spark_python_task` path is undocumented.
3. **`__file__` is not defined under a `spark_python_task`** → `NameError`. File-mode
   `TorchDistributor` needs the script path, so resolve it from `sys.argv[0]` or pass it explicitly.

**The simplification that followed:** `TorchDistributor` file mode only shells out to `torchrun`
anyway, so for **single-node multi-GPU on classic, skip TorchDistributor entirely** and call
`python -m torch.distributed.run --standalone --nproc_per_node=N`. Identical launcher to the AIR CLI
path, which also makes the two runtimes genuinely comparable. TorchDistributor earns its keep only
for multi-node across Spark workers (`local_mode=False`).

### CLI vs notebook: there is no notebook to look at
With `databricks air run`, code is tarred and uploaded as a snapshot; there is **no notebook in the
workspace**. From the run logs:
`/Workspace/Users/<you>/.air/repo_snapshots/.internal/<name>_<hash>.tar.gz` → extracted to
`/databricks/code_source/<dir>` on the node, ephemerally. Visible afterwards: the tarball, the Jobs
run, and the MLflow run with logs. Not visible: anything editable. **Your local file or git is the
source of truth.** The trade is version control and multi-node vs. an interactive surface — and
`@distributed` only exists on the notebook side.

## ⚠️ Package versions diverge sharply between the two runtimes

| Package | AI Runtime (AI env v6) | Classic DBR 18 ML GPU |
|---|---|---|
| Python | 3.12.3 | 3.12.3 |
| torch | **2.11.0+cu130** | **2.9.0+cu129** |
| CUDA | **13.0** | **12.9** |
| **transformers** | **5.8.1** | **4.57.1** |
| datasets | (AI env) | 4.3.0 |
| accelerate | (AI env) | 1.11.0 |
| mlflow | 3.13.0 | 3.8.1 |
| peft / trl | 0.19.1 / 0.24.0 | not preinstalled |

**This is the most actionable finding for the "which Python packages" agenda item.**
`transformers` differs by a **major version** (5.x vs 4.x). Code written against one may break on the
other — this is a real migration cost that the "1-line change to move from TorchDistributor to
`@distributed`" story does not mention. Anyone standardizing across both runtimes must pin versions
explicitly rather than rely on preinstalled defaults.

Secondary: classic DBR 18 ML GPU needed **zero** added libraries for the ViT workload — torch,
torchvision, transformers, datasets, accelerate and mlflow were all preinstalled. AIR needed `peft`
and `trl` declared for the LoRA job. Classic's "batteries included" claim is genuinely true; the
tradeoff is the 10-minute start.

## Gotchas found by running it (not in the docs)

### AI Runtime CLI / YAML
1. `code_source` requires the **nested** form — `type: snapshot` + `snapshot.root_path`. The docs'
   flat `root_path: .` is rejected: *"field root_path not found in type aircmd.codeSourceConfig"*.
2. **`$CODE_SOURCE_PATH` already points at the uploaded directory.** Re-appending the dir name gives
   `/databricks/code_source/playbook/playbook/...`. `root_path` resolves relative to the YAML file.
3. `experiment_name` must be **bare** (alphanumeric, `-`, `_`). The absolute `/Users/...` path the
   MLflow docs show is rejected by the YAML validator.
4. **`version: 'databricks_ai_v6'` selects the AI image** (torch preinstalled). Plain `version: 6`
   gives the **Standard** image with **no torch at all** — undocumented, and a silent trap: the job
   starts fine and fails on `import torch`. Confirmed by the log line
   `databricks-ai environment activated (VIRTUAL_ENV=/opt/databricks-environments/databricks-ai)`.
5. **AIR auto-retries deterministic failures** even with no `max_retries` specified — observed
   `attempt_number: 2` on a plain path typo. Set `max_retries: 0` while developing or you burn
   minutes re-running broken code.
6. `databricks air run -h config.<field>` is a **built-in schema explorer** and is more current than
   the docs page. Use it as the source of truth for YAML fields.
7. `databricks air get <id>` takes the id positionally — not `get run <id>`.
8. **HF Hub egress works** from AIR, unauthenticated with a rate-limit warning. Set `HF_TOKEN` via
   the YAML `secrets:` block for real work.
9. `databricks air pools list` → `FEATURE_DISABLED` in this workspace. That is the *provisioned
   pools* API only; on-demand AIR works fine. Don't read it as "AIR is disabled."

### MLflow model logging
10. `mlflow.pytorch.log_model` **fails on an Accelerate/AMP-prepared model**:
    `PicklingError: Cannot pickle a prepared model with automatic mixed precision`. And bf16 weights
    break signature inference: `Input type (float) and bias type (c10::BFloat16) should be the same`.
    **Fix:** `trainer.accelerator.unwrap_model(model)` → `.to("cpu").to(torch.float32).eval()` →
    `mlflow.transformers.log_model(...)`. Works identically on both runtimes.
11. For LoRA, log the **merged** fp32 model (`merge_and_unload()`), not the adapter, when the artifact
    needs to be loadable off-platform without PEFT.

### Data
12. Legacy un-namespaced HF dataset ids now fail: `beans` →
    `HfUriError: Repository id must be 'namespace/name'`. Use `AI-Lab-Makerere/beans`.
13. **Parquet staging loses the `ClassLabel` feature** — integer labels survive, names don't. Persist
    label names separately (`labels.json` beside the parquet) or the model gets meaningless
    `LABEL_0/1/2` outputs.
14. Under `/Volumes/{cat}/{schema}/`, every first-level directory must be a **registered UC volume**
    or `os.makedirs` fails with `OSError [Errno 95] Operation not supported`.
15. Classic warns that the HF datasets cache at `/tmp/.hf.data.cache` is not persistent — set
    `HF_DATASETS_CACHE` under `/Volumes` to fix.

### Classic compute
16a. **A classic `spark_python_task` has NO implicit MLflow experiment.** `mlflow.start_run()` fails
    with `RESOURCE_DOES_NOT_EXIST: Could not find experiment with ID None`. AIR sets one from the
    YAML `experiment_name`; classic *notebooks* inherit one; classic `.py` tasks get nothing. Fix:
    call `mlflow.set_experiment("/Users/<you>/<name>")` in the script, or set
    `MLFLOW_EXPERIMENT_NAME`. The templates now take `--EXPERIMENT=`.
16b. **`spark_python_task` on an `existing_cluster_id` cannot take env vars** — cluster
    `spark_env_vars` can't be changed without a restart. The templates therefore accept
    `--KEY=value` argv overrides, which works identically on both runtimes.
16. `data_security_mode: SINGLE_USER` + `single_user_name` is required for UC volume access **and**
    UC model registration from DBR ML. Both worked with it set.
17. `spark_python_task` works fine on classic. The notebook/wheel-task-only restriction applies to
    **serverless** GPU.
18. Running an env-probe task **on the same job cluster** as training avoids paying a second
    10-minute cluster start. Multi-task jobs sharing one job cluster are the right pattern.
19. **UI trap:** the Compute page's default button is *"Create with Personal Compute"*, and that
    policy permits exactly one GPU node type — `g4dn.xlarge` (T4 16 GB). A10G is unreachable that
    way. Use "Create compute" and set policy to Unrestricted.

## Artifacts created

| Thing | Location |
|---|---|
| UC schema | `classic_stable_qxxm2v_catalog.dl_training` |
| Volumes | `.../data`, `.../checkpoints` |
| Dataset | `/Volumes/.../data/beans/{train,validation,test}.parquet` + `labels.json` |
| Templates | `/Users/jiayi.wu/gpu_training_playbook/playbook/{00_setup,01_finetune_vision,02_finetune_llm_lora}.py` |
| AIR configs | `.../playbook/{setup,vision_air,lora_air}.yaml` |
| Classic job | id `525908896925673` (`vit-beans-classic-dbr18-g5xlarge`) |
| All-purpose cluster | `gpu-dev-a10g` / `1025-184517-gqc40ys0` — g5.xlarge, 30-min auto-terminate |
| UC models | `vit_beans` v1, `vit_beans_classic` v1, `llm_lora_qwen2_5_0_5b` v1 |
| MLflow experiments | `playbook-vision`, `playbook-llm-lora`, `playbook-setup`, `air-gate-check` |

## Still to do
- Model export validation + the known ONNX `[Errno 95]` artifact-download repro
- Confirm the ViT data-pipeline hypothesis: pin identical torch/transformers, vary
  `dataloader_num_workers`, and compare host vCPU counts
- Multi-node 8xH100 via CLI + `torchrun`
- Deck, HTML, customer-facing findings doc
