# Training on Databricks — playbook workspace

**Folder to look at: `/Users/jiayi.wu/gpu_training_playbook/`**

A reusable playbook for training deep-learning and LLM workloads on Databricks GPU compute,
covering both **AI Runtime** (serverless GPU) and **classic Databricks Runtime ML**.

Validated in workspace `fevm-classic-stable-qxxm2v` (AWS us-east-1, org 7474656479102553);
runs executed 2026-10-25. Nothing here is account-specific — retarget via the template knobs.

| File | What it is |
|---|---|
| **`README.md`** | This index — status, results, every ID and URL |
| **`FINDINGS.md`** | The substance: measured results, analysis, 21 gotchas found by running it |
| `playbook/` | The reusable templates (the real artifact) |
| `gate/` | Phase 0 gate-check workload (throwaway, kept as the minimal AIR example) |

Engagement-specific context (the customer layer, account constraints, workshop scoping) is kept
**out of this folder on purpose** so the playbook stays reusable. It lives in the plan file at
`~/.claude/plans/i-have-an-investigation-cosmic-scroll.md`.

---

## ⚠️ Phase status — 4 of 8 complete

| Phase | What | Status |
|---|---|---|
| 0 | Gate: is serverless GPU usable here | ✅ **DONE** — yes |
| 1 | Research consolidation | 🟡 **PARTIAL** — research gathered (4 sweeps); formal evidence table not written; open items below |
| 2 | Workspace foundation | ✅ **DONE** |
| 3 | ViT fine-tune, both runtimes | ✅ **DONE** |
| 4 | LoRA SFT, both runtimes | ✅ **DONE** |
| 5 | Knob matrix + deprecation validation | ❌ **NOT STARTED** |
| 6 | Model export validation (incl. ONNX bug repro) | ❌ **NOT STARTED** |
| 7 | Deliverables: deck, HTML, customer findings doc | ❌ **NOT STARTED** (`FINDINGS.md` is internal working notes, not the deliverable) |

**Nothing customer-facing exists yet.** The four training runs and the engineering findings are
done; the deck, HTML, customer-facing writeup, and the per-customer use-case layer are not.

---

## Results — all four runs

Same GPU throughout: **NVIDIA A10G, 23.7 GB**. Single cold runs, **not benchmarks** (n=1).

### ViT-B/16 fine-tune — beans, 1034 train / 133 val, 3 classes, 3 epochs
| | AI Runtime | Classic DBR 18 ML |
|---|---|---|
| Compute start | **~40 s** | **621 s** |
| Train loop | 48.4 s | 38.2 s |
| Throughput | 64.06 img/s | 81.27 img/s |
| Eval accuracy | 0.985 | 1.000 |
| Total wall | **183 s** | **852 s** |

### LoRA SFT — Qwen2.5-0.5B, dolly-15k 2000 ex, 1 epoch, seq 512
| | AI Runtime | Classic DBR 18 ML (warm) |
|---|---|---|
| Train loop | 192.1 s | 187.1 s |
| Throughput | 10.41 samp/s | 10.69 samp/s |
| Train / eval loss | 2.1520 / 2.127 | 2.1519 / 2.126 |
| Trainable params | 1.75% (8.8M / 502.8M) | identical |
| Peak GPU memory | **4.99 GB** | 6.26 GB |

### The three conclusions worth carrying into any deck
1. **Serverless is 4.7× faster end to end** on short work (183 s vs 852 s) — almost entirely the
   10.4-minute cluster provision. Crossover ≈ **46 min** of training-loop time.
2. **GPU-bound training performs the same on both.** ViT showed classic +26.9%, but LoRA only
   +2.7% — so the ViT gap is a host-side data-pipeline effect (JPEG decode / dataloader), not
   platform compute. Earlier "classic compute is faster" reading does **not** hold.
3. **Training is equivalent**: loss 2.1519 vs 2.1520 from the same script across different
   `transformers` major versions. Best evidence for "same script, just retargeted."

---

## Run inventory

### AI Runtime (serverless) — `databricks air get <id>` / `databricks air logs <id>`
| Run ID | What | Result |
|---|---|---|
| `452490544541336` | Phase 0 gate check | ✅ 43 s — A10G, CUDA 13.0, no torch in Standard env |
| `158476151680539` | Dataset staging, attempt 1 | ❌ path double-count; auto-retried 3× |
| `511803762976341` | Dataset staging, attempt 2 | ❌ un-namespaced HF dataset id |
| `445934738341769` | Dataset staging, attempt 3 | ✅ 62 s |
| `418974866197030` | Dataset staging + `labels.json` | ✅ 62 s |
| `30340105639463` | accidental duplicate | cancelled |
| `36768608062273` | ViT, attempt 1 | ❌ MLflow pickling on AMP model |
| **`524144357207390`** | **ViT final** | ✅ **183 s** |
| **`403504790938480`** | **LoRA final** | ✅ **282 s** |

### Classic GPU — `databricks jobs get-run <id>`
| Run ID | What | Result |
|---|---|---|
| **`262067301714482`** | **ViT on ephemeral job cluster** (job `525908896925673`) | ✅ **852 s** |
| `319044087812224` | LoRA, attempt 1 | cancelled — no env vars |
| `1111932953359008` | LoRA, attempt 2 | ❌ no implicit MLflow experiment |
| **`791070746142052`** | **LoRA final** (task run `75260188212023`) | ✅ 371 s exec |

### Compute
| Resource | ID | State |
|---|---|---|
| All-purpose cluster `gpu-dev-a10g` | `1025-184517-gqc40ys0` | **RUNNING**, 30-min idle auto-terminate — terminate when done |
| Classic job `vit-beans-classic-dbr18-g5xlarge` | `525908896925673` | job definition left in place |

---

## Unity Catalog + MLflow

**Schema** `classic_stable_qxxm2v_catalog.dl_training`

Volumes:
- `/Volumes/classic_stable_qxxm2v_catalog/dl_training/data/beans/` — `train.parquet`,
  `validation.parquet`, `test.parquet`, `labels.json`
- `/Volumes/classic_stable_qxxm2v_catalog/dl_training/checkpoints/` — per-run checkpoint dirs

Registered models (all v1):
| Model | From |
|---|---|
| `...dl_training.vit_beans` | AIR ViT |
| `...dl_training.vit_beans_classic` | classic ViT |
| `...dl_training.llm_lora_qwen2_5_0_5b` | AIR LoRA (merged fp32) |
| `...dl_training.llm_lora_qwen2_5_0_5b_classic` | classic LoRA (merged fp32) |

MLflow experiments:
| ID | Name |
|---|---|
| `3849731302370519` | `/Users/jiayi.wu@databricks.com/playbook-vision` |
| `3849731302370540` | `/Users/jiayi.wu@databricks.com/playbook-llm-lora` |
| `3849731302370500` | `/Users/jiayi.wu@databricks.com/playbook-setup` |
| `3849731302370494` | `/Users/jiayi.wu@databricks.com/air-gate-check` |

Base URL: `https://fevm-classic-stable-qxxm2v.cloud.databricks.com`
— experiments at `/ml/experiments/<id>`, job runs at `/jobs/runs/<id>`,
cluster at `/compute/clusters/1025-184517-gqc40ys0`

---

## The templates — `playbook/`

Parameterized; retarget with **no code edits**. Both env vars and `--KEY=value` argv work, so the
same file runs under AIR (`env_variables:` in YAML) and classic (`spark_python_task.parameters`).

| File | Purpose |
|---|---|
| `00_setup.py` | Stage a HF dataset to a UC Volume as parquet + persist `labels.json` |
| `01_finetune_vision.py` | ViT image-classification fine-tune |
| `02_finetune_llm_lora.py` | LoRA SFT of a causal LM, merges and registers the portable fp32 model |
| `setup.yaml`, `vision_air.yaml`, `lora_air.yaml` | AIR CLI workload configs |

Key knobs: `CATALOG`, `SCHEMA`, `DATASET`/`DATASET_ID`, `BASE_MODEL`, `EPOCHS`, `BATCH_SIZE`,
`MAX_SEQ_LEN`, `LORA_R`/`LORA_ALPHA`, `BF16`, `MODEL_NAME`, `RUNTIME_TAG`, `EXPERIMENT`.

Run AIR: `databricks air run --file playbook/vision_air.yaml`
Classic: `databricks jobs submit --json @/tmp/lora_classic_job.json`

Workspace copies (used by classic): `/Workspace/Users/jiayi.wu@databricks.com/playbook_classic/`

---

## Open items

**Research, unresolved — any of these could be asked wrong in the room:**
- DBU SKU names and rates for AIR and classic GPU (pricing page didn't render; don't quote Slack)
- Max runtime / idle timeout: docs say 2 days notebook, 14 days CLI, 15-min connect; an internal
  source says 7 days and 60-min idle. **Contradictory**
- Multi-node H100 status — four internal sources disagree
- CSP / HIPAA / PCI coverage contradictory; FedRAMP High and DoD IL5 documented unsupported.
  **Do not assert FedRAMP Moderate**
- Glean + Confluence MCP tokens expired, so `go/AIR/Enablement` and `go/product-availability` are
  still second-hand

**Engineering, not yet done:**
- Export validation + the known ONNX `[Errno 95]` artifact-download repro (Phase 6)
- Confirm the ViT data-pipeline hypothesis with pinned versions and varied `dataloader_num_workers`
- Multi-node 8xH100 via CLI + `torchrun`
- Empirically confirm the deprecation claims (Petastorm / HorovodRunner / HyperOpt) — Phase 5

**Per-engagement (kept in the plan file, not here):** customer constraints, the serverless
infrastructure-boundary analysis, and the use-case layer. Deliberately excluded so this folder can
be handed to the next engagement unchanged.
