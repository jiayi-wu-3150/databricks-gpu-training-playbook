# Diffusion inpainting — what I changed and what I measured

Source: `databricks-industry-solutions/cv-playground`, branch `mmt_diffusion_inpainting`,
`projects/diffusion_inpainting`. Updated and validated **2026-10-27** in
`fevm-classic-stable-qxxm2v`.

## Status: BOTH paths are GREEN

Run `274618671165412` — `TERMINATED: SUCCESS`, full notebook including post-training inference.

| Metric | Value |
|---|---|
| Compute | all-purpose `g5.12xlarge`, 4× A10G, DBR 18.x ML GPU |
| Launcher | `TorchDistributor(num_processes=4, local_mode=True, use_gpu=True)` — **function mode** |
| Steps | 10 (smoke) |
| `train/loss` | 0.14698 |
| `val/loss` | 0.004544 (= `val/best_loss`) |
| `train/steps_per_sec` | 0.2464 |
| `train/total_minutes` | 1.51 |
| Artifacts | `checkpoints/`, `run_info/rank_0_log.txt`, `samples/step_00010_sample_0.png` |
| MLflow | exp `1874631799642857`, run `4948459ffbec4d13b721f4286322d8cc` |

**The data is synthetic.** The real OTR dataset is ~80 GB; `00_make_smoke_otr_mds.py` builds an
OTR-*shaped* MDS set (`id`/`aug_idx`/`input`/`target`/`mask`) by drawing text overlays on the
Food-101 images already in UC. This validates the **pipeline**, not inpainting quality. Do not
quote `val/loss 0.0045` as a quality result — it is 10 steps on synthetic overlays.

### Serverless / AI Runtime path — GREEN

Run `753433848654637` — `TERMINATED: SUCCESS`, full notebook including inference.

| Metric | Serverless (AIR) | Classic |
|---|---|---|
| Accelerator | `GPU_8xH100`, `remote=false` (co-located) | `g5.12xlarge` 4× A10G |
| Launcher | `@distributed(num_accelerators=8, accelerator_type="GPU_8xH100")` | `TorchDistributor` function mode |
| World size | **8** (the notebook asserts `len(results)==8` and `ranks_seen==[0..7]`) | 4 |
| `train/steps_per_sec` | **1.1411** | 0.2464 |
| `train/total_minutes` | 1.165 | 1.513 |
| `train/loss` | 0.008791 | 0.14698 |
| `val/loss` | 0.001645 | 0.004544 |
| Samples written | 4 PNGs | 1 PNG |
| MLflow run | `21d202f02b3c411484eb3d30f04913b3` (+ child `amazing-moth-101`) | `4948459ffbec4d13b721f4286322d8cc` |

`steps_per_sec` is ~4.6x higher on 8xH100 than 4xA10G. **Not a benchmark** — 10 steps, synthetic
data, 2x the GPUs on far faster silicon. Treat as "it runs and scales", nothing more.

Note `@distributed` created a **nested child run** under the notebook's parent run, exactly as the
AIR docs describe.

## FINAL STATE: each runtime on its own documented data path

Reworked after confirming the AI Runtime data-loading guidance. The goal is **same model, same
data, format adjusted per runtime** — not an apples-to-apples comparison.

| | Classic | Serverless (AI Runtime) |
|---|---|---|
| Data format | MDS shards | one `.npz` per sample on a volume |
| Loader | `StreamingDataset` | **`UCVolumeDataset` + `databricks.air.data.DataLoader`** |
| diffusers | `0.39.0` (capped by MDS) | **`0.42.0`** |
| transformers | 4.57.x (DBR's) | **5.x** (AIR's own) |
| Launcher | `TorchDistributor` function mode | `@distributed`, 8 ranks |
| Validated run | `274618671165412` | `175620959119063` |
| `train/steps_per_sec` | 0.2464 | **1.5587** |
| `train/loss` | 0.14698 | 0.008793 |
| Samples written | 1 PNG | 4 PNGs |
| MLflow run | `4948459ffbec4d13b721f4286322d8cc` | `60582a78215348d1b0e2f79757e47587` |

Staged by `01_mds_to_volume_files.py` (the `prep_data.py` equivalent): reads the same MDS the
classic notebook trains on, re-emits 256/32/32 `.npz` files. **Dropping Mosaic Streaming from the
serverless side is what freed `transformers 5.x` + `diffusers 0.42`** — the constraint chain
disappears rather than being worked around.

For the record, the same 8xH100 notebook on MDS (`753433848654637`) got 1.1411 steps/sec against
1.5587 on `UCVolumeDataset`. **Not a benchmark** (10 steps, n=1, different val sampling) but
directionally consistent with the docs' claim that the AIR DataLoader prefetches and caches while
the GPU computes. Do not compare `val/loss` across the two runs — the validation sampling differs.

### Why `UCVolumeDataset` on serverless — three independent sources

1. **Docs** (`ai-runtime/develop-workloads/dataloading`): for unstructured volume data, prescribes
   `UCVolumeDataset` + `databricks.air.data.DataLoader` as the optimized path, and shows the
   decoding-wrapper pattern (it yields cached local *paths*, not tensors).
2. **CLI help** (`air config -h`): the workload YAML has **no data field at all** — only
   `experiment_name`/`environment`/`compute`/`command`, plus `code_source` which is explicitly
   *code*. Data is never shipped by the CLI; the script reads `/Volumes/...` at runtime and the
   path is passed via `parameters` (→ `HYPERPARAMETERS_PATH`) or `env_variables`.
3. **Field guidance** (Slack `#apa-ai-runtime`, 2026-10-10): "AI Runtime's documented pattern is to
   access training data, model assets, and checkpoints through Unity Catalog Volumes, using
   UCVolumeDataset/DataLoader and distributed checkpointing"; and "consolidate data into large
   files… monitor data-loader throughput… use asynchronous distributed checkpointing."

The **cookbook** is the lone dissenter — it uses HF `datasets` (`load_dataset`/`load_from_disk`) —
but every cookbook training example is a *text* workload, where that is the right tool. No
cookbook example trains on unstructured image files, so it does not actually contradict the docs.

### Known-wrong-at-scale caveat

All three sources say **consolidate into large files** (docs: ~1-10 GiB each). Our smoke set is 320
files of ~1 MB, the opposite. Fine for validating the path; for a real 74k-sample OTR run, batch
many samples per file. Related: writing many small archives to a volume is also how we hit
`OSError: [Errno 5]` below.

## ⚠️ The finding that matters most: Mosaic Streaming caps the whole model stack

`mosaicml-streaming 0.13.0` declares `transformers<5,>=4.21.3`. transformers 4.x caps
`huggingface_hub<1.0`. `diffusers>=0.40` needs `hub>=1.23`; `>=0.41` needs `hub>=1.32`. So:

```
MDS -> transformers<5 -> huggingface_hub<1.0 -> diffusers <= 0.39
```

**Mosaic Streaming and current diffusers cannot coexist.** Asking for both fails with
*"Library installation failed ... Unable to find or download the required package"* — and the
identical conflict separately kills the in-worker `pip install` inside `@distributed`, which is a
different installer and had to be fixed independently.

MDS is **not deprecated** (0.13.0 is the latest release) but it is *anchored to the pre-5.x
ecosystem*. On AI Runtime v6, whose own stack is transformers 5.x, pinning `transformers<5` to keep
MDS means deliberately opting out of the runtime's native stack.

### The fork to put in front of the customer

| Keep MDS | Move to `UCVolumeDataset` |
|---|---|
| ceiling of `diffusers 0.39`, transformers 4.x | unlocks `diffusers 0.42+`, transformers 5.x |
| data layout + augmentation already built | restructure to many-file layout, own decode/collate |
| identical stack on both runtimes (simpler) | the **AIR-documented** data path |

This is also two independent arguments landing on the same answer: the AIR docs prescribe
`UCVolumeDataset`, and the dependency graph *forces* it if you want a current diffusers.

## Serverless-specific defects fixed

7. **`DDP_ERROR_HANDLING = "fallback"` hid every real error.** Hardcoded. On a worker exception it
   printed the cause to stdout (which a Job run does not expose) and continued to the inference
   cell, so the job died on an unrelated `FileNotFoundError` and the true cause was lost. Default
   changed to `"fail"`, overridable via the `DDP_ERROR_HANDLING` env var.
8. **The in-worker `pip install` sat OUTSIDE the try block**, so a resolution failure there wrote no
   traceback at all — the absence of traceback files was the clue that located it. Moved inside,
   and `-q` dropped so pip's resolver output survives.
9. **`_WORKER_PACKAGES` carried the conflicting pins** (`diffusers==0.42.0` + `hub>=1.32` +
   `mosaicml-streaming`) even after the job environment was fixed. Two installers, same conflict,
   fixed separately.
10. Same unguarded `_cache_to_local(BASE_PIPELINE_PATH)` as the classic notebook (line ~1894).
11. **`np.savez` to a UC Volume fails with `OSError: [Errno 5] Input/output error`.** A `.npz` is a
    ZIP, which needs random-access/seek writes; volume FUSE wants sequential ones. Build it in
    `io.BytesIO` and commit with a single `write()`. (Companion to the `Errno 95` trap: the first
    path segment under `/Volumes/{cat}/{schema}/` must be a *registered volume* — `os.makedirs`
    cannot create one.)


## Defects found in the repo (4)

1. **Deprecated AIR API.** `from serverless_gpu import distributed` and
   `@distributed(gpus=..., gpu_type='A10'|'H100')`. Current API is `databricks.air`, and it takes a
   single `accelerator_type` enum encoding chip **and** per-node count
   (`GPU_1xA10`/`GPU_1xH100`/`GPU_8xH100`/`GPU_8xB300`), not a name plus a count. Support for
   `gpus`/`gpu_type` ends at GPU environment v7. Added `_air_accelerator_type()` to map between
   them, plus a `B300` preset. Also fixed 3 stale doc URLs (the page moved under `/connecting/`).
2. **Interactive-only auth crashes any job run.** `_nb_ctx.browserHostName().get()` raised
   `java.util.NoSuchElementException: None.get` through Py4J, killing the notebook before training.
   **Measured** (notebook `98_ctx_probe.py`): in a Job run `apiToken()`, `apiUrl()`,
   `notebookPath()` and `spark.conf spark.databricks.workspaceUrl` are all present —
   **only `browserHostName()` raises**, which makes sense as there is no browser. Fixed with a
   three-tier host fallback: `browserHostName()` → `DATABRICKS_HOST` → `spark.databricks.workspaceUrl`.
3. **Hardcoded, undocumented UC volume.** `DIFFUSION_VOL = /Volumes/{cat}/{schema}/diffusion` is
   **not** widget-driven (unlike `volume`), and the first path segment under
   `/Volumes/{cat}/{schema}/` must be a *registered volume* — `os.makedirs` cannot create one, so it
   surfaces as the opaque `OSError: [Errno 95] Operation not supported`. The Quick Start never
   mentions it. Fixed by creating the volume; the requirement is now documented here.
4. **Successful training followed by a crash.** The training path (line ~1262) checks for
   `model_index.json` and falls back to the HF hub, but the inference path (line ~1654) called
   `_cache_to_local(BASE_PIPELINE_PATH)` unguarded — and *nothing in the notebook ever writes that
   cache*. So a green training run ended in
   `FileNotFoundError: .../diffusion/_base_pipeline`. Mirrored the training guard.

## Two defects I introduced, then fixed

5. **`diffusers==0.42.0` was the wrong pin for classic.** I originally read
   `huggingface_hub~=0.36` as a bug. It is not. On DBR 18.x ML GPU, `transformers 4.57.6` caps
   `huggingface_hub<1.0`, so hub stays at **0.36.2**; diffusers ≥0.40 needs hub ≥1.23 and ≥0.41
   needs ≥1.32, failing with `ImportError: cannot import name 'resolve_revision'`. Measured ceiling:
   **`diffusers==0.39.0`** is the newest release accepting `hub>=0.34`. The original unpinned
   `diffusers` + low hub pin was self-consistent; my pin broke it.
   **Consequence: the two runtimes need different pins** — classic `diffusers==0.39.0`
   (transformers 4.x), AIR v6 can take `0.42.0` (transformers 5.x). One floating dependency cannot
   serve both.
6. **Blank-host regression.** Making the auth block job-safe, I assumed blank credentials were fine
   because "workers inherit the cluster identity". False for `TorchDistributor` `local_mode` workers
   — they are plain `torchrun` subprocesses with no notebook auth (**measured**: a bare subprocess
   calling MLflow with no creds fails). The token was fine; only the host was missing, leaving MLflow
   with URI `databricks` and nowhere to send it.

## Things I predicted would break and that did not

- **`StreamingDataset(local=<FUSE volume path>)` worked** with 4 ranks. I flagged the missing
  `remote=`/`local=` split as high-risk; it was not a problem at this scale. Mosaic Streaming
  0.13.0 is also current (latest on PyPI), so the `~=0.13.0` pin is fine — MDS is **not** the stale
  part. It is, however, the *classic* data path: AIR documents `UCVolumeDataset` instead.
- The `%pip install` + in-worker `pip install` pattern did not trigger the Serverless-GPU-Standard-v4
  CUDA fork-safety failure — but that was only tested on **classic**, which the internal thread said
  is immune. Still a live risk for the serverless notebook.

## Debugging notes worth keeping

- **A Job run exposes no readable notebook stdout.** `get-run-output` returns an empty `logs`
  field, and `jobs/runs/export` returns a client-rendered HTML shell with zero cell outputs
  (grepped 330 KB for `Traceback` — 0 hits). The repo's per-rank MLflow logs only exist if training
  gets far enough to gather them, so a setup-time worker crash leaves nothing behind.
  **Fix applied:** `_train_wrapper` now writes each rank's traceback to
  `{DIFFUSION_VOL}/_debug/err_rank{N}.txt` and re-raises.
- Those debug files **persist across runs**. Always check the timestamp written at the top; a stale
  file sent me chasing an already-fixed error.
- `%pip check` as a line magic **exits 1 on conflicts and aborts the notebook**. Run it as a
  captured subprocess if you want the conflict list rather than a dead run.
- `%pip install -q` hides the resolver's downgrade and conflict warnings — exactly the output that
  explains defect 5. Dropped `-q`.

## Files

| File | Purpose |
|---|---|
| `00_make_smoke_otr_mds.py` | synthesize OTR-shaped MDS (train/OTR_easy/OTR_hard) — avoids the 80 GB download |
| `finetune_sd2_inpainting_classic.py` | **validated** — 4×A10G, TorchDistributor function mode |
| `finetune_sd2_inpainting_serverless.py` | API-migrated, **not yet run** |
| `98_ctx_probe.py` | what auth material a Job run exposes (presence/length only, never values) |
| `99_env_probe.py` | dependency resolution report, `pip check` captured not fatal |
