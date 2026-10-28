# GPU training demo — notebooks, one per runtime

Built from the **current official doc templates** (not reverse-engineered), so each notebook
demonstrates the launcher and data APIs its own runtime actually documents.

| Notebook | Compute | Launcher / API demonstrated |
|---|---|---|
| `00_stage_sharded_data.py` | any UC-enabled compute | re-shards `train.parquet` into 32 shards so `UCVolumeDataset`'s per-`(rank, worker)` partitioning is exercised |
| `01_serverless_gpu_single.py` | serverless GPU `GPU_1xA10` | single-GPU notebook training + `UCVolumeDataset` + `databricks.air.data.DataLoader` |
| `02_serverless_gpu_distributed.py` | serverless GPU `GPU_8xH100` | **`@distributed()` + `.distributed(...)`**, `UCVolumeWriter` DCP checkpoint, returns collected from all ranks |
| `03_classic_gpu_torchdistributor.py` | all-purpose `g5.12xlarge` (4×A10G) | **`TorchDistributor` function mode** (the documented notebook workflow) |
| `04_classic_gpu_deepspeed.py` | all-purpose `g5.12xlarge` (4×A10G) | **`DeepspeedTorchDistributor`** with ZeRO stage 2 |

## Why notebooks

`@distributed` resolves credentials through `dbutils` and therefore **requires notebook context**.
Submitted as a CLI (`air run`) job it fails with `cluster_id is required in the configuration`.
TorchDistributor is the mirror image: its *function* mode is documented only for notebooks, and it
breaks from a `spark_python_task` because the module is re-executed in every subprocess. Notebooks
are the one surface where both runtimes run their own canonical pattern.

## Hard constraints from the docs

- `@distributed` distributed training **requires `GPU_8xH100` or `GPU_8xB300`**. `1xA10`/`1xH100`
  are single-GPU only, so there is no cheap multi-GPU serverless demo.
- `@distributed` is **single-node**; multi-node needs the AIR CLI with `torchrun`.
- `databricks-sdk-air` is preinstalled in **GPU environment v6+**; v5 and below use `serverless_gpu`.
- For CLI workloads using `databricks.air.data`, add `databricks-sdk-air[data]` to dependencies.
- `DeepspeedTorchDistributor` needs **DBR 14.0 ML+** and uses **camelCase** args
  (`numGpus`, `nnodes`, `localMode`, `useGpu`, `deepspeedConfig`) unlike `TorchDistributor`.
- `NCCL_P2P_DISABLE=1` is set on the classic cluster — the docs flag a peer-access error on the
  **AWS G5 suite**, which is exactly `g5.12xlarge`.

## Runtime-specific, by necessity

The training *loop* is the same shape on both sides, but these do not transfer:

| | Serverless (AI Runtime) | Classic |
|---|---|---|
| Data from volumes | `UCVolumeDataset` auto-shards per `(rank, worker)` | hand-partition the file list (`all_shards[rank::world]`) |
| Checkpoints | `UCVolumeWriter` + `torch.distributed.checkpoint` | `torch.save` / `save_pretrained` to `/Volumes` |
| Launcher | `@distributed` (notebook) / `torchrun` (CLI) | `TorchDistributor` or `DeepspeedTorchDistributor` |

Workspace: `fevm-classic-stable-qxxm2v`. Data: `classic_stable_qxxm2v_catalog.dl_training`.

## Running these as jobs (verified 2026-10-25, `fevm-classic-stable-qxxm2v`)

The classic notebooks just need `existing_cluster_id`. The **serverless GPU** notebooks were the
fiddly ones; the previously-recorded recipe was stale in three ways. What actually works:

```json
{
  "tasks": [{
    "task_key": "single",
    "notebook_task": {"notebook_path": ".../01_serverless_gpu_single"},
    "compute": {"hardware_accelerator": "GPU_1xA10"},
    "environment_key": "air"
  }],
  "environments": [{
    "environment_key": "air",
    "spec": {
      "environment_version": "4",
      "dependencies": ["transformers", "databricks-sdk-air[data]"]
    }
  }]
}
```

Corrections, each one an error the API actually returned:

1. **There is no `compute_type` field.** The task-level block is only
   `{"hardware_accelerator": ...}` — `compute_type: SERVERLESS` is silently dropped with
   `Warning: unknown field: compute_type`. Confirmed against the CLI's own bundle schema:
   `jobs.Compute` has exactly one property, `hardware_accelerator`, enum
   `GPU_1xA10 | GPU_8xH100 | GPU_1xH100 | GPU_8xB300`.
2. **An environment is mandatory** once task-level `compute` is set:
   *"An environment is required for serverless task … when `compute` is set on task level."*
3. **`base_environment: databricks_ai_v6` is rejected in this workspace**:
   *"Only custom base environments (Workspace or Volume absolute paths ending with '.yaml' or
   '.yml') are currently supported."* The docs show the named form, so this is gated on a workspace
   preview. Fall back to `environment_version`.
4. **`environment_version: "4"` is the minimal env, not the AI env.** `torch` and `torchvision`
   import fine, but `transformers` does not — `ModuleNotFoundError: No module named 'transformers'`.
   Declare it, plus `databricks-sdk-air[data]` for `databricks.air.data`.

**For the live demo, prefer interactive attachment** — open the notebook, Environment side panel →
**AI v6** (PyTorch/Transformers preloaded) and attach the GPU accelerator. That avoids all of the
above, and it is also the only way `@distributed` gets the notebook context it requires.

Also note: `jobs.AiRuntimeTask` exists in the schema (requires `deployments` + `experiment`, takes
`code_source_path`) but it is built for **code-archive** workloads, not notebooks — so it is not the
path for these.
