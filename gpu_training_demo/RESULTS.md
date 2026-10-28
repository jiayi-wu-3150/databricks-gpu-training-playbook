# Measured results — 2026-10-25, `fevm-classic-stable-qxxm2v`

All four notebooks ran green. ViT-B/16 on sharded Food-101, 1 epoch, 1 shard/rank, per-GPU batch 64
(DeepSpeed micro-batch 32).

| Notebook | Compute | World | Images | Wall (s) | img/s | Final loss | Peak GPU mem |
|---|---|---|---|---|---|---|---|
| `01` serverless single | AIR `GPU_1xA10` | 1 | 1,536 | 27.1 | 56.6 | 3.791 | 8.81 GB |
| `02` serverless `@distributed` | AIR `GPU_8xH100` | 8 | 6,144 | 208.0 | 29.5 | 3.799 | — |
| `03` classic `TorchDistributor` | `g5.12xlarge` 4×A10G | 4 | 3,072 | 12.47 | 246.3 | 3.981 | — |
| `04` classic DeepSpeed ZeRO-2 | `g5.12xlarge` 4×A10G | 4 | 3,072 | 11.40 | 269.4 | 3.418 | 4.41 GB |

## The three things the retry actually proved

1. **`@distributed` works in notebook context — the CLI failure was not a dead end.** Run as a
   `notebook_task` on serverless GPU it initialised 8 NCCL ranks and trained. The old finding
   ("fails in an `air run` CLI job with `cluster_id is required`") stands, but it is specifically a
   *CLI snapshot-job* limitation, not a property of `@distributed`.
2. **`TorchDistributor` function mode works in a notebook.** The old playbook had to fall back to
   file mode because function mode breaks under `spark_python_task` (module re-execution
   re-triggers the launcher). In a notebook the documented function-mode pattern just runs.
3. **`DeepspeedTorchDistributor` works** — the classic option the playbook never covered. It was
   also the *fastest* configuration here (269 img/s) at *half* the memory (4.41 GB vs 8.81 GB),
   which is exactly ZeRO-2's pitch: optimizer state sharded instead of replicated.

## ⚠️ Do NOT put the 8×H100 number on a slide as a platform comparison

29.5 img/s on 8×H100 versus 246 img/s on 4×A10G is **not** a hardware result. The AI Runtime
DataLoader's own instrumentation says why — on every one of the 8 ranks:

```
dataloader/between_next_seconds/rank_0..7  ≈ 206.5 – 206.8 s   (of 208.0 s total wall)
dataloader/fetch_seconds/rank_0..7         ≈ 0.49 – 0.67 s
```

**~99% of wall time was spent outside the loader**, i.e. in our own per-shard Python work: decoding
~768 JPEGs with PIL and building tensors *in the training loop's main process*. Fetching the shard
from the volume cost well under a second. The H100s were idle almost the entire run.

That is a genuinely useful demo beat — the AIR `DataLoader` instrumentation diagnosed a data-path
bottleneck without any extra profiling — but the throughput column cannot be read as
"H100 slower than A10G".

**Not established:** why the *same* per-rank work (768 images) took ~206 s on the 8×H100 node but
12.5 s on the classic node. Candidates, untested: 8 ranks contending for CPU while each decodes
single-threaded; a minimal `environment_version: 4` lacking the optimised PIL/torchvision builds
DBR ML ships; differing default thread counts. **Do not assert a cause without testing it.**

**The fix** (next step, not done here): move decode into the DataLoader workers rather than the
training loop — wrap the shard in a map-style dataset or do the PIL/tensor work in `collate_fn`, and
raise `num_workers` (the AIR DataLoader defaults to 6/rank when distributed, but we passed 2).

## Failures worth keeping

- **DeepSpeed `{"type": "AdamW"}` fails on DBR 18 ML GPU.** It instantiates `FusedAdam`, which
  JIT-compiles `fused_adam.so` during `deepspeed.initialize()` and dies:
  `ImportError: /root/.cache/torch_extensions/py312_cu129/fused_adam/fused_adam.so: cannot open
  shared object file` — 4 ranks racing into one extension cache. Fix: `"torch_adam": True`.
- **DeepSpeed needs no `%pip install` on DBR 18 ML GPU** — it is preinstalled; the import resolved
  to `/databricks/python/lib/python3.12/site-packages/deepspeed`.
- **`@distributed` creates its own MLflow experiment**, named after the *notebook* and placed in the
  user's home (`/Users/<you>/02_serverless_gpu_distributed`), **not** under the notebook's folder and
  not in any experiment you set yourself. Budget for this when hunting for the run.
- `UCVolumeWriter` + `dcp.save` wrote `.metadata` plus one `__<rank>_0.distcp` per rank (9 files for
  8 ranks) into the UC volume — confirmed present.
