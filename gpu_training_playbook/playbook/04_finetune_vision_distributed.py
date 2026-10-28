"""Distributed (data-parallel) ViT fine-tune on Databricks GPU compute.

THE POINT OF THIS TEMPLATE: the training function is identical on both runtimes.
Only the LAUNCHER differs.

  AIR notebook : from databricks.air import distributed
                 @distributed(num_accelerators=8, accelerator_type="GPU_8xH100")
                 (NOTEBOOK ONLY. Fails in `air run` CLI jobs: it needs dbutils
                  and errors with "cluster_id is required".)
  AIR CLI      : torchrun --standalone --nproc_per_node=8 this_script.py
  Classic    : from pyspark.ml.torch.distributor import TorchDistributor
               TorchDistributor(num_processes=4, local_mode=True, use_gpu=True).run(fn)

Both end up running standard PyTorch DDP: NCCL process group, DistributedSampler,
DistributedDataParallel. Pick the launcher with LAUNCHER=air|classic.

Logs to MLflow from rank 0 only.
"""
import os
import sys

for _arg in sys.argv[1:]:
    if _arg.startswith("--") and "=" in _arg:
        _k, _v = _arg[2:].split("=", 1)
        os.environ[_k.upper().replace("-", "_")] = _v

import json
import time

CATALOG = os.environ.get("CATALOG", "classic_stable_qxxm2v_catalog")
SCHEMA = os.environ.get("SCHEMA", "dl_training")
DATASET = os.environ.get("DATASET", "food101")
BASE_MODEL = os.environ.get("BASE_MODEL", "google/vit-base-patch16-224")
EPOCHS = int(os.environ.get("EPOCHS", "1"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))       # per-GPU
LR = float(os.environ.get("LR", "1e-4"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
LAUNCHER = os.environ.get("LAUNCHER", "air").lower()
NPROC = int(os.environ.get("NPROC", "8"))
ACCEL = os.environ.get("ACCEL", "GPU_8xH100")
RUNTIME_TAG = os.environ.get("RUNTIME_TAG", "unknown")
EXPERIMENT = os.environ.get("EXPERIMENT", "")

DATA_VOL = f"/Volumes/{CATALOG}/{SCHEMA}/data/{DATASET}"

# TorchDistributor file mode needs this script's own path. `__file__` is NOT
# defined under a Databricks spark_python_task, so fall back to argv[0], then to
# an explicit --SCRIPT_PATH override.
SELF_PATH = (os.environ.get("SCRIPT_PATH")
             or globals().get("__file__")
             or (sys.argv[0] if sys.argv and sys.argv[0].endswith(".py") else ""))
if SELF_PATH:
    SELF_PATH = os.path.abspath(SELF_PATH)


def train_fn():
    """Runs once per process. Identical under both launchers."""
    import torch
    import torch.distributed as dist
    import numpy as np
    import mlflow
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from datasets import load_dataset
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    is_main = rank == 0

    if is_main:
        print(f"world_size={world} gpu={torch.cuda.get_device_name(local_rank)} "
              f"torch={torch.__version__}")

    with open(f"{DATA_VOL}/labels.json") as fh:
        label_names = json.load(fh)
    n_classes = len(label_names)

    ds = load_dataset("parquet", data_files={
        "train": f"{DATA_VOL}/train.parquet"})["train"]
    processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
    mean = torch.tensor(processor.image_mean).view(3, 1, 1)
    std = torch.tensor(processor.image_std).view(3, 1, 1)
    size = processor.size.get("height", 224)

    import torchvision.transforms as T
    tf = T.Compose([T.Resize((size, size)), T.ToTensor()])

    def collate(batch):
        xs = torch.stack([(tf(b["image"].convert("RGB")) - mean) / std for b in batch])
        ys = torch.tensor([int(b["labels"]) for b in batch])
        return xs, ys

    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                        collate_fn=collate, num_workers=NUM_WORKERS,
                        pin_memory=True, drop_last=True,
                        persistent_workers=NUM_WORKERS > 0)

    model = AutoModelForImageClassification.from_pretrained(
        BASE_MODEL, num_labels=n_classes, ignore_mismatched_sizes=True).to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    lossf = torch.nn.CrossEntropyLoss()
    scaler_dtype = torch.bfloat16

    if is_main and EXPERIMENT:
        mlflow.set_experiment(EXPERIMENT)
    run_ctx = (mlflow.start_run(run_name=f"ddp-vit-{RUNTIME_TAG}-ws{world}")
               if is_main else _Null())

    with run_ctx:
        if is_main:
            mlflow.set_tags({"runtime": RUNTIME_TAG, "workload": "vision-ddp",
                             "launcher": LAUNCHER})
            mlflow.log_params({"world_size": world, "per_gpu_batch": BATCH_SIZE,
                               "global_batch": BATCH_SIZE * world, "epochs": EPOCHS,
                               "lr": LR, "num_workers": NUM_WORKERS,
                               "base_model": BASE_MODEL, "dataset": DATASET,
                               "n_classes": n_classes, "n_train": len(ds),
                               "gpu": torch.cuda.get_device_name(local_rank),
                               "torch": torch.__version__})
        t0 = time.time()
        seen = 0
        for ep in range(EPOCHS):
            sampler.set_epoch(ep)
            model.train()
            for step, (x, y) in enumerate(loader):
                x = x.to(local_rank, non_blocking=True)
                y = y.to(local_rank, non_blocking=True)
                with torch.autocast("cuda", dtype=scaler_dtype):
                    out = model(pixel_values=x).logits
                    loss = lossf(out, y)
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                seen += x.size(0)
                if is_main and step % 20 == 0:
                    print(f"ep{ep} step{step} loss={loss.item():.4f}", flush=True)

        dist.barrier()
        wall = time.time() - t0
        total_seen = seen * world
        if is_main:
            metrics = {"train_seconds": wall,
                       "images_per_second": total_seen / wall,
                       "images_per_second_per_gpu": total_seen / wall / world,
                       "final_loss": float(loss.item()),
                       "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
            mlflow.log_metrics(metrics)
            summary = {"runtime": RUNTIME_TAG, "launcher": LAUNCHER,
                       "world_size": world,
                       "gpu": torch.cuda.get_device_name(local_rank),
                       "train_seconds": round(wall, 1),
                       "images_per_second": round(total_seen / wall, 2),
                       "per_gpu_img_per_s": round(total_seen / wall / world, 2),
                       "global_batch": BATCH_SIZE * world,
                       "final_loss": round(float(loss.item()), 4),
                       "peak_gpu_mem_gb": round(metrics["peak_gpu_mem_gb"], 2)}
            print("SUMMARY " + json.dumps(summary), flush=True)
    dist.destroy_process_group()


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# ----------------------------------------------------------------- launchers
if LAUNCHER == "air":
    from databricks.air import distributed

    wrapped = distributed(num_accelerators=NPROC, accelerator_type=ACCEL)(train_fn)
    wrapped.distributed()

elif LAUNCHER == "classic":
    from pyspark.ml.torch.distributor import TorchDistributor

    # FILE mode, not function mode. Passing train_fn directly fails from a
    # spark_python_task: TorchDistributor pickles the function and re-executes
    # this module in every subprocess, which re-triggers this launcher and dies
    # with "TorchDistributor failed during training". File mode runs torchrun on
    # the file instead. --LAUNCHER=torchrun makes the child take the torchrun
    # branch rather than recursing (argv overrides the inherited env var).
    if not SELF_PATH or not os.path.exists(SELF_PATH):
        raise SystemExit(f"cannot locate this script for TorchDistributor file mode "
                         f"(SELF_PATH={SELF_PATH!r}); pass --SCRIPT_PATH=/path/to/script.py")
    print(f"TorchDistributor file mode on {SELF_PATH} with {NPROC} processes")
    TorchDistributor(num_processes=NPROC, local_mode=True, use_gpu=True).run(
        SELF_PATH, "--LAUNCHER=torchrun")

elif LAUNCHER == "spawn":
    # Classic single-node multi-GPU WITHOUT TorchDistributor.
    # Per the official docs, TorchDistributor's documented workflow is
    # NOTEBOOK-based ("Launch distributed training in a notebook cell"); there is
    # no documented spark_python_task path, and function mode fails there. Its
    # file mode merely shells out to `torchrun`, so for single-node we invoke
    # torch.distributed.run directly. Same launcher as the AIR CLI path, which
    # also makes the two runtimes genuinely comparable.
    import subprocess

    if not SELF_PATH or not os.path.exists(SELF_PATH):
        raise SystemExit(f"cannot locate this script (SELF_PATH={SELF_PATH!r}); "
                         f"pass --SCRIPT_PATH=/path/to/script.py")
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
           f"--nproc_per_node={NPROC}", SELF_PATH, "--LAUNCHER=torchrun"]
    print("launching:", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.run(cmd, check=False).returncode)

elif LAUNCHER in ("torchrun", "single"):
    # AIR CLI path. @distributed does NOT work here: it calls
    # set_databricks_credentials_from_dbutils() and fails with
    # "cluster_id is required" because it expects notebook context.
    # Launch with torchrun instead; it sets RANK/WORLD_SIZE/LOCAL_RANK, and AIR
    # additionally provides NUM_NODES/NODE_RANK/MASTER_ADDR/MASTER_PORT for
    # multi-node.
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    train_fn()

else:
    raise SystemExit(f"unknown LAUNCHER={LAUNCHER}")
