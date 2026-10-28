# Databricks notebook source
# MAGIC %md
# MAGIC # 02a — Fine-tune SD2 inpainting on **SERVERLESS GPU (AI Runtime)**
# MAGIC
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | Compute | serverless `GPU_8xH100`, **`environment_version: "6"`** |
# MAGIC | Launcher | **`@distributed`** — notebook-only; an `air run` CLI job would need `torchrun` |
# MAGIC | Data | the **same** sharded Parquet, read with **`UCVolumeDataset` + `air DataLoader`** |
# MAGIC | Sharding | **file**-level: `n_files >= R x W` or ranks starve and DDP **hangs** |
# MAGIC | Base model | **Unity Catalog** `@base` — no Hugging Face egress |
# MAGIC
# MAGIC Dependencies are declared in the job's `environments` block, **not** `%pip`: on Serverless
# MAGIC GPU Standard, training after a `%pip install` hits a CUDA fork-safety limitation.
# MAGIC
# MAGIC The classic twin is `02b_finetune_classic.py`. The *training step* is identical; only the
# MAGIC launcher and the loader differ.

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text("catalog", "classic_stable_qxxm2v_catalog", "Catalog")
dbutils.widgets.text("schema", "sd2_playbook", "Schema")
dbutils.widgets.text("base_model", "sd2_inpaint_base", "UC base model")
dbutils.widgets.text("base_alias", "base", "Base alias")
dbutils.widgets.dropdown("accelerator", "GPU_8xH100", ["GPU_1xA10", "GPU_1xH100", "GPU_8xH100"], "Accelerator")
dbutils.widgets.text("num_accelerators", "8", "num_accelerators (TOTAL GPUs)")
dbutils.widgets.text("num_training_steps", "10", "Optimizer steps")
dbutils.widgets.text("per_gpu_batch_size", "4", "Batch / GPU")
dbutils.widgets.text("grad_accum", "2", "Grad accumulation")
dbutils.widgets.text("learning_rate", "1e-5", "Learning rate")
dbutils.widgets.text("dataloader_workers", "2", "DataLoader workers / rank")

C = dbutils.widgets.get("catalog"); S = dbutils.widgets.get("schema")
UC_BASE = f"{C}.{S}.{dbutils.widgets.get('base_model')}"
BASE_ALIAS = dbutils.widgets.get("base_alias")
ACCEL = dbutils.widgets.get("accelerator")
NGPU = int(dbutils.widgets.get("num_accelerators"))
STEPS = int(dbutils.widgets.get("num_training_steps"))
BATCH = int(dbutils.widgets.get("per_gpu_batch_size"))
ACCUM = int(dbutils.widgets.get("grad_accum"))
LR = float(dbutils.widgets.get("learning_rate"))
WORKERS = int(dbutils.widgets.get("dataloader_workers"))

PQ = f"/Volumes/{C}/{S}/data/parquet"
CKPT_ROOT = f"/Volumes/{C}/{S}/checkpoints"
DEBUG = f"{CKPT_ROOT}/_debug"
PROMPT = "remove text from image, clean background"

import os, glob, json, time
n_train_files = len(glob.glob(f"{PQ}/train/*.parquet"))
streams = NGPU * WORKERS
print(f"train shards: {n_train_files} | streams (R x W) = {NGPU} x {WORKERS} = {streams}")
assert n_train_files >= streams, (
    f"UCVolumeDataset shards the FILE LIST across {streams} (rank, worker) streams but only "
    f"{n_train_files} shards exist -> {streams - n_train_files} streams would get NOTHING and "
    f"DDP would HANG at the next all-reduce. Re-run 00_prepare_data with train_shards >= {streams}, "
    f"or lower dataloader_workers.")

CONFIG = {"runtime": "serverless-air-env-v6", "accelerator": ACCEL,
          "launcher": "databricks.air.distributed", "data_loader": "UCVolumeDataset+air.DataLoader",
          "diffusers": "0.39.0", "num_accelerators": NGPU, "steps": STEPS,
          "per_gpu_batch_size": BATCH, "grad_accum": ACCUM, "learning_rate": LR,
          "dataloader_workers": WORKERS, "global_batch": BATCH * ACCUM * NGPU,
          "train_shards": n_train_files, "streams": streams,
          "uc_base_model": f"{UC_BASE}@{BASE_ALIAS}"}
print(json.dumps(CONFIG, indent=2))

# COMMAND ----------

# DBTITLE 1,Download the base pipeline from UC ONCE (remote=False -> workers share this node)
import mlflow
mlflow.set_registry_uri("databricks-uc")
t0 = time.time()
_root = mlflow.artifacts.download_artifacts(f"models:/{UC_BASE}@{BASE_ALIAS}")
BASE_DIR = next((r for r, _, fs in os.walk(_root) if "model_index.json" in fs), None)
assert BASE_DIR, f"no model_index.json under {_root}"
print(f"base pipeline from UC -> {BASE_DIR} ({round(time.time()-t0,1)}s)")

os.makedirs(DEBUG, exist_ok=True)
TS = time.strftime("%Y%m%d_%H%M%S")
CKPT = f"{CKPT_ROOT}/serverless_{TS}"

# COMMAND ----------

def train_fn(base_dir, pq_root, ckpt_dir, steps, batch, accum, lr, workers, prompt, cfg):
    """Runs once per rank under @distributed. All imports inside: the function is cloudpickled."""
    import os, io, json, time, traceback
    rank = int(os.environ.get("RANK", 0))
    try:
        import torch, torch.nn.functional as F
        import numpy as np
        import pyarrow.parquet as pq_
        from PIL import Image
        from torch.utils.data import IterableDataset
        from accelerate import Accelerator
        from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler
        from databricks.air.data import UCVolumeDataset
        from databricks.air.data import DataLoader as AirDataLoader

        acc = Accelerator(gradient_accumulation_steps=accum, mixed_precision="fp16")
        dev, is_main = acc.device, acc.is_main_process
        RES = 512

        def decode_row(r):
            g = lambda k, m: Image.open(io.BytesIO(r[k])).convert(m).resize(
                (RES, RES), Image.LANCZOS if m == "RGB" else Image.NEAREST)
            t = lambda im, c: torch.from_numpy(np.array(im)).float().reshape(RES, RES, c).permute(2, 0, 1)
            return {"input": t(g("input_jpg", "RGB"), 3) / 127.5 - 1.0,
                    "target": t(g("target_jpg", "RGB"), 3) / 127.5 - 1.0,
                    "mask": (t(g("mask_png", "L"), 1) / 255.0 > 0.5).float()}

        class ParquetShardRows(IterableDataset):
            """Docs-prescribed wrapper: UCVolumeDataset yields CACHED LOCAL PATHS (one Parquet
            shard each), auto-partitioned per (rank, worker). We decode rows out of them."""

            def __init__(self, path):
                self._paths = UCVolumeDataset(path)

            def __iter__(self):
                for local_path in self._paths:
                    for row in pq_.read_table(local_path).to_pylist():
                        yield decode_row(row)

        train_dl = AirDataLoader(ParquetShardRows(f"{pq_root}/train"), batch_size=batch,
                                 num_workers=workers, pin_memory=True, drop_last=True)
        # Validation needs random access -> plain file read, "approach 2" in the AIR docs.
        import glob as _g
        val_rows = []
        for f in sorted(_g.glob(f"{pq_root}/val/*.parquet")):
            val_rows.extend(pq_.read_table(f).to_pylist())

        pipe = StableDiffusionInpaintPipeline.from_pretrained(base_dir, torch_dtype=torch.float32)
        unet, vae, txt, tok = pipe.unet, pipe.vae, pipe.text_encoder, pipe.tokenizer
        sched = DDPMScheduler.from_pretrained(base_dir, subfolder="scheduler")
        vae.requires_grad_(False); txt.requires_grad_(False)
        unet.enable_gradient_checkpointing()
        opt = torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=1e-2)
        unet, opt, train_dl = acc.prepare(unet, opt, train_dl)
        vae.to(dev); txt.to(dev)
        ids = tok([prompt], padding="max_length", max_length=tok.model_max_length,
                  truncation=True, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            emb = txt(ids)[0]

        def build_input(b):
            """SD2-inpainting UNet takes 9 channels: 4 noisy latent + 1 mask + 4 masked-image."""
            tgt, msk = b["target"].to(dev), b["mask"].to(dev)
            with torch.no_grad():
                lat = vae.encode(tgt).latent_dist.sample() * vae.config.scaling_factor
                mlat = vae.encode(tgt * (msk < 0.5).float()).latent_dist.sample() * vae.config.scaling_factor
            m_small = F.interpolate(msk, size=lat.shape[-2:], mode="nearest")
            noise = torch.randn_like(lat)
            ts = torch.randint(0, sched.config.num_train_timesteps, (lat.shape[0],), device=dev).long()
            return torch.cat([sched.add_noise(lat, noise, ts), m_small, mlat], dim=1), ts, noise

        unet.train()
        step, last, epochs, t_start = 0, float("nan"), 0, time.time()
        # Each stream holds only 1/(R x W) of the shards, so the iterable exhausts well before
        # `steps` optimizer steps. Re-iterate across epochs.
        while step < steps and epochs < 50:
            epochs += 1
            for b in train_dl:
                with acc.accumulate(unet):
                    mi, ts, noise = build_input(b)
                    pred = unet(mi, ts, encoder_hidden_states=emb.repeat(mi.shape[0], 1, 1)).sample
                    loss = F.mse_loss(pred.float(), noise.float())
                    acc.backward(loss)
                    opt.step(); opt.zero_grad(set_to_none=True)
                if acc.sync_gradients:
                    step += 1; last = loss.item()
                    if is_main and (step % 5 == 0 or step == steps):
                        print(f"step {step}/{steps} loss {last:.4f} (epoch {epochs})", flush=True)
                    if step >= steps:
                        break
        wall = time.time() - t_start

        unet.eval(); vtot, vn = 0.0, 0
        with torch.no_grad():
            for i in range(0, min(len(val_rows), 8 * batch), batch):
                chunk = [decode_row(r) for r in val_rows[i:i + batch]]
                if not chunk:
                    break
                b = {k: torch.stack([c[k] for c in chunk]) for k in chunk[0]}
                mi, ts, noise = build_input(b)
                pred = unet(mi, ts, encoder_hidden_states=emb.repeat(mi.shape[0], 1, 1)).sample
                vtot += F.mse_loss(pred.float(), noise.float()).item(); vn += 1
        unet.train()

        cdir = f"{ckpt_dir}/best"
        out = {"rank": rank, "world_size": acc.num_processes, "steps": step, "epochs": epochs,
               "train_loss": round(last, 6), "val_loss": round(vtot / max(vn, 1), 6),
               "train_seconds": round(wall, 2), "steps_per_sec": round(step / wall, 4),
               "gpu": torch.cuda.get_device_name(0),
               "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
               "checkpoint": f"{cdir}/unet"}
        if is_main:
            os.makedirs(cdir, exist_ok=True)
            acc.unwrap_model(unet).save_pretrained(f"{cdir}/unet")
            with open(f"{cdir}/training_metadata.json", "w") as fh:
                json.dump({**cfg, **out}, fh, indent=2)
        acc.wait_for_everyone()
        return out
    except BaseException:
        # @distributed reports only "Training subprocess failed. See its STDERR above", and a Job
        # run does not expose worker stdout. Persist each rank's traceback to the volume.
        try:
            os.makedirs(cfg["_debug"], exist_ok=True)
            with open(f"{cfg['_debug']}/air_err_rank{rank}.txt", "w") as fh:
                fh.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n" + traceback.format_exc())
        except BaseException:
            pass
        raise

# COMMAND ----------

from databricks.air import distributed

cfg = {**CONFIG, "_debug": DEBUG}
mlflow.set_experiment(f"/Users/{spark.sql('select current_user()').first()[0]}/sd2_playbook_serverless")

with mlflow.start_run(run_name=f"serverless-{TS}") as run:
    mlflow.log_params(CONFIG)
    runner = distributed(num_accelerators=NGPU, accelerator_type=ACCEL)(train_fn)
    results = runner.distributed(BASE_DIR, PQ, CKPT, STEPS, BATCH, ACCUM, LR, WORKERS, PROMPT, cfg)

    rs = results if isinstance(results, list) else [results]
    assert len(rs) == NGPU, f"expected {NGPU} rank results, got {len(rs)}"
    ranks = sorted(r["rank"] for r in rs)
    assert ranks == list(range(NGPU)), f"missing ranks: {ranks}"
    r0 = next(r for r in rs if r["rank"] == 0)
    mlflow.log_metrics({k: v for k, v in r0.items() if isinstance(v, (int, float))})
    mlflow.set_tags({"runtime": "serverless-air", "launcher": "@distributed", "accelerator": ACCEL})

print(json.dumps(r0, indent=2))
with open(f"/Volumes/{C}/{S}/data/finetune_serverless_result.json", "w") as fh:
    json.dump({**cfg, **r0, "all_ranks": rs, "checkpoint_dir": CKPT,
               "mlflow_run_id": run.info.run_id}, fh, indent=2)
