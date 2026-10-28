# Databricks notebook source
# MAGIC %md
# MAGIC # 02b — Fine-tune SD2 inpainting on **CLASSIC GPU**
# MAGIC
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | Compute | all-purpose `g5.12xlarge` (4 x A10G), DBR **18 LTS** ML GPU |
# MAGIC | Launcher | `TorchDistributor(local_mode=True)` — **function mode**, the documented notebook workflow |
# MAGIC | Data | sharded Parquet + `pyarrow`, **row**-level sharding via `DistributedSampler` |
# MAGIC | Base model | loaded from **Unity Catalog** `@base` — no Hugging Face egress |
# MAGIC | diffusers | `0.39.0` — capped by `transformers<5` -> `huggingface_hub<1.0` |
# MAGIC
# MAGIC The serverless twin is `02a_finetune_serverless.py`. The *training step* is identical; only
# MAGIC the launcher and the loader differ.

# COMMAND ----------

# MAGIC %pip install 'diffusers==0.39.0' accelerate bitsandbytes

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text("catalog", "classic_stable_qxxm2v_catalog", "Catalog")
dbutils.widgets.text("schema", "sd2_playbook", "Schema")
dbutils.widgets.text("base_model", "sd2_inpaint_base", "UC base model")
dbutils.widgets.text("base_alias", "base", "Base model alias")
dbutils.widgets.dropdown("num_gpus", "4", ["1", "2", "4"], "GPUs")
dbutils.widgets.text("num_training_steps", "10", "Optimizer steps")
dbutils.widgets.text("per_gpu_batch_size", "1", "Batch / GPU")
dbutils.widgets.text("grad_accum", "8", "Grad accumulation")
dbutils.widgets.text("learning_rate", "1e-5", "Learning rate")
dbutils.widgets.text("val_every", "10", "Validate every N steps")

C = dbutils.widgets.get("catalog"); S = dbutils.widgets.get("schema")
UC_BASE = f"{C}.{S}.{dbutils.widgets.get('base_model')}"
BASE_ALIAS = dbutils.widgets.get("base_alias")
NUM_GPUS = int(dbutils.widgets.get("num_gpus"))
STEPS = int(dbutils.widgets.get("num_training_steps"))
BATCH = int(dbutils.widgets.get("per_gpu_batch_size"))
ACCUM = int(dbutils.widgets.get("grad_accum"))
LR = float(dbutils.widgets.get("learning_rate"))
VAL_EVERY = int(dbutils.widgets.get("val_every"))

PQ = f"/Volumes/{C}/{S}/data/parquet"
CKPT_ROOT = f"/Volumes/{C}/{S}/checkpoints"
DEBUG = f"/Volumes/{C}/{S}/checkpoints/_debug"
PROMPT = "remove text from image, clean background"
CONFIG = {"runtime": "classic-dbr18lts-g5.12xlarge", "launcher": "TorchDistributor.function_mode",
          "data_loader": "pyarrow+DistributedSampler", "diffusers": "0.39.0",
          "num_gpus": NUM_GPUS, "steps": STEPS, "per_gpu_batch_size": BATCH,
          "grad_accum": ACCUM, "learning_rate": LR,
          "global_batch": BATCH * ACCUM * NUM_GPUS, "uc_base_model": f"{UC_BASE}@{BASE_ALIAS}"}
print(CONFIG)

# COMMAND ----------

# DBTITLE 1,Job-safe auth (workers are plain subprocesses with no notebook auth)
import os, mlflow
_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()


def _opt(getter):
    """browserHostName() raises in a JOB run; apiToken()/apiUrl() do not."""
    try:
        return getter().get()
    except Exception:
        return None


def _workspace_url():
    try:
        return spark.conf.get("spark.databricks.workspaceUrl")   # works in BOTH contexts
    except Exception:
        return None


_host = (_opt(_ctx.browserHostName) or os.environ.get("DATABRICKS_HOST", "").replace("https://", "")
         or _workspace_url())
DATABRICKS_HOST = f"https://{_host}" if _host else ""
DATABRICKS_TOKEN = _opt(_ctx.apiToken) or os.environ.get("DATABRICKS_TOKEN", "")
print("host:", DATABRICKS_HOST or "(none)", "| token:", "present" if DATABRICKS_TOKEN else "MISSING")
assert DATABRICKS_HOST and DATABRICKS_TOKEN, "workers need explicit host+token for MLflow/UC"

# COMMAND ----------

# DBTITLE 1,Download the base pipeline from UC ONCE on the driver
import time, json, glob, shutil
t0 = time.time()
mlflow.set_registry_uri("databricks-uc")
_root = mlflow.artifacts.download_artifacts(f"models:/{UC_BASE}@{BASE_ALIAS}")
BASE_DIR = None
for r, dirs, files in os.walk(_root):
    if "model_index.json" in files:
        BASE_DIR = r
        break
assert BASE_DIR, f"no model_index.json under {_root}"
print(f"base pipeline from UC -> {BASE_DIR} ({round(time.time()-t0,1)}s)")
# local_mode=True puts every rank on this node, so all ranks read this same local path.

# COMMAND ----------

os.makedirs(DEBUG, exist_ok=True)
TS = time.strftime("%Y%m%d_%H%M%S")


def train_fn(base_dir, pq_root, ckpt_dir, steps, batch, accum, lr, val_every,
             prompt, host, token, cfg):
    """Runs once per rank. All imports inside (TorchDistributor pickles this)."""
    import os, io, json, glob, time, traceback
    os.environ["DATABRICKS_HOST"] = host          # workers inherit no notebook auth
    os.environ["DATABRICKS_TOKEN"] = token
    os.environ["MLFLOW_TRACKING_URI"] = "databricks"

    import torch, torch.nn.functional as F
    import numpy as np
    import pyarrow.parquet as pq_
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    from accelerate import Accelerator
    import mlflow
    from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler

    rank = int(os.environ.get("RANK", 0))
    try:
        acc = Accelerator(gradient_accumulation_steps=accum, mixed_precision="fp16")
        dev = acc.device
        is_main = acc.is_main_process

        # ---- data: map-style over Parquet; Accelerate adds the DistributedSampler ----
        class ParquetTriplets(Dataset):
            """Row-level sharding. 256 samples / 21.7 MB fits in memory; for real OTR use a
            lazy per-shard reader instead."""

            def __init__(self, split, res=512):
                self.rows = []
                for f in sorted(glob.glob(f"{pq_root}/{split}/*.parquet")):
                    self.rows.extend(pq_.read_table(f).to_pylist())
                self.res = res

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, i):
                r = self.rows[i]
                g = lambda k, m: Image.open(io.BytesIO(r[k])).convert(m).resize(
                    (self.res, self.res), Image.LANCZOS if m == "RGB" else Image.NEAREST)
                t = lambda im, c: torch.from_numpy(np.array(im)).float().reshape(
                    self.res, self.res, c).permute(2, 0, 1)
                inp = t(g("input_jpg", "RGB"), 3) / 127.5 - 1.0
                tgt = t(g("target_jpg", "RGB"), 3) / 127.5 - 1.0
                msk = (t(g("mask_png", "L"), 1) / 255.0 > 0.5).float()
                return {"input": inp, "target": tgt, "mask": msk}

        train_ds = ParquetTriplets("train")
        val_ds = ParquetTriplets("val")
        train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=2,
                              pin_memory=True, drop_last=True)
        val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=2)

        # ---- model from the UC-staged pipeline (no HF egress) ----
        pipe = StableDiffusionInpaintPipeline.from_pretrained(base_dir, torch_dtype=torch.float32)
        unet, vae, txt, tok = pipe.unet, pipe.vae, pipe.text_encoder, pipe.tokenizer
        sched = DDPMScheduler.from_pretrained(base_dir, subfolder="scheduler")
        vae.requires_grad_(False); txt.requires_grad_(False)
        unet.enable_gradient_checkpointing()
        opt = torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=1e-2)

        unet, opt, train_dl, val_dl = acc.prepare(unet, opt, train_dl, val_dl)
        vae.to(dev); txt.to(dev)
        ids = tok([prompt], padding="max_length", max_length=tok.model_max_length,
                  truncation=True, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            emb = txt(ids)[0]

        def build_input(batch_d):
            """SD2-inpainting UNet takes 9 channels: 4 noisy latent + 1 mask + 4 masked-image."""
            tgt, msk = batch_d["target"].to(dev), batch_d["mask"].to(dev)
            with torch.no_grad():
                lat = vae.encode(tgt).latent_dist.sample() * vae.config.scaling_factor
                masked = tgt * (msk < 0.5).float()
                mlat = vae.encode(masked).latent_dist.sample() * vae.config.scaling_factor
            m_small = F.interpolate(msk, size=lat.shape[-2:], mode="nearest")
            noise = torch.randn_like(lat)
            ts = torch.randint(0, sched.config.num_train_timesteps, (lat.shape[0],), device=dev).long()
            noisy = sched.add_noise(lat, noise, ts)
            return torch.cat([noisy, m_small, mlat], dim=1), ts, noise

        def run_val():
            unet.eval(); tot, n = 0.0, 0
            with torch.no_grad():
                for j, b in enumerate(val_dl):
                    if j >= 8:
                        break
                    mi, ts, noise = build_input(b)
                    pred = unet(mi, ts, encoder_hidden_states=emb.repeat(mi.shape[0], 1, 1)).sample
                    tot += F.mse_loss(pred.float(), noise.float()).item(); n += 1
            unet.train()
            return tot / max(n, 1)

        unet.train()
        step, last, t_start = 0, float("nan"), time.time()
        while step < steps:
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
                        print(f"step {step}/{steps} loss {last:.4f}", flush=True)
                    if step >= steps:
                        break
        wall = time.time() - t_start
        vloss = run_val()

        out = {"rank": rank, "world_size": acc.num_processes, "steps": step,
               "train_loss": round(last, 6), "val_loss": round(vloss, 6),
               "train_seconds": round(wall, 2), "steps_per_sec": round(step / wall, 4),
               "gpu": torch.cuda.get_device_name(0),
               "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}

        # TorchDistributor returns ONE rank's result and it is not guaranteed to be rank 0
        # (observed: rank 3). So every rank reports the checkpoint path; only rank 0 writes it.
        cdir = f"{ckpt_dir}/best"
        out["checkpoint"] = f"{cdir}/unet"
        if is_main:
            os.makedirs(cdir, exist_ok=True)
            acc.unwrap_model(unet).save_pretrained(f"{cdir}/unet")
            with open(f"{cdir}/training_metadata.json", "w") as fh:
                json.dump({**cfg, **out}, fh, indent=2)
        acc.wait_for_everyone()
        return out
    except BaseException:
        # TorchDistributor surfaces only "failed during training"; worker stdout is not
        # exposed in a Job run. Persist each rank's traceback.
        try:
            os.makedirs(cfg["_debug"], exist_ok=True)
            with open(f"{cfg['_debug']}/classic_err_rank{rank}.txt", "w") as fh:
                fh.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n" + traceback.format_exc())
        except BaseException:
            pass
        raise

# COMMAND ----------

from pyspark.ml.torch.distributor import TorchDistributor

CKPT = f"{CKPT_ROOT}/classic_{TS}"
cfg = {**CONFIG, "_debug": DEBUG}
mlflow.set_experiment(f"/Users/{spark.sql('select current_user()').first()[0]}/sd2_playbook_classic")

with mlflow.start_run(run_name=f"classic-{TS}") as run:
    mlflow.log_params(CONFIG)          # the old notebooks logged ZERO params
    dist = TorchDistributor(num_processes=NUM_GPUS, local_mode=True, use_gpu=True)
    res = dist.run(train_fn, BASE_DIR, PQ, CKPT, STEPS, BATCH, ACCUM, LR, VAL_EVERY,
                   PROMPT, DATABRICKS_HOST, DATABRICKS_TOKEN, cfg)
    r = res[0] if isinstance(res, list) else res
    mlflow.log_metrics({k: v for k, v in r.items() if isinstance(v, (int, float))})
    mlflow.set_tags({"runtime": "classic", "launcher": "TorchDistributor"})

print(json.dumps(r, indent=2))
with open(f"/Volumes/{C}/{S}/data/finetune_classic_result.json", "w") as fh:
    json.dump({**cfg, **r, "checkpoint_dir": CKPT, "mlflow_run_id": run.info.run_id}, fh, indent=2)
