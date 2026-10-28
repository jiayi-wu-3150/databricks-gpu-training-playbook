# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Synthesize a small OTR-shaped MDS dataset (smoke data)
# MAGIC
# MAGIC The real OTR dataset is **~80 GB** and `download_otr_v0.1.py` + `convert_otr_to_mds` take
# MAGIC hours. The fine-tuning notebooks only need MDS shards with the right **schema** to validate
# MAGIC end to end (model load → DDP → streaming → MLflow → checkpoint → inference).
# MAGIC
# MAGIC This builds that, in minutes, from the Food-101 images already staged in UC — drawing
# MAGIC random text overlays so the task is genuinely "remove the overlaid text":
# MAGIC
# MAGIC | Column | Type | Here |
# MAGIC |---|---|---|
# MAGIC | `id` | str | `smoke_000123` |
# MAGIC | `aug_idx` | int | always `0` (no augmentation) |
# MAGIC | `input` | jpeg | Food-101 image **with** text drawn on it |
# MAGIC | `target` | jpeg | the clean original |
# MAGIC | `mask` | png | binary mask of the text boxes |
# MAGIC
# MAGIC Splits match what the training notebook expects: `train`, `OTR_easy`, `OTR_hard`.
# MAGIC
# MAGIC > **This validates the pipeline, not model quality.** 10 smoke steps on synthetic overlays
# MAGIC > tells you the code runs; it tells you nothing about inpainting accuracy on real OTR.

# COMMAND ----------

# MAGIC %pip install -q mosaicml-streaming~=0.13.0 'pillow~=12.1'

# COMMAND ----------

CATALOG = "classic_stable_qxxm2v_catalog"
SCHEMA = "dl_training"
SRC_SHARDS = f"/Volumes/{CATALOG}/{SCHEMA}/data/food101_shards"
MDS_OUT = f"/Volumes/{CATALOG}/{SCHEMA}/data/OTR_like_MDS"
RESOLUTION = 512
SPLIT_SIZES = {"train": 256, "OTR_easy": 32, "OTR_hard": 32}

# COMMAND ----------

import io, os, glob, json, random, shutil
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
from streaming import MDSWriter

COLUMNS = {"id": "str", "aug_idx": "int", "input": "jpeg", "target": "jpeg", "mask": "png"}
WORDS = ["SAMPLE", "DRAFT", "CONFIDENTIAL", "50% OFF", "SALE", "NEW", "LIMITED", "2026",
         "WATERMARK", "PREVIEW", "COPY", "BEST PRICE"]


def _font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def make_sample(img, rng, hard: bool):
    """Return (input_with_text, clean_target, binary_mask)."""
    target = img.convert("RGB").resize((RESOLUTION, RESOLUTION), Image.BICUBIC)
    inp = target.copy()
    mask = Image.new("L", (RESOLUTION, RESOLUTION), 0)
    d_in, d_mask = ImageDraw.Draw(inp), ImageDraw.Draw(mask)

    for _ in range(rng.randint(2, 4) if hard else rng.randint(1, 2)):
        text = rng.choice(WORDS)
        size = rng.randint(28, 64) if hard else rng.randint(36, 72)
        font = _font(size)
        x0, y0 = rng.randint(0, RESOLUTION - 200), rng.randint(0, RESOLUTION - 80)
        box = d_in.textbbox((x0, y0), text, font=font)
        fill = tuple(rng.randint(200, 255) for _ in range(3)) if rng.random() < 0.5 \
            else tuple(rng.randint(0, 60) for _ in range(3))
        d_in.text((x0, y0), text, font=font, fill=fill)
        # Mask covers the text box, dilated slightly — inpainting needs margin.
        d_mask.rectangle([box[0] - 4, box[1] - 4, box[2] + 4, box[3] + 4], fill=255)
    return inp, target, mask

# COMMAND ----------

shards = sorted(glob.glob(f"{SRC_SHARDS}/*.parquet"))
print(f"source shards: {len(shards)}")
assert shards, f"no parquet shards at {SRC_SHARDS}"


def iter_images(limit):
    seen = 0
    for sp in shards:
        for rec in pq.read_table(sp).to_pylist():
            v = rec["image"]
            if isinstance(v, dict):
                v = v.get("bytes")
            if not isinstance(v, (bytes, bytearray)):
                continue
            yield Image.open(io.BytesIO(v))
            seen += 1
            if seen >= limit:
                return

# COMMAND ----------

total = sum(SPLIT_SIZES.values())
src = iter_images(total)
rng = random.Random(17)

for split, n in SPLIT_SIZES.items():
    out = f"{MDS_OUT}/{split}"
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)
    hard = split == "OTR_hard"
    with MDSWriter(out=out, columns=COLUMNS, compression="zstd", size_limit=1 << 26) as w:
        for i in range(n):
            img = next(src)
            inp, tgt, msk = make_sample(img, rng, hard)
            w.write({"id": f"smoke_{split}_{i:05d}", "aug_idx": 0,
                     "input": inp, "target": tgt, "mask": msk})
    files = os.listdir(out)
    print(f"{split:10s} {n:4d} samples -> {len(files)} files ({', '.join(sorted(files)[:3])}...)")

print(f"\nMDS root: {MDS_OUT}")
print("Set the training notebook's 'MDS Folder' widget to: OTR_like_MDS")

# COMMAND ----------

# Read back through StreamingDataset exactly as training will, to prove the shards are valid.
from streaming import StreamingDataset

for split in SPLIT_SIZES:
    ds = StreamingDataset(local=f"{MDS_OUT}/{split}", shuffle=False)
    s = ds[0]
    print(f"{split:10s} len={len(ds):4d} keys={sorted(s.keys())} "
          f"input={s['input'].size} target={s['target'].size} mask={s['mask'].size} "
          f"mask_extrema={s['mask'].getextrema()}")
