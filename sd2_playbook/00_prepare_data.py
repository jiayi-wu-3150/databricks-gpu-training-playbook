# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Prepare data: raw triplets for inspection + sharded Parquet for training
# MAGIC
# MAGIC Synthesises an Overlay-Text-Removal task from real photographs, then emits **two** layouts:
# MAGIC
# MAGIC | Layout | Purpose |
# MAGIC |---|---|
# MAGIC | `raw/{split}/{input,target,mask,meta}/` | human-browsable PNGs + ground-truth word boxes — display in notebooks, inspect by eye |
# MAGIC | `parquet/{split}/part_*.parquet` | training-ready: one row per sample, three binary columns |
# MAGIC
# MAGIC **Three splits**, kept distinct: `train` / `val` (in-loop validation) / `test` (final metrics
# MAGIC only, never seen during training). Reusing val as test makes the final numbers validation
# MAGIC metrics wearing a test-set label.
# MAGIC
# MAGIC **Shard count matters.** `UCVolumeDataset` shards the *file list* across `R x W` streams
# MAGIC (ranks x DataLoader workers), so `n_files >= R x W` or ranks starve and DDP **hangs**.
# MAGIC Default 16 train shards suits 8 ranks x 2 workers (the pre-flight assert in 02a enforces this).
# MAGIC
# MAGIC Compute: **serverless CPU** — no GPU needed. Runs in a few minutes.

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text("catalog", "classic_stable_qxxm2v_catalog", "Catalog")
dbutils.widgets.text("schema", "sd2_playbook", "Schema")
dbutils.widgets.text("source_images", "/Volumes/classic_stable_qxxm2v_catalog/dl_training/data/food101_shards", "Source image parquet dir")
dbutils.widgets.text("n_train", "256", "Train samples")
dbutils.widgets.text("n_val", "64", "Val samples")
dbutils.widgets.text("n_test", "64", "Test samples")
dbutils.widgets.text("train_shards", "16", "Train parquet shards (>= R x W)")
dbutils.widgets.text("eval_shards", "4", "Val/test parquet shards")
dbutils.widgets.text("raw_samples_per_split", "8", "Raw PNGs to write per split (-1 = all)")
dbutils.widgets.text("resolution", "512", "Resolution")
dbutils.widgets.text("seed", "17", "Random seed")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
SRC = dbutils.widgets.get("source_images")
SPLIT_N = {"train": int(dbutils.widgets.get("n_train")),
           "val": int(dbutils.widgets.get("n_val")),
           "test": int(dbutils.widgets.get("n_test"))}
SPLIT_SHARDS = {"train": int(dbutils.widgets.get("train_shards")),
                "val": int(dbutils.widgets.get("eval_shards")),
                "test": int(dbutils.widgets.get("eval_shards"))}
RAW_PER_SPLIT = int(dbutils.widgets.get("raw_samples_per_split"))
RES = int(dbutils.widgets.get("resolution"))
SEED = int(dbutils.widgets.get("seed"))

BASE = f"/Volumes/{CATALOG}/{SCHEMA}/data"
RAW, PQ = f"{BASE}/raw", f"{BASE}/parquet"
print(f"raw     -> {RAW}\nparquet -> {PQ}")
print(f"splits  -> {SPLIT_N}  shards -> {SPLIT_SHARDS}")

# COMMAND ----------

import io, os, json, glob, random, shutil
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

WORDS = ["SAMPLE", "DRAFT", "CONFIDENTIAL", "50% OFF", "SALE", "NEW", "LIMITED",
         "2026", "WATERMARK", "PREVIEW", "COPY", "BEST PRICE"]

ARROW_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("split", pa.string()),
    ("input_jpg", pa.binary()),
    ("target_jpg", pa.binary()),
    ("mask_png", pa.binary()),
    ("words", pa.string()),       # JSON: [{"text":..., "box":[x0,y0,x1,y1]}] -> exact GT for OCR metric
])


def _font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def make_sample(img, rng, hard: bool):
    """Returns (input_with_text, clean_target, binary_mask, [{text, box}]).

    The mask is the dilated text bounding box -- the same approach real OTR uses
    (masks generated from word_bboxes), not a tight glyph outline.
    """
    target = img.convert("RGB").resize((RES, RES), Image.BICUBIC)
    inp = target.copy()
    mask = Image.new("L", (RES, RES), 0)
    d_in, d_mask = ImageDraw.Draw(inp), ImageDraw.Draw(mask)
    words = []
    for _ in range(rng.randint(2, 4) if hard else rng.randint(1, 2)):
        text = rng.choice(WORDS)
        font = _font(rng.randint(28, 64) if hard else rng.randint(36, 72))
        x0, y0 = rng.randint(0, RES - 200), rng.randint(0, RES - 80)
        box = d_in.textbbox((x0, y0), text, font=font)
        fill = tuple(rng.randint(200, 255) for _ in range(3)) if rng.random() < 0.5 \
            else tuple(rng.randint(0, 60) for _ in range(3))
        d_in.text((x0, y0), text, font=font, fill=fill)
        d_mask.rectangle([box[0] - 4, box[1] - 4, box[2] + 4, box[3] + 4], fill=255)
        words.append({"text": text, "box": [box[0] - 4, box[1] - 4, box[2] + 4, box[3] + 4]})
    return inp, target, mask, words


def enc(img, mode, fmt):
    b = io.BytesIO()
    img.convert(mode).save(b, format=fmt)
    return b.getvalue()

# COMMAND ----------

shards = sorted(glob.glob(f"{SRC}/*.parquet"))
assert shards, f"no source parquet at {SRC}"
total_needed = sum(SPLIT_N.values())
print(f"{len(shards)} source shards; need {total_needed} images")


def iter_images(limit):
    seen = 0
    for sp in shards:
        for rec in pq.read_table(sp).to_pylist():
            v = rec.get("image")
            if isinstance(v, dict):
                v = v.get("bytes")
            if not isinstance(v, (bytes, bytearray)):
                continue
            yield Image.open(io.BytesIO(v))
            seen += 1
            if seen >= limit:
                return


src_iter = iter_images(total_needed)
rng = random.Random(SEED)

# COMMAND ----------

summary = {}
for split, n in SPLIT_N.items():
    raw_dirs = {k: f"{RAW}/{split}/{k}" for k in ("input", "target", "mask", "meta")}
    pq_dir = f"{PQ}/{split}"
    for d in list(raw_dirs.values()) + [pq_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    n_raw = n if RAW_PER_SPLIT < 0 else min(RAW_PER_SPLIT, n)
    rows_per_shard = max(1, -(-n // SPLIT_SHARDS[split]))   # ceil
    rows, shard_i, written = [], 0, 0
    hard = split == "test"      # test split gets denser overlays

    def flush(rows, shard_i):
        if not rows:
            return 0
        pq.write_table(pa.Table.from_pylist(rows, schema=ARROW_SCHEMA),
                       f"{pq_dir}/part_{shard_i:04d}.parquet", compression="zstd")
        return len(rows)

    for i in range(n):
        sid = f"{split}_{i:06d}"
        inp, tgt, msk, words = make_sample(next(src_iter), rng, hard)
        # raw layout: only the first n_raw, for inspection/display
        if i < n_raw:
            inp.save(f"{raw_dirs['input']}/{sid}.png")
            tgt.save(f"{raw_dirs['target']}/{sid}.png")
            msk.save(f"{raw_dirs['mask']}/{sid}.png")
            with open(f"{raw_dirs['meta']}/{sid}.json", "w") as fh:
                json.dump({"id": sid, "split": split, "words": words}, fh)
        rows.append({"id": sid, "split": split,
                     "input_jpg": enc(inp, "RGB", "JPEG"),
                     "target_jpg": enc(tgt, "RGB", "JPEG"),
                     "mask_png": enc(msk, "L", "PNG"),
                     "words": json.dumps(words)})
        if len(rows) >= rows_per_shard:
            written += flush(rows, shard_i); rows, shard_i = [], shard_i + 1
    written += flush(rows, shard_i)

    files = sorted(glob.glob(f"{pq_dir}/*.parquet"))
    mb = sum(os.path.getsize(f) for f in files) / 1e6
    summary[split] = {"samples": written, "parquet_files": len(files),
                      "mb": round(mb, 1), "raw_pngs": n_raw, "hard_overlays": hard}
    print(f"{split:6s} {written:4d} samples -> {len(files):3d} shards, {mb:6.1f} MB, "
          f"{n_raw} raw PNGs, hard={hard}")

# COMMAND ----------

# Verify: read back through pyarrow exactly as the CLASSIC loader will.
for split in SPLIT_N:
    files = sorted(glob.glob(f"{PQ}/{split}/*.parquet"))
    t = pq.read_table(files[0])
    r = t.slice(0, 1).to_pylist()[0]
    inp = Image.open(io.BytesIO(r["input_jpg"]))
    msk = Image.open(io.BytesIO(r["mask_png"]))
    assert msk.getextrema() != (0, 0), f"{split}: mask is empty"
    print(f"{split:6s} cols={t.column_names} rows_in_shard={t.num_rows} "
          f"input={inp.size}{inp.mode} mask_extrema={msk.getextrema()} "
          f"words={len(json.loads(r['words']))}")

out = f"{BASE}/prepare_summary.json"
with open(out, "w") as fh:
    json.dump({"base": BASE, "splits": summary, "resolution": RES, "seed": SEED}, fh, indent=2)
print("\n" + json.dumps(summary, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Show a triplet
# MAGIC Why `raw/` exists: a reader can see the task rather than infer it from column names.

# COMMAND ----------

import glob as _g
sid = sorted(_g.glob(f"{RAW}/test/input/*.png"))[0]
name = os.path.basename(sid)
trip = [Image.open(f"{RAW}/test/{k}/{name}") for k in ("input", "mask", "target")]
W, H = trip[0].size
strip = Image.new("RGB", (W * 3 + 32, H + 36), (24, 24, 28))
d = ImageDraw.Draw(strip)
for i, (im, label) in enumerate(zip(trip, ["input (text overlaid)", "mask (text region)", "target (clean GT)"])):
    x = 8 + i * (W + 8)
    strip.paste(im.convert("RGB"), (x, 28))
    d.text((x + 4, 8), label, fill=(235, 235, 240))
meta = json.load(open(f"{RAW}/test/meta/{name.replace('.png', '.json')}"))
print(name, "->", [w["text"] for w in meta["words"]])
display(strip)
