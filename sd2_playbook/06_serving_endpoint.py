# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Model Serving: create -> validate -> **delete**
# MAGIC
# MAGIC Stands up a **GPU serving endpoint** in front of the UC registered model, sends one real
# MAGIC request, then **tears it down in a `finally` block**.
# MAGIC
# MAGIC ## Why it deletes itself
# MAGIC
# MAGIC A GPU endpoint holds a GPU. `scale_to_zero_enabled` reduces idle cost but does **not**
# MAGIC eliminate it, and a cold replica must load a 2.58 GB pipeline before the first response — so
# MAGIC scale-to-zero is slow on first hit and always-on bills continuously. This playbook proves
# MAGIC the path end to end without leaving a meter running. For a real deployment, decide
# MAGIC explicitly whether the latency profile justifies always-on.
# MAGIC
# MAGIC The delete is in `finally` deliberately: a failed validation must not leak a GPU endpoint.
# MAGIC
# MAGIC ## Serving is runtime-agnostic
# MAGIC
# MAGIC Model Serving is its own managed compute — it does **not** run on the cluster or AI Runtime
# MAGIC environment you trained on. That is exactly why `pip_requirements` on the model version
# MAGIC matters: it is the only description of the environment the endpoint will build. Point this
# MAGIC notebook at either `sd2_inpaint_serverless` or `sd2_inpaint_classic`.
# MAGIC
# MAGIC Compute: **serverless CPU** — this notebook only calls REST APIs.

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text("catalog", "classic_stable_qxxm2v_catalog", "Catalog")
dbutils.widgets.text("schema", "sd2_playbook", "Schema")
dbutils.widgets.text("uc_model", "sd2_inpaint_serverless", "UC model")
dbutils.widgets.text("alias", "champion", "Alias")
dbutils.widgets.text("endpoint_name", "sd2-inpaint-playbook", "Endpoint name")
dbutils.widgets.dropdown("workload_type", "GPU_MEDIUM", ["GPU_SMALL", "GPU_MEDIUM"], "Workload type")
dbutils.widgets.text("ready_timeout_minutes", "45", "Ready timeout (min)")
dbutils.widgets.text("num_inference_steps", "20", "Denoising steps for the test request")
dbutils.widgets.dropdown("delete_when_done", "true", ["true", "false"], "Delete endpoint when done")

C = dbutils.widgets.get("catalog"); S = dbutils.widgets.get("schema")
MODEL = dbutils.widgets.get("uc_model")
UC_MODEL = f"{C}.{S}.{MODEL}"
ALIAS = dbutils.widgets.get("alias")
EP = dbutils.widgets.get("endpoint_name")
WORKLOAD = dbutils.widgets.get("workload_type")
TIMEOUT_MIN = int(dbutils.widgets.get("ready_timeout_minutes"))
STEPS = int(dbutils.widgets.get("num_inference_steps"))
DELETE = dbutils.widgets.get("delete_when_done") == "true"

import os, io, json, time, base64, requests
from PIL import Image, ImageDraw

_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()


def _opt(g):
    try:
        return g().get()
    except Exception:
        return None


HOST = "https://" + (_opt(_ctx.browserHostName) or spark.conf.get("spark.databricks.workspaceUrl"))
TOKEN = _opt(_ctx.apiToken)
assert TOKEN, "need a token to call the serving API"
H = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
print(f"endpoint : {EP}\nmodel    : {UC_MODEL}@{ALIAS}\nworkload : {WORKLOAD}")

# COMMAND ----------

result = {"endpoint": EP, "uc_model": UC_MODEL, "alias": ALIAS,
          "workload_type": WORKLOAD, "deleted": False}
created = False
try:
    # Resolve the alias to a concrete version: served_entities takes entity_version, not an alias.
    # NOTE the key asymmetry: this endpoint returns "version", while the MODEL object's
    # aliases[] array returns "version_num" for the same concept. Accept either.
    r = requests.get(f"{HOST}/api/2.1/unity-catalog/models/{UC_MODEL}/aliases/{ALIAS}",
                     headers=H, timeout=60)
    r.raise_for_status()
    _a = r.json()
    _v = _a.get("version", _a.get("version_num"))
    assert _v is not None, f"no version in alias response: {sorted(_a)[:12]}"
    VERSION = str(_v)
    result["version"] = VERSION
    print(f"{UC_MODEL}@{ALIAS} -> version {VERSION}")

    # ---- create ----
    body = {"name": EP, "config": {"served_entities": [{
        "name": "primary", "entity_name": UC_MODEL, "entity_version": VERSION,
        "workload_type": WORKLOAD, "workload_size": "Small", "scale_to_zero_enabled": True}]}}
    cr = requests.post(f"{HOST}/api/2.0/serving-endpoints", headers=H, json=body, timeout=120)
    if cr.status_code >= 400 and "already exists" in cr.text.lower():
        print("endpoint exists; updating served entities")
        requests.put(f"{HOST}/api/2.0/serving-endpoints/{EP}/config", headers=H,
                     json=body["config"], timeout=120).raise_for_status()
    else:
        cr.raise_for_status()
    created = True
    t0 = time.time()

    # ---- wait for READY (a diffusion pipeline takes minutes: env build + 2.58 GB load) ----
    state, last = None, ""
    while time.time() - t0 < TIMEOUT_MIN * 60:
        g = requests.get(f"{HOST}/api/2.0/serving-endpoints/{EP}", headers=H, timeout=60).json()
        st = g.get("state", {})
        state, cfg = st.get("ready"), st.get("config_update")
        line = f"ready={state} config_update={cfg} ({round(time.time()-t0)}s)"
        if line != last:
            print(line, flush=True); last = line
        if state == "READY":
            break
        if cfg == "UPDATE_FAILED":
            raise RuntimeError(f"config update failed: {json.dumps(g.get('state'))}")
        time.sleep(20)
    result["ready_seconds"] = round(time.time() - t0, 1)
    if state != "READY":
        raise TimeoutError(f"endpoint not READY after {TIMEOUT_MIN} min (state={state})")
    print(f"READY in {result['ready_seconds']}s")

    # ---- validate with one real request ----
    img = Image.new("RGB", (512, 512), (120, 120, 120))
    msk = Image.new("L", (512, 512), 0)
    ImageDraw.Draw(msk).rectangle([150, 200, 360, 260], fill=255)
    b64 = lambda im, f: base64.b64encode(
        (lambda b: (im.save(b, format=f), b.getvalue())[1])(io.BytesIO())).decode()
    payload = {"dataframe_records": [{"image": b64(img, "PNG"), "mask": b64(msk, "PNG"),
                                      "prompt": "remove text from image, clean background"}],
               "params": {"num_inference_steps": STEPS, "guidance_scale": 7.5}}
    t0 = time.time()
    inv = requests.post(f"{HOST}/serving-endpoints/{EP}/invocations", headers=H,
                        json=payload, timeout=600)
    result["invocation_seconds"] = round(time.time() - t0, 2)
    inv.raise_for_status()
    pred_b64 = inv.json()["predictions"][0]
    out_img = Image.open(io.BytesIO(base64.b64decode(pred_b64)))
    result["prediction_size"] = list(out_img.size)
    result["validated"] = True
    print(f"invocation OK in {result['invocation_seconds']}s -> {out_img.size}")

    pth = f"/Volumes/{C}/{S}/data/serving_prediction.png"
    out_img.save(pth)
    result["prediction_saved"] = pth
except Exception as e:
    result["error"] = f"{type(e).__name__}: {e}"
    print("FAILED:", result["error"])
    raise
finally:
    # A GPU endpoint must not survive a failure. Delete unconditionally unless asked not to.
    if created and DELETE:
        d = requests.delete(f"{HOST}/api/2.0/serving-endpoints/{EP}", headers=H, timeout=120)
        result["deleted"] = d.status_code < 400
        print(f"delete -> {d.status_code} (deleted={result['deleted']})")
    elif created:
        print(f"WARNING: endpoint '{EP}' left RUNNING (delete_when_done=false) — it bills.")
    with open(f"/Volumes/{C}/{S}/data/serving_{MODEL}.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
