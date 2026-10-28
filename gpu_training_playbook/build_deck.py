#!/usr/bin/env python3
"""Build the training-playbook deck into a copy of the Databricks brand template.

Usage: python3 -I build_deck.py <presentation_id>
"""
import json
import subprocess
import sys
import urllib.request

PRES = sys.argv[1] if len(sys.argv) > 1 else sys.exit("need presentation id")
QUOTA = "gcp-sandbox-field-eng"
# Layout object IDs differ per template. Override with env vars when re-skinning:
#   L_TITLE / L_CONTENT / L_TITLE_ONLY / L_DIVIDER / L_CLOSING
# Discover them with:
#   GET /v1/presentations/<id>?fields=layouts(objectId,layoutProperties)
import os as _os
L_TITLE = _os.environ.get("L_TITLE", "p61")
L_CONTENT = _os.environ.get("L_CONTENT", "p64")
L_TITLE_ONLY = _os.environ.get("L_TITLE_ONLY", "p63")
L_DIVIDER = _os.environ.get("L_DIVIDER", "p84")
L_CLOSING = _os.environ.get("L_CLOSING", "")

# Slide object IDs must be unique across the presentation. Because we create the
# new slides BEFORE deleting the old ones (see below), a rebuild would collide
# with the previous run's IDs, so every run gets a fresh prefix.
import time as _time
RUN = _os.environ.get("RUN_TAG") or f"r{int(_time.time()) % 100000:05d}"


def token():
    return subprocess.run(["gcloud", "auth", "application-default", "print-access-token"],
                          capture_output=True, text=True, check=True).stdout.strip()


TOK = token()


def api(path, body=None, method=None):
    url = f"https://slides.googleapis.com/v1/presentations/{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method or ("POST" if body else "GET"))
    req.add_header("Authorization", f"Bearer {TOK}")
    req.add_header("x-goog-user-project", QUOTA)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode()[:600])
        raise


def batch(requests, chunk=40):
    out = []
    for i in range(0, len(requests), chunk):
        out.append(api(f"{PRES}:batchUpdate", {"requests": requests[i:i + chunk]}))
    return out


# ---------------------------------------------------------------- slide content
# (layout, title, body_lines, table)   table = [[header...], [row...], ...]
from deck_content import SLIDES as _SLIDES

_LAYOUT = {"L_TITLE": L_TITLE, "L_CONTENT": L_CONTENT,
           "L_TITLE_ONLY": L_TITLE_ONLY, "L_DIVIDER": L_DIVIDER}
SLIDES = [(_LAYOUT[k], t, b, tb) for (k, t, b, tb) in _SLIDES]

if L_CLOSING:
    SLIDES.append((L_CLOSING, "", None, None))

# ------------------------------------------------------------------ 1. create slides
# ORDER MATTERS: create the new slides BEFORE deleting the template's sample
# slides. Google Slides garbage-collects layouts that no slide references, so
# deleting everything first silently destroys the branded layouts and the next
# createSlide fails with "The object (<layoutId>) could not be found".
meta = api(f"{PRES}?fields=slides(objectId)")
stale = [x["objectId"] for x in meta.get("slides", [])]

reqs = []
for i, (layout, _t, _b, _tbl) in enumerate(SLIDES):
    # insertionIndex matters: a layout is only legal under the master that governs
    # the insertion point. Appending inherits the LAST sample slide's master, which
    # may not contain the branded layouts; inserting at the front uses the first's.
    reqs.append({"createSlide": {"objectId": f"{RUN}s{i:02d}",
                                 "insertionIndex": i,
                                 "slideLayoutReference": {"layoutId": layout}}})
batch(reqs)
print(f"created {len(SLIDES)} slides")

# ------------------------------------------------------------------ 2. drop samples
if stale:
    batch([{"deleteObject": {"objectId": o}} for o in stale])
    print(f"deleted {len(stale)} template sample slides")

# move our slides to the front in declared order
batch([{"updateSlidesPosition": {
    "slideObjectIds": [f"{RUN}s{i:02d}" for i in range(len(SLIDES))],
    "insertionIndex": 0}}])

# ------------------------------------------------------------------ 3. fill text
meta = api(f"{PRES}?fields=slides(objectId,pageElements(objectId,shape(placeholder(type),text)))")
by_id = {s["objectId"]: s for s in meta["slides"]}

fill = []
for i, (layout, title, body, tbl) in enumerate(SLIDES):
    sid = f"{RUN}s{i:02d}"
    els = by_id.get(sid, {}).get("pageElements", []) or []
    ph = {}
    for e in els:
        t = (e.get("shape", {}).get("placeholder") or {}).get("type")
        if t and t not in ph:
            ph[t] = e["objectId"]
    t_target = ph.get("TITLE") or ph.get("CENTERED_TITLE")
    b_target = ph.get("BODY") or ph.get("SUBTITLE")
    if title and t_target:
        fill.append({"insertText": {"objectId": t_target, "text": title}})
        # The branded title layouts use a narrow, fixed-position title box sitting
        # directly above the subtitle, so a title that wraps to 3 lines collides
        # with it. Cap the size on title slides.
        if layout == L_TITLE:
            fill.append({"updateTextStyle": {
                "objectId": t_target, "textRange": {"type": "ALL"},
                "style": {"fontSize": {"magnitude": 30, "unit": "PT"}},
                "fields": "fontSize"}})
    if body and b_target:
        fill.append({"insertText": {"objectId": b_target, "text": "\n".join(body)}})
        # shrink body text to fit: the template default overflows past ~5 wrapped lines
        n = sum(len(x) for x in body)
        pts = 16 if n < 400 else (14 if n < 650 else 12)
        fill.append({"updateTextStyle": {
            "objectId": b_target, "textRange": {"type": "ALL"},
            "style": {"fontSize": {"magnitude": pts, "unit": "PT"}}, "fields": "fontSize"}})
    elif not body and b_target:
        # table slides: remove the empty body box so it can't overlap the table
        fill.append({"deleteObject": {"objectId": b_target}})
batch(fill)
print(f"filled text on {len(SLIDES)} slides")

# ------------------------------------------------------------------ 4. tables
NAVY = {"red": 0.0, "green": 0.192, "blue": 0.349}
for i, (layout, title, body, tbl) in enumerate(SLIDES):
    if not tbl:
        continue
    sid, tid = f"{RUN}s{i:02d}", f"{RUN}t{i:02d}"
    rows, cols = len(tbl), max(len(r) for r in tbl)
    h = min(3.9, 0.42 * rows)
    api(f"{PRES}:batchUpdate", {"requests": [{"createTable": {
        "objectId": tid, "rows": rows, "columns": cols,
        "elementProperties": {
            "pageObjectId": sid,
            "size": {"width": {"magnitude": int(8.6 * 914400), "unit": "EMU"},
                     "height": {"magnitude": int(h * 914400), "unit": "EMU"}},
            "transform": {"scaleX": 1, "scaleY": 1,
                          "translateX": int(0.7 * 914400),
                          "translateY": int(1.55 * 914400), "unit": "EMU"}}}}]})
    cell = []
    populated = set()
    for r, row in enumerate(tbl):
        for c, val in enumerate(row):
            if val:
                populated.add((r, c))
                cell.append({"insertText": {"objectId": tid,
                                            "cellLocation": {"rowIndex": r, "columnIndex": c},
                                            "text": str(val)}})
    # header band + sizing
    cell.append({"updateTableCellProperties": {
        "objectId": tid,
        "tableRange": {"location": {"rowIndex": 0, "columnIndex": 0},
                       "rowSpan": 1, "columnSpan": cols},
        "tableCellProperties": {"tableCellBackgroundFill": {"solidFill": {"color": {"rgbColor": NAVY}}}},
        "fields": "tableCellBackgroundFill"}})
    # style only cells that actually contain text; the API rejects empty ones
    for (r, c) in sorted(populated):
        style = {"fontSize": {"magnitude": 9, "unit": "PT"}}
        fields = "fontSize"
        if r == 0:
            style["bold"] = True
            style["foregroundColor"] = {"opaqueColor": {"rgbColor": {"red": 1, "green": 1, "blue": 1}}}
            fields = "fontSize,bold,foregroundColor"
        cell.append({"updateTextStyle": {
            "objectId": tid, "cellLocation": {"rowIndex": r, "columnIndex": c},
            "textRange": {"type": "ALL"}, "style": style, "fields": fields}})
    batch(cell)
    print(f"  table on slide {i}: {rows}x{cols}")

print(f"\nDONE: https://docs.google.com/presentation/d/{PRES}/edit")
