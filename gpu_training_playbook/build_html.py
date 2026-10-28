#!/usr/bin/env python3
"""Render deck_content.SLIDES into one self-contained HTML deck.

Reads the SAME content module build_deck.py uses, so the HTML and the Google
Slides deck cannot drift.

Usage: python3 build_html.py [out.html]
Keys in the browser: arrows/space navigate, `o` overview grid, `p` print.
"""
import html
import re
import sys

from deck_content import SLIDES

dst = sys.argv[1] if len(sys.argv) > 1 else "deck.html"


def inline(t):
    """Markdown-ish inline -> HTML. Escape first so nothing can inject markup."""
    t = html.escape(str(t))
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<!['\"=>])(https?://[^\s<>)\]]+)",
               r'<a href="\1" target="_blank" rel="noopener">\1</a>', t)
    return t


def render_body(lines):
    """Body lines are plain strings; lines indented with spaces become sub-items."""
    out, buf, sub = [], [], []

    def flush():
        if buf:
            out.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in buf) + "</ul>")
            buf.clear()

    def flush_sub():
        if sub:
            out.append("<ul class='sub'>" + "".join(f"<li>{inline(x)}</li>" for x in sub) + "</ul>")
            sub.clear()

    for ln in lines:
        if ln.startswith("   ") or ln.startswith("\t"):
            # indented continuation: code-ish or nested detail
            txt = ln.strip()
            if re.match(r"^[\w.\[\]@*\-]+\s*[=({]|^(python|torchrun|databricks|mlflow|m =|from |import )", txt):
                flush()
                out.append(f"<pre><code>{html.escape(txt)}</code></pre>")
            else:
                flush()
                sub.append(txt)
        else:
            flush_sub()
            buf.append(ln)
    flush_sub()
    flush()
    return "\n".join(out)


def render_table(rows):
    cols = max(len(r) for r in rows)
    head = rows[0] + [""] * (cols - len(rows[0]))
    thead = "".join(f"<th>{inline(c)}</th>" for c in head)
    body = []
    for r in rows[1:]:
        r = list(r) + [""] * (cols - len(r))
        body.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
    return (f"<table><thead><tr>{thead}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table>")


cards, in_appendix = [], False
n = 0
for layout, title, body, table in SLIDES:
    if layout == "L_DIVIDER":
        in_appendix = True
    n += 1
    cls = "slide appendix" if in_appendix else "slide"
    if layout == "L_TITLE":
        cls = "slide title-slide"
    if layout == "L_DIVIDER":
        cls = "slide divider"
    inner = ""
    if body:
        inner += render_body(body)
    if table:
        inner += render_table(table)
    cards.append(
        f'<section class="{cls}" id="s{n}">'
        f'<header><h2>{inline(title)}</h2></header>'
        f'<div class="body">{inner}</div>'
        f'<footer><span>{inline(title)[:70]}</span>'
        f'<span class="num">{n} / {len(SLIDES)}</span></footer></section>')

SHELL = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Training Deep Learning &amp; LLMs on Databricks GPUs</title>
<style>
:root{--ink:#1B3139;--muted:#5A6E75;--red:#FF3621;--navy:#003159;--line:#DCE2E4;--bg:#F7F8F9;--code:#F1F3F5}
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:var(--bg);color:var(--ink);
  font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Helvetica,Arial,sans-serif}
#bar{position:fixed;top:0;left:0;height:3px;background:var(--red);width:0;z-index:50;transition:width .15s}
#deck{max-width:1180px;margin:0 auto;padding:28px 20px 80px}
.slide{background:#fff;border:1px solid var(--line);border-radius:10px;margin:0 0 26px;
  padding:30px 36px 14px;box-shadow:0 1px 3px rgba(0,0,0,.05);scroll-margin-top:16px}
.slide.title-slide{background:linear-gradient(135deg,#C4281C,#FF3621);border:none;color:#fff;padding:66px 36px 30px}
.slide.title-slide h2{color:#fff;font-size:40px}
.slide.title-slide li{color:#FFEDEA}
.slide.title-slide footer{border-top-color:rgba(255,255,255,.3);color:#FFD9D2}
.slide.divider{background:var(--navy);color:#fff;border:none}
.slide.divider h2{color:#fff;font-size:34px;margin:0}
.slide.divider footer{display:none}
.slide.appendix{border-left:4px solid var(--navy)}
header h2{margin:0 0 18px;font-size:27px;line-height:1.22;letter-spacing:-.015em}
p{margin:9px 0}
ul{margin:9px 0 9px 22px;padding:0}
ul.sub{margin:4px 0 8px 30px;color:var(--muted);font-size:14.5px}
li{margin:5px 0}
code{background:var(--code);padding:1px 5px;border-radius:4px;
  font:13.5px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
pre{background:var(--code);border:1px solid var(--line);border-radius:8px;
  padding:11px 14px;overflow:auto;margin:9px 0}
pre code{background:none;padding:0;font-size:13px}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:13.5px}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:left;vertical-align:top}
thead th{background:var(--navy);color:#fff;font-weight:600;font-size:13px}
tbody tr:nth-child(even){background:#FAFBFC}
strong{color:#0F2A33}
.title-slide strong{color:#fff}
a{color:var(--red)}
footer{display:flex;justify-content:space-between;gap:12px;margin-top:20px;padding-top:9px;
  border-top:1px solid var(--line);font-size:11.5px;color:var(--muted)}
footer .num{white-space:nowrap}
#hint{position:fixed;right:14px;bottom:12px;font-size:11.5px;color:var(--muted);
  background:#fff;border:1px solid var(--line);border-radius:999px;padding:5px 13px;z-index:40}
body.grid #deck{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:14px}
body.grid .slide{padding:14px 16px 8px;max-height:250px;overflow:hidden;cursor:pointer}
body.grid header h2{font-size:15px;margin-bottom:8px}
body.grid .body{font-size:11px;opacity:.72}
body.grid footer{display:none}
@media print{
  body{background:#fff}#hint,#bar{display:none}#deck{max-width:none;padding:0}
  .slide{page-break-after:always;break-after:page;border:none;box-shadow:none;
    border-radius:0;margin:0;padding:22px 26px;min-height:97vh}
}
</style></head>
<body>
<div id="bar"></div>
<div id="deck">
__CARDS__
</div>
<div id="hint">← → navigate &nbsp;·&nbsp; <b>o</b> overview &nbsp;·&nbsp; <b>p</b> print</div>
<script>
(function(){
  var slides=[].slice.call(document.querySelectorAll('.slide')),cur=0,bar=document.getElementById('bar');
  function mark(){bar.style.width=((cur+1)/slides.length*100)+'%';}
  function nearest(){var b=0,bd=1e9;slides.forEach(function(s,i){
    var d=Math.abs(s.getBoundingClientRect().top-10);if(d<bd){bd=d;b=i;}});return b;}
  function go(i){cur=Math.max(0,Math.min(slides.length-1,i));
    slides[cur].scrollIntoView({behavior:'smooth',block:'start'});mark();}
  document.addEventListener('keydown',function(e){
    if(e.metaKey||e.ctrlKey||e.altKey)return;var k=e.key;
    if(k==='ArrowRight'||k==='ArrowDown'||k===' '||k==='PageDown'){e.preventDefault();go(nearest()+1);}
    else if(k==='ArrowLeft'||k==='ArrowUp'||k==='PageUp'){e.preventDefault();go(nearest()-1);}
    else if(k==='Home'){e.preventDefault();go(0);}
    else if(k==='End'){e.preventDefault();go(slides.length-1);}
    else if(k==='o'||k==='O'){document.body.classList.toggle('grid');}
    else if(k==='p'||k==='P'){window.print();}
  });
  document.addEventListener('click',function(e){
    if(!document.body.classList.contains('grid'))return;
    var s=e.target.closest('.slide');if(!s)return;
    document.body.classList.remove('grid');go(slides.indexOf(s));
  });
  window.addEventListener('scroll',function(){cur=nearest();mark();},{passive:true});
  mark();
})();
</script>
</body></html>"""

open(dst, "w", encoding="utf-8").write(SHELL.replace("__CARDS__", "\n".join(cards)))
print(f"wrote {dst}: {len(SLIDES)} slides, "
      f"{sum(1 for s in SLIDES if s[3])} tables")
