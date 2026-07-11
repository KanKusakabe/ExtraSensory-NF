"""Japanese, self-explanatory GitHub-Pages generator for each NF variation.

Every variation page has the SAME 5 sections so that browsing the page alone tells
you: what data, how it was learned, what came out, how to read the figures, and
what it MEANS / how you'd use it.

    1. このデータは何か
    2. 何を・どう学習したか
    3. 結果
    4. 図の見方
    5. 解釈と意味・使い道

Call `write_all(docs_dir, repo_title, repo_desc, variations)`.
Figures are expected under `<docs_dir>/figures/` and referenced as `figures/<name>`.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

CSS = """
:root{--acc:#c2410c;--acc2:#d97757;--ink:#222;--mut:#666;--bg:#faf8f5}
*{box-sizing:border-box}
body{font:16px/1.75 -apple-system,"Hiragino Sans","Noto Sans JP",sans-serif;color:var(--ink);
 max-width:960px;margin:0 auto;padding:2rem 1rem}
a{color:var(--acc)}
h1{line-height:1.35;margin:.2rem 0}
.sub{color:var(--mut)}
.crumb{font-size:.9rem;margin-bottom:1rem}
.kpis{display:flex;gap:.8rem;flex-wrap:wrap;margin:1.2rem 0}
.kpi{background:#f5f3f0;border-radius:12px;padding:.7rem 1rem;min-width:120px}
.kpi b{display:block;font-size:1.5rem;color:var(--acc)}
.kpi span{font-size:.8rem;color:var(--mut)}
section{margin:2rem 0}
h2{border-left:5px solid var(--acc2);padding-left:.6rem;font-size:1.25rem}
figure{margin:1.4rem 0}
img{width:100%;border:1px solid #e5e5e5;border-radius:10px;background:#fff}
figcaption{color:#555;margin-top:.4rem;font-size:.92rem}
.interp{background:var(--bg);border-left:3px solid var(--acc2);padding:.8rem 1rem;border-radius:0 8px 8px 0}
code{background:#f0eee9;padding:.1rem .35rem;border-radius:4px}
table{border-collapse:collapse;width:100%;margin:1rem 0}
th,td{border:1px solid #e5e5e5;padding:.4rem .6rem;text-align:left;font-size:.94rem}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:1rem;margin-top:1.5rem}
.card{border:1px solid #e5e5e5;border-radius:12px;padding:1rem;text-decoration:none;color:inherit;display:block}
.card:hover{border-color:var(--acc2)}
.card h3{margin:.1rem 0 .3rem}
.badge{display:inline-block;font-size:.72rem;padding:.1rem .5rem;border-radius:999px;background:#eee;color:#555}
.badge.done{background:#e7f3e7;color:#256b25}
.badge.hold{background:#fdeecd;color:#8a5a00}
"""


def _kpis(metrics):
    if not metrics:
        return ""
    chips = "".join(f'<div class="kpi"><b>{html.escape(str(v))}</b><span>{html.escape(k)}</span></div>'
                    for k, v in metrics.items())
    return f'<div class="kpis">{chips}</div>'


def _figs(figures):
    out = []
    for f, cap in figures:
        out.append(f'<figure><img src="figures/{html.escape(f)}" alt="{html.escape(cap)}">'
                   f'<figcaption>{cap}</figcaption></figure>')
    return "\n".join(out)


def variation_html(repo_title, v):
    return f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(v['title'])} · {html.escape(repo_title)}</title>
<style>{CSS}</style></head><body>
<div class="crumb"><a href="index.html">&larr; {html.escape(repo_title)} トップ</a></div>
<h1>{html.escape(v['title'])}</h1>
<p class="sub">{v.get('tagline','')}</p>
{_kpis(v.get('metrics'))}
<section><h2>1. このデータは何か</h2>{v.get('data','')}</section>
<section><h2>2. 何を・どう学習したか</h2>{v.get('method','')}</section>
<section><h2>3. 結果</h2>{v.get('results','')}
{_figs(v.get('figures',[]))}</section>
<section><h2>4. 図の見方</h2>{v.get('howto','')}</section>
<section><h2>5. 解釈と意味・使い道</h2><div class="interp">{v.get('interpretation','')}</div></section>
<p class="sub" style="margin-top:2rem">条件付き Normalizing Flow（zuko NSF）で自動生成。
NF×生活データ探索シリーズの一部。</p>
</body></html>"""


def index_html(repo_title, repo_desc, variations, raw_intro="", outlook=""):
    cards = []
    for v in variations:
        st = v.get('status', 'done')
        cls = 'done' if st == 'done' else ('hold' if st == 'hold' else '')
        label = {'done': '完了', 'hold': '保留', '': st}.get(st, st)
        cards.append(
            f'<a class="card" href="{v["id"]}.html"><span class="badge {cls}">{label}</span>'
            f'<h3>{html.escape(v["title"])}</h3><p class="sub">{v.get("tagline","")}</p></a>')
    raw_block = (f'<section><h2>元データの中身（何が入っているか）</h2>'
                 f'<div class="interp">{raw_intro}</div></section>') if raw_intro else ""
    outlook_block = (f'<section><h2>考察：どんなフォーマットのデータがあれば何ができるか</h2>'
                     f'{outlook}</section>') if outlook else ""
    return f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(repo_title)}</title>
<style>{CSS}</style></head><body>
<h1>{html.escape(repo_title)}</h1>
<p class="sub">{repo_desc}</p>
<p>条件付き <b>Normalizing Flow</b> で「その人の"いつも"の密度」を学習し、
<code>SURPRISE = -log p</code> や生成・潜在で使う。各カードが1つのモデル（バリエーション）。
ページを開くと <b>データ / 学習方法 / 結果 / 図の見方 / 解釈と使い道</b> が分かる。</p>
{raw_block}
<h2>モデル一覧（クリックで各ページへ）</h2>
<div class="cards">{''.join(cards)}</div>
{outlook_block}
<p class="sub" style="margin-top:2rem">自動生成。NF×生活データ探索シリーズ。</p>
</body></html>"""


def write_all(docs_dir, repo_title, repo_desc, variations, raw_intro="", outlook=""):
    docs = Path(docs_dir)
    (docs / "figures").mkdir(parents=True, exist_ok=True)
    for v in variations:
        (docs / f"{v['id']}.html").write_text(variation_html(repo_title, v))
    (docs / "index.html").write_text(index_html(repo_title, repo_desc, variations, raw_intro, outlook))
    (docs / "manifest.json").write_text(json.dumps(
        [{k: v.get(k) for k in ('id', 'title', 'tagline', 'status', 'metrics')} for v in variations],
        ensure_ascii=False, indent=1))


def bullets(items):
    return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"
