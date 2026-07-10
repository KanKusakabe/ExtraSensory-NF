# ExtraSensory-NF

実生活のスマホ/時計センサ（[ExtraSensory](http://extrasensory.ucsd.edu/) — 60人・毎分の行動＋自己申告の文脈）を
**条件付き Normalizing Flow** で学習し、「いまの行動の意外さ」と「生活型の個人差」を確率で扱う探索。

**▶ 結果ページ（GitHub Pages）**: https://kankusakabe.github.io/ExtraSensory-NF/

## バリエーション
- **V3 逐次サプライズ → 割り込み可否**: `p(今の行動 | 直近6分, user)`。サプライズが立つ瞬間＝行動の切り替わり/取り込み中。
- **V5 生活型の潜在地図（個人性）**: `p(行動 | user, 時刻)` の user 埋め込みで生活パターンを地図化＋個人化ゲイン。

各ページに **データ / 学習方法 / 結果 / 図の見方 / 解釈と使い道** を掲載。

## 再現
```bash
# ExtraSensory の per-uuid features+labels を data/ に展開し、
python build.py   # 全変種を学習→図→docs/（Pages）生成
```
`nfcommon/`（zuko NSF + 埋め込み/GRU エンコーダ + 指標 + 日本語Pages生成）を共有基盤として使用。

_NF×生活データ探索シリーズ。姉妹: PMData-NF / GeoLife-NF。_
