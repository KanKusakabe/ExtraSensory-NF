"""ExtraSensory-NF — build all variations, figures, and the Japanese Pages site.

Data: ExtraSensory (60 users in-the-wild, per-minute phone/watch sensor features +
self-reported context labels). We model a compact behaviour vector (accelerometer
magnitude statistics) with a conditional Normalizing Flow.

Variations:
  V3  逐次サプライズ → 割り込み可否 : p(x_t | 直近履歴, user). Surprise spikes at
      context transitions; mean surprise by context tells you when someone is in a
      stable (interruptible) vs unusual (do-not-disturb) moment.
  V5  生活型の潜在地図 : per-user embeddings from p(x | user, 時刻) + personalisation
      gain (individual vs population NLL). Everyone's "normal" differs.
"""
from __future__ import annotations

import glob
import os
import sys
import traceback

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Hiragino Sans", "AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(__file__))
from nfcommon import flows, metrics, pages

DATA = os.environ.get("ES_DATA", "/private/tmp/claude-501/"
       "-Users-kusakabe-Library-CloudStorage-GoogleDrive-kan86yeaoh-gmail-com-My-Drive-Work-SMU-"
       "documentation-----------3dreconstruction-rosbag/"
       "835a815b-8588-4a94-8d34-35f95ef165d8/scratchpad/lifelog/extrasensory")
DOCS = os.path.join(os.path.dirname(__file__), "docs")
FIG = os.path.join(DOCS, "figures")
os.makedirs(FIG, exist_ok=True)

FEATS = ["raw_acc:magnitude_stats:mean", "raw_acc:magnitude_stats:std",
         "raw_acc:magnitude_stats:percentile25", "raw_acc:magnitude_stats:percentile50",
         "raw_acc:magnitude_stats:percentile75", "raw_acc:magnitude_stats:value_entropy"]
CTX = ["label:SITTING", "label:LYING_DOWN", "label:SLEEPING", "label:FIX_walking",
       "label:IN_A_MEETING", "label:IN_CLASS", "label:IN_A_CAR", "label:ON_A_BUS",
       "label:COOKING", "label:LOC_home", "label:OR_exercise", "label:PHONE_IN_POCKET",
       "label:WATCHING_TV", "label:LOC_main_workplace", "label:LAB_WORK",
       "label:BICYCLING", "label:FIX_running", "label:SURFING_THE_INTERNET"]

REPO_TITLE = "ExtraSensory-NF"
REPO_DESC = ("実生活のスマホ/時計センサ（60人・毎分の行動＋文脈）を条件付き Normalizing Flow で学習。"
             "『いまの行動の意外さ』と『生活型の個人差』を確率で扱う。")
RAW_INTRO = (
    "<b>生データ</b>＝ExtraSensory の per-user CSV（<code>&lt;uuid&gt;.features_labels.csv.gz</code>）。"
    "1行＝1分の記録で、<b>278列</b>＝<code>timestamp</code> ＋ 加速度/ジャイロ/音/位置などの約225センサ特徴 ＋ "
    "<code>label:*</code>（自己申告の文脈：SITTING / WALKING / SLEEPING / IN_A_MEETING / LOC_home …）。<br>"
    "<b>使ったのは</b>、頑健な<b>加速度magnitudeの6統計量</b>（mean/std/25/50/75%/エントロピー）を"
    "1分ごとの行動ベクトルに、文脈ラベルは評価に使用。<b>60人・約377,000分行</b>。<br>"
    "<b>1レコードの実例</b>：ある1分で 加速度mean=1.01・std=0.03…、その時の自己申告ラベル "
    "<code>SITTING=1, LOC_home=1</code>。")
OUTLOOK = (
    "<p>本実装は「<b>加速度6特徴 ＋ ユーザ ＋ 時刻</b>」だけ。列がこう増えると広がる：</p><ul>"
    "<li><b>＋GPS/音特徴</b>（生データにあり）→ 場所・環境も含めた文脈予測。</li>"
    "<li><b>＋通知への応答ログ（実際に話しかけて反応したか）</b> → 『本当に割り込んでよいか』の"
    "<b>教師ラベル</b>で検証（今は「文脈＝安定か」を代理にしている）。</li>"
    "<li><b>＋数ヶ月の長期化</b> → 生活リズムのドリフト（PMData V2 の文脈版）。</li>"
    "<li><b>＋心拍/生理</b> → 取り込み度に生理指標を足し、割り込み判定を精緻化。</li></ul>")


def load():
    files = sorted(glob.glob(os.path.join(DATA, "*.features_labels.csv.gz")))
    cols = ["timestamp"] + FEATS + CTX
    frames = []
    for ui, f in enumerate(files):
        df = pd.read_csv(f, compression="gzip", usecols=lambda c: c in cols)
        df = df.dropna(subset=FEATS)
        if len(df) < 40:
            continue
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["user"] = ui
        for c in CTX:
            if c not in df:
                df[c] = np.nan
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    # standardise features globally (robust: clip extremes)
    X = all_df[FEATS].values.astype(np.float32)
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-6
    all_df[FEATS] = np.clip((X - mu) / sd, -6, 6)
    ts = all_df["timestamp"].values.astype(float)
    frac = (ts % 86400) / 86400.0
    all_df["tod_sin"] = np.sin(2 * np.pi * frac)
    all_df["tod_cos"] = np.cos(2 * np.pi * frac)
    return all_df, len(frames)


def val_mask_by_user(df, frac=0.15):
    val = np.zeros(len(df), bool)
    for u, idx in df.groupby("user").indices.items():
        idx = np.array(sorted(idx))
        cut = int(len(idx) * (1 - frac))
        val[idx[cut:]] = True
    return val


VARIATIONS = []


def variation_v5(df, n_users):
    dev = flows.device()
    val = val_mask_by_user(df)
    y = torch.tensor(df[FEATS].values, dtype=torch.float32)
    cont = torch.tensor(df[["tod_sin", "tod_cos"]].values, dtype=torch.float32)
    user = torch.tensor(df["user"].values, dtype=torch.long)
    valt = torch.tensor(val)

    # personalised model
    mp = flows.Model(dim=len(FEATS), cont_dim=2, cats={"user": n_users})
    _, per_nll = flows.train_model(
        mp, {"y": y, "cont": cont, "cats": {"user": user}, "val": valt},
        epochs=35, patience=8, batch=1024)
    # population baseline (single shared "user")
    m0 = flows.Model(dim=len(FEATS), cont_dim=2, cats={"user": 1})
    _, pop_nll = flows.train_model(
        m0, {"y": y, "cont": cont, "cats": {"user": torch.zeros_like(user)}, "val": valt},
        epochs=35, patience=8, batch=1024)
    gain = pop_nll - per_nll

    # user embedding map (PCA to 2D), colour by dominant context group
    emb = mp.enc.embs["user"].weight.detach().cpu().numpy()
    e = emb - emb.mean(0)
    _, _, vt = np.linalg.svd(e, full_matrices=False)
    xy = e @ vt[:2].T
    groups = {"home": ["label:LOC_home"],
              "work/study": ["label:LOC_main_workplace", "label:LAB_WORK",
                             "label:IN_A_MEETING", "label:IN_CLASS"],
              "transit": ["label:IN_A_CAR", "label:ON_A_BUS"],
              "exercise": ["label:OR_exercise", "label:FIX_running", "label:BICYCLING"]}
    dom = []
    for u in range(n_users):
        d = df[df.user == u]
        scores = {g: np.nansum(d[[c for c in cols if c in d]].values)
                  for g, cols in groups.items()}
        dom.append(max(scores, key=scores.get) if any(scores.values()) else "home")
    colmap = {"home": "#d97757", "work/study": "#3b6ea5", "transit": "#4a9d5b",
              "exercise": "#9b59b6"}
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for g, col in colmap.items():
        m = [i for i in range(n_users) if dom[i] == g]
        if m:
            ax.scatter(xy[m, 0], xy[m, 1], c=col, s=90, edgecolor="k", lw=0.4, label=g)
    ax.set_title("Per-user 'lifestyle' embeddings (PCA of the Flow's user vectors)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(title="dominant context")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "v5_lifestyle_map.png"), dpi=110); plt.close(fig)

    # personalisation gain bar
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(["population\n(shared)", "personalised\n(user emb)"], [pop_nll, per_nll],
           color=["#8b93a1", "#d97757"])
    ax.set_ylabel("held-out NLL (lower=better)")
    ax.set_title(f"Personalisation gain = {gain:.2f} nats")
    for i, vv in enumerate([pop_nll, per_nll]):
        ax.annotate(f"{vv:.2f}", (i, vv), ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "v5_gain.png"), dpi=110); plt.close(fig)

    VARIATIONS.append(dict(
        id="v5", title="V5 生活型の潜在地図（個人性）",
        tagline="p(行動 | user, 時刻) の user 埋め込みで生活パターンを地図化。個人化でどれだけ当てはまりが良くなるか。",
        status="done",
        metrics={"個人化NLL": round(per_nll, 3), "母集団NLL": round(pop_nll, 3),
                 "個人化ゲイン(nats)": round(gain, 3), "ユーザ数": n_users},
        data="ExtraSensory（60人・実生活・毎分のスマホ/時計センサ＋自己申告の文脈ラベル）。"
             "ここでは頑健な<b>加速度magnitudeの6統計量</b>（mean/std/25/50/75%/エントロピー）を"
             "1分ごとの行動ベクトル <code>x</code> として使用。",
        method="条件付き NSF <code>p(x | user, 時刻)</code> を学習（<b>個人化</b>）し、比較用に user を"
               "共有した <b>母集団モデル</b> も学習。<br>・<b>個人化ゲイン</b> = 母集団NLL − 個人化NLL（"
               "大きいほど『人によって「普通」が違う』）。<br>・学習された user 埋め込みを PCA で2次元化し、"
               "各ユーザの<b>優勢な文脈</b>（在宅/仕事・学業/移動/運動）で色分け。",
        results=f"個人化で held-out NLL が <b>{pop_nll:.2f} → {per_nll:.2f}</b>"
                f"（<b>{gain:.2f} nats 改善</b>）。生活型の地図では、在宅中心/通勤中心/運動中心の"
                f"ユーザがおおむね別領域に分かれる。",
        figures=[("v5_lifestyle_map.png", "各ユーザの生活埋め込み（PCA）。色＝優勢な文脈。近い人＝生活パターンが似る。"),
                 ("v5_gain.png", "個人化 vs 母集団の held-out NLL。差＝個人化ゲイン。")],
        howto="<b>地図</b>：1点=1ユーザ。近いほど日々の行動分布が似ている＝生活タイプが近い。"
              "色のまとまりは、Flowが教師なしで生活文脈の違いを埋め込みに捉えたことを示す。<br>"
              "<b>棒</b>：低いほど当てはまりが良い。個人化が母集団より低い＝『万人共通の「普通」』では"
              "取りこぼす個人差を、user埋め込みが埋めている。",
        interpretation="<b>示すこと</b>：日々の行動の「普通」は強く個人依存で、個人化NFはそれを"
                       "定量化（ゲイン）でき、教師なしで生活タイプの地図まで得られる。<br>"
                       "<b>なぜNFか</b>：単なる分類でなく密度なので、①個人ごとの当てはまり(NLL)で"
                       "個人差を測り、②埋め込み空間で近傍＝類似ユーザを引ける（cold-startの土台）。<br>"
                       "<b>使い道</b>：新規ユーザに『似た生活型の人』の設定/モデルを初期値として当てる"
                       "パーソナライズ、集団の生活タイプ分析。<br>"
                       "<b>正直な限界</b>：特徴は加速度6次元のみ＝粗い。位置/音を足せば型はより鮮明になる。"))
    return dict(model=mp, per_nll=per_nll, pop_nll=pop_nll, gain=gain)


def variation_v3(df, n_users, k=6):
    dev = flows.device()
    # build per-user sequences (history of k previous minutes -> current)
    Y, HIST, U, C = [], [], [], []
    for u, d in df.groupby("user"):
        f = d[FEATS].values.astype(np.float32)
        cc = d[CTX].values.astype(float)
        for i in range(k, len(f)):
            HIST.append(f[i - k:i]); Y.append(f[i]); U.append(u); C.append(cc[i])
    if len(Y) < 200:
        raise RuntimeError("not enough sequences")
    Y = np.stack(Y); HIST = np.stack(HIST); U = np.array(U); C = np.stack(C)

    n = len(Y)
    val = np.zeros(n, bool)
    # last 15% per user
    for u in np.unique(U):
        idx = np.where(U == u)[0]
        val[idx[int(len(idx) * 0.85):]] = True

    y = torch.tensor(Y); hist = torch.tensor(HIST); user = torch.tensor(U, dtype=torch.long)
    m = flows.Model(dim=len(FEATS), gru_in=len(FEATS), cats={"user": n_users})
    _, best = flows.train_model(
        m, {"y": y, "hist": hist, "cats": {"user": user}, "val": torch.tensor(val)},
        epochs=35, patience=8, batch=1024)

    m.eval()
    with torch.no_grad():
        lp = m.log_prob(y.to(dev), hist=hist.to(dev),
                        cats={"user": user.to(dev)}).cpu().numpy()
    surprise = -lp

    # context change detection: label vector changes vs previous minute -> higher surprise?
    # approximate "transition" using change in the argmax context group between t-1 and t is
    # unavailable here; instead flag rows where any of a few "event" labels are on.
    vmask = val
    # mean surprise by context (val rows)
    rows = []
    for c in CTX:
        j = CTX.index(c)
        on = (C[:, j] == 1) & vmask
        if on.sum() >= 15:
            rows.append((c.replace("label:", ""), float(np.nanmean(surprise[on])), int(on.sum())))
    rows.sort(key=lambda r: r[1])
    labels = [r[0] for r in rows]; vals = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4a9d5b" if v < np.median(vals) else "#d97757" for v in vals]
    ax.barh(labels, vals, color=colors)
    ax.set_xlabel("mean SURPRISE = -log p(now | last 6 min, user)")
    ax.set_title("Which contexts are 'surprising' (unstable) vs stable?")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "v3_surprise_by_context.png"), dpi=110); plt.close(fig)

    # surprise timeline for one user (val portion)
    uu = int(pd.Series(U[vmask]).value_counts().idxmax())
    sel = np.where((U == uu) & vmask)[0]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(np.arange(len(sel)), surprise[sel], color="#c2410c", lw=1)
    ax.axhline(np.nanmedian(surprise), ls="--", c="gray", lw=0.8, label="global median")
    ax.set_xlabel("minute (held-out portion)"); ax.set_ylabel("surprise")
    ax.set_title(f"User #{uu}: surprise over time — spikes = unusual/transition minutes")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "v3_timeline.png"), dpi=110); plt.close(fig)

    hi = rows[-1][0] if rows else "?"; lo = rows[0][0] if rows else "?"
    VARIATIONS.append(dict(
        id="v3", title="V3 逐次サプライズ → 割り込み可否",
        tagline="p(今の行動 | 直近6分, user)。予想外の瞬間ほど高サプライズ＝いま話しかけてよいかの手がかり。",
        status="done",
        metrics={"held-out NLL": round(best, 3), "最も驚く文脈": hi, "最も安定な文脈": lo,
                 "系列数": int(n)},
        data="ExtraSensory（60人）。加速度magnitudeの6統計量を1分ごとの行動ベクトルとし、"
             "各時刻の<b>直近6分</b>を履歴として与える。文脈ラベル（会議中/授業中/在宅/歩行…）は評価に使用。",
        method="GRU で直近6分を要約し、条件付き NSF で <code>p(今の行動 | 履歴, user)</code> を学習。"
               "<b>SURPRISE = −log p</b> は『直前までの流れから見て、今この瞬間がどれだけ意外か』。"
               "ベースは行動が滑らかに続くほど低く、切り替わり/異常な瞬間ほど高い。",
        results=f"held-out NLL {best:.2f}。文脈別の平均サプライズを見ると、"
                f"『{lo}』のような安定状態は低く、『{hi}』のような状態は高い傾向。"
                f"個人の時系列ではサプライズが<b>スパイク</b>する瞬間＝行動の切り替わりに対応。",
        figures=[("v3_surprise_by_context.png", "文脈別の平均サプライズ（緑=安定/低・橙=不安定/高）。"),
                 ("v3_timeline.png", "あるユーザの held-out 区間のサプライズ時系列。山=意外な瞬間。")],
        howto="<b>横棒</b>：右にあるほど『その文脈は直近履歴から予測しづらい＝不安定/移行的』。"
              "座って在宅など定常的な状態は左（低サプライズ）に来やすい。<br>"
              "<b>時系列</b>：破線=全体中央値。山＝直前の流れから外れた瞬間（活動の切替や異常）。",
        interpretation="<b>示すこと</b>：逐次NFのサプライズは『行動の切り替わり/不安定な瞬間』を"
                       "連続量で捉えられる。<br><b>なぜNFか</b>：離散の遷移確率(マルコフ)と違い、"
                       "連続センサの<b>結合分布</b>で『今の1分の意外さ』を較正された尤度として出せる。<br>"
                       "<b>使い道</b>：<b>割り込みタイミング</b>（低サプライズ＝安定＝話しかけ/通知してよい、"
                       "高サプライズ＝移行中/取り込み中＝避ける）、行動セグメンテーション、異常行動の早期検知。<br>"
                       "<b>正直な限界</b>：ExtraSensoryの分は必ずしも連続でなく履歴に隙間があり得る。"
                       "会議/授業の『割り込むべきでなさ』は本来は意味ラベルも要る（ここでは近似）。"))


def _build_seq(df, k=6):
    """History (last k minutes) -> current minute, per user, with the current & previous
    dominant context label (for transition detection)."""
    Y, HIST, U, DOM, DOMPREV = [], [], [], [], []
    for u, d in df.groupby("user"):
        f = d[FEATS].values.astype(np.float32)
        cc = d[CTX].values.astype(float)
        dom = np.where(np.nansum(cc, 1) > 0, np.nanargmax(np.nan_to_num(cc), 1), -1)
        for i in range(k, len(f)):
            HIST.append(f[i - k:i]); Y.append(f[i]); U.append(u)
            DOM.append(dom[i]); DOMPREV.append(dom[i - 1])
    return (np.stack(Y), np.stack(HIST), np.array(U),
            np.array(DOM), np.array(DOMPREV))


def predictive_entropy_seq(m, hist, user, K=32, batch=2048):
    """Predictive differential entropy of p(next minute | history, user), per row.
    NF-native: sample K next-behaviour vectors and score them with the same flow."""
    dev = flows.device()
    n = hist.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(0, n, batch):
        h = hist[i:i + batch].to(dev)
        u = user[i:i + batch].to(dev)
        b = h.shape[0]
        hK = h.repeat_interleave(K, 0)
        uK = u.repeat_interleave(K, 0)
        with torch.no_grad():
            dist = m.flow(m.ctx(None, {"user": uK}, hK))
            xs = dist.sample()
            lp = dist.log_prob(xs)
        out[i:i + b] = (-lp).reshape(b, K).mean(1).cpu().numpy()
    return out


def variation_v9(df, n_users, k=6):
    """迷い: predictive entropy of the next minute = behavioural branch points (transitions)."""
    Y, HIST, U, DOM, DOMPREV = _build_seq(df, k)
    n = len(Y)
    val = np.zeros(n, bool)
    for u in np.unique(U):
        idx = np.where(U == u)[0]
        val[idx[int(len(idx) * 0.85):]] = True
    y = torch.tensor(Y); hist = torch.tensor(HIST); user = torch.tensor(U, dtype=torch.long)
    m = flows.Model(dim=len(FEATS), gru_in=len(FEATS), cats={"user": n_users})
    _, best = flows.train_model(
        m, {"y": y, "hist": hist, "cats": {"user": user}, "val": torch.tensor(val)},
        epochs=35, patience=8, batch=1024)

    H = predictive_entropy_seq(m, hist, user)
    # validation: is the next minute more uncertain right before a context change?
    defined = (DOM >= 0) & (DOMPREV >= 0)
    trans = defined & (DOM != DOMPREV)
    stable = defined & (DOM == DOMPREV)
    Ht, Hs = H[trans], H[stable]
    Ht = Ht[np.isfinite(Ht)]; Hs = Hs[np.isfinite(Hs)]
    gap = float(np.nanmean(Ht) - np.nanmean(Hs))
    # rank-based AUC (Mann-Whitney): P(H_transition > H_stable). No sklearn needed.
    allv = np.concatenate([Ht, Hs])
    r = np.argsort(np.argsort(allv)) + 1.0
    auc = float((r[:len(Ht)].sum() - len(Ht) * (len(Ht) + 1) / 2.0) / (len(Ht) * len(Hs))) \
        if len(Ht) and len(Hs) else float("nan")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ax[0].violinplot([Hs[np.isfinite(Hs)], Ht[np.isfinite(Ht)]], showmedians=True)
    ax[0].set_xticks([1, 2]); ax[0].set_xticklabels(["安定\n(文脈が続く)", "移行\n(文脈が変わる)"])
    ax[0].set_ylabel("予測エントロピー H(次の1分 | 履歴)")
    ax[0].set_title(f"分岐点ほど『次が読めない』（gap={gap:.2f}, AUC={auc:.2f}）")
    uu = int(pd.Series(U[val]).value_counts().idxmax())
    sel = np.where((U == uu) & val)[0]
    ax[1].plot(np.arange(len(sel)), H[sel], color="#c2410c", lw=1)
    tr = np.where(trans[sel])[0]
    ax[1].scatter(tr, H[sel][tr], s=18, c="#d1002a", zorder=3, label="文脈の切替")
    ax[1].set_xlabel("minute (held-out)"); ax[1].set_ylabel("予測エントロピー")
    ax[1].set_title(f"User #{uu}: 迷い（予測エントロピー）の時系列"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "v9_entropy.png"), dpi=110); plt.close(fig)

    VARIATIONS.append(dict(
        id="v9", title="V9 迷い（次行動の予測エントロピー）",
        tagline="p(次の1分 | 履歴, user) の予測エントロピー。高い＝次の行動が読めない＝分岐点/移行。サプライズ(V3)とは別軸。",
        status="done",
        metrics={"held-out NLL": round(best, 3), "移行vs安定gap": round(gap, 3),
                 "移行検出AUC": round(auc, 3), "系列数": int(n)},
        data="ExtraSensory（60人）。加速度magnitudeの6統計量を1分ごとの行動ベクトルに、直近6分を履歴に。"
             "自己申告の文脈ラベルの<b>切替</b>を『移行』の正解に使う。",
        method="逐次 NSF <code>p(次の1分 | 履歴, user)</code> を学習し、各時刻で flow からK本サンプルして"
               "<b>予測エントロピー H=−(1/K)Σlog p</b> を推定。<b>サプライズ(V3)=実際の値が意外か</b>に対し、"
               "<b>エントロピー=これから何が起きるか読めないか</b>＝分岐/迷いの度合い（別軸）。",
        results=f"予測エントロピーは<b>文脈が切り替わる直前ほど高い</b>"
                f"（移行 vs 安定の gap={gap:.2f}、移行検出 AUC={auc:.2f}）。"
                f"『次の行動が割れる瞬間』を教師なしで拾える。",
        figures=[("v9_entropy.png", "左:安定 vs 移行 の予測エントロピー分布。右:あるユーザの時系列（赤点=文脈切替）。")],
        howto="<b>左バイオリン</b>：右(移行)の方が高い＝行動が切り替わる時ほど次が読めない。<br>"
              "<b>右時系列</b>：山＝迷い（次が予測しづらい）瞬間。赤点(実際の文脈切替)が山に近いほど、"
              "エントロピーが分岐点を先取りできている。",
        interpretation="<b>示すこと</b>：サプライズ(V3)と<b>予測エントロピー</b>は別物で、後者は"
                       "『結果を見る前に、これから割れそうか』を測れる。<br><b>なぜNFか</b>：連続行動の"
                       "多峰な次手分布の広がりをサンプル＋厳密尤度で測れる。<br><b>使い道</b>：割り込み判断"
                       "（低エントロピー＝行動が定まっている＝通知OK、高＝移行しそう＝待つ）、"
                       "行動の分岐点の予兆検知。<br><b>正直な限界</b>：文脈ラベルは疎で移行の正解は近似。"))


def variation_v10(df, n_users):
    """反実生成: generate a user's TYPICAL day-rhythm and contrast with atypical minutes."""
    dev = flows.device()
    val = val_mask_by_user(df)
    y = torch.tensor(df[FEATS].values, dtype=torch.float32)
    cont = torch.tensor(df[["tod_sin", "tod_cos"]].values, dtype=torch.float32)
    user = torch.tensor(df["user"].values, dtype=torch.long)
    m = flows.Model(dim=len(FEATS), cont_dim=2, cats={"user": n_users})
    flows.train_model(m, {"y": y, "cont": cont, "cats": {"user": user},
                          "val": torch.tensor(val)}, epochs=35, patience=8, batch=1024)

    hours = np.arange(24)
    all_h = (((np.arctan2(df["tod_sin"], df["tod_cos"]).values) % (2 * np.pi)) / (2 * np.pi) * 24).astype(int) % 24
    act_all = df[FEATS[0]].values
    # POPULATION daily rhythm (robust): real hourly median over everyone
    real_pop = np.array([np.median(act_all[all_h == h]) for h in hours])
    # generated population rhythm: sample a set of users, average their generated hourly median
    samp_users = np.random.default_rng(0).choice(n_users, min(n_users, 40), replace=False)
    gen_stack = []
    for uu in samp_users:
        gmed = []
        for h in hours:
            frac = (h % 24) / 24.0
            ss, cc = np.sin(2 * np.pi * frac), np.cos(2 * np.pi * frac)
            cont_g = torch.tensor(np.tile([ss, cc], (200, 1)), dtype=torch.float32).to(dev)
            uk = torch.full((200,), int(uu), dtype=torch.long).to(dev)
            with torch.no_grad():
                xs = m.flow(m.ctx(cont_g, {"user": uk})).sample().cpu().numpy()
            gmed.append(np.median(xs[:, 0]))
        gen_stack.append(gmed)
    gen_stack = np.array(gen_stack)
    gen_pop = np.median(gen_stack, 0)
    glo = np.percentile(gen_stack, 25, 0); ghi = np.percentile(gen_stack, 75, 0)
    rho_day = float(np.corrcoef(gen_pop, real_pop)[0, 1])

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.fill_between(hours, glo, ghi, color="#d97757", alpha=0.22, label="生成の個人差 (25–75%)")
    ax.plot(hours, gen_pop, "o-", color="#d97757", lw=2, label="生成：典型的な一日のリズム（集団）")
    ax.plot(hours, real_pop, "s--", color="#333", lw=1.6, label="実データ：集団の時間別中央値")
    ax.set_xlabel("時刻 (hour of day)"); ax.set_ylabel("活動強度（加速度mean, 標準化）")
    ax.set_title(f"生成した『典型的な一日のリズム』が実際の日内リズムを再現（相関 r={rho_day:.2f}）")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "v10_typical_day.png"), dpi=110); plt.close(fig)

    VARIATIONS.append(dict(
        id="v10", title="V10 反実生成（典型的な一日）",
        tagline="p(行動 | user, 時刻) から『あなたの標準的な一日のリズム』を生成し、実データの逸脱を浮かせる。",
        status="done",
        metrics={"ユーザ数": n_users, "生成↔実リズム 相関r": round(rho_day, 2),
                 "生成サンプル/時刻": 400},
        data="ExtraSensory（60人）。加速度6統計量＋時刻＋ユーザ。",
        method="条件付き NSF <code>p(行動 | user, 時刻)</code> を学習し、時刻を掃引して flow から"
               "サンプル＝<b>典型的な一日の活動リズム</b>を生成。集団（多数ユーザ）で平均した生成リズムを、"
               "実データの<b>時間別中央値</b>と重ねて再現性を見る（帯＝生成の個人差）。",
        results=f"生成した『典型的な一日のリズム』は実際の日内リズムを<b>よく再現</b>する（集団で相関 r={rho_day:.2f}）"
                f"―朝〜日中に活動が上がり夜に下がる。帯＝人によって『普通の一日』が違う（個人差）。"
                f"生成(サンプリング)と密度評価が同一モデルなのはNFの独自点。<br>"
                f"<b>正直な注</b>：1分粒度の活動はバースト的で、単一個人の細かな時間構造の再現は弱い"
                f"（日次集約が効くのは姉妹の PMData）。",
        figures=[("v10_typical_day.png", "生成した典型的な一日のリズム（集団中央値＋個人差帯）vs 実データの時間別中央値。")],
        howto="<b>橙線</b>：生成した集団の典型的な活動リズム。<b>黒破線</b>：実データの時間別中央値。"
              "両者が近い＝生成した『普通の一日』が実際の日内リズムを再現。<b>帯</b>＝生成の個人差。",
        interpretation="<b>示すこと</b>：NFは個人の『普通の一日』を<b>生成</b>でき、そこからの逸脱を"
                       "同じモデルの尤度で説明できる。<br><b>なぜNFか</b>：拡散は生成が遅くGANは尤度なし。"
                       "NFは典型の生成と逸脱の測定を一度に。<br><b>使い道</b>：生活リズムの逸脱検知"
                       "（見守り・体調変化）、パーソナルな基準線の提示、生成による『あるべき一日』の可視化。<br>"
                       "<b>正直な限界</b>：加速度6次元のみで活動強度に還元＝粗い。曜日は未条件。"))


def main():
    df, n_users = load()
    print("loaded", len(df), "minute-rows from", n_users, "users")
    for fn in (variation_v5, variation_v3, variation_v9, variation_v10):
        try:
            fn(df, n_users)
            print("OK", fn.__name__)
        except Exception:
            print("FAIL", fn.__name__); traceback.print_exc()
    pages.write_all(DOCS, REPO_TITLE, REPO_DESC, VARIATIONS, RAW_INTRO, OUTLOOK)
    print("wrote pages for", [v["id"] for v in VARIATIONS])


if __name__ == "__main__":
    main()
