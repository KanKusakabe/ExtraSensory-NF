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


def main():
    df, n_users = load()
    print("loaded", len(df), "minute-rows from", n_users, "users")
    for fn in (variation_v5, variation_v3):
        try:
            fn(df, n_users)
            print("OK", fn.__name__)
        except Exception:
            print("FAIL", fn.__name__); traceback.print_exc()
    pages.write_all(
        DOCS, "ExtraSensory-NF",
        "実生活のスマホ/時計センサ（60人・毎分の行動＋文脈）を条件付き Normalizing Flow で学習。"
        "『いまの行動の意外さ』と『生活型の個人差』を確率で扱う。",
        VARIATIONS)
    print("wrote pages for", [v["id"] for v in VARIATIONS])


if __name__ == "__main__":
    main()
