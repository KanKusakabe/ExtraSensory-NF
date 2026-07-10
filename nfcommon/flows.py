"""Shared conditional Normalizing-Flow scaffolding (zuko NSF + embedding/GRU encoder).

Reused across the life-log NF projects (PMData / ExtraSensory / GeoLife ...).
Mirrors the motionsim FlowHead: an encoder builds a context vector from continuous
features + categorical embeddings (e.g. user) + an optional GRU over a history
sequence; a zuko Neural Spline Flow gives exact log p(y | context).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import zuko


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Encoder(nn.Module):
    def __init__(self, cont_dim=0, cats=None, gru_in=0, emb=16, gru_hidden=32, out_dim=64):
        super().__init__()
        cats = cats or {}
        self.embs = nn.ModuleDict({k: nn.Embedding(n, emb) for k, n in cats.items()})
        self.cat_names = list(cats.keys())
        self.use_gru = gru_in > 0
        if self.use_gru:
            self.gru = nn.GRU(gru_in, gru_hidden, batch_first=True)
        in_dim = cont_dim + emb * len(self.cat_names) + (gru_hidden if self.use_gru else 0)
        assert in_dim > 0, "Encoder needs at least one conditioning input"
        self.mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(),
                                 nn.Linear(out_dim, out_dim), nn.ReLU())
        self.out_dim = out_dim

    def forward(self, cont=None, cats=None, hist=None):
        parts = []
        if cont is not None and cont.shape[-1] > 0:
            parts.append(cont)
        if cats:
            for k in self.cat_names:
                parts.append(self.embs[k](cats[k]))
        if self.use_gru and hist is not None:
            _, h = self.gru(hist)
            parts.append(h.squeeze(0))
        return self.mlp(torch.cat(parts, dim=-1))


class Model(nn.Module):
    """Conditional NSF over y (dim-D) given (cont, cats, hist)."""

    def __init__(self, dim, cont_dim=0, cats=None, gru_in=0,
                 transforms=3, hidden=(128, 128), ctx_out=64):
        super().__init__()
        self.enc = Encoder(cont_dim, cats, gru_in, out_dim=ctx_out)
        self.flow = zuko.flows.NSF(features=dim, context=self.enc.out_dim,
                                   transforms=transforms, hidden_features=hidden)

    def ctx(self, cont=None, cats=None, hist=None):
        return self.enc(cont, cats, hist)

    def log_prob(self, y, cont=None, cats=None, hist=None):
        return self.flow(self.ctx(cont, cats, hist)).log_prob(y)

    def nll(self, y, cont=None, cats=None, hist=None):
        return -self.log_prob(y, cont=cont, cats=cats, hist=hist).mean()

    @torch.no_grad()
    def sample(self, cont=None, cats=None, hist=None):
        """One sample per row of the given context batch. Returns [B, dim]."""
        c = self.ctx(cont, cats, hist)
        return self.flow(c).sample()


def standardize_fit(arr: np.ndarray):
    mu = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0) + 1e-6
    return mu, sd


def train_model(model, tensors, epochs=200, batch=256, lr=1e-3, patience=40, verbose=False):
    """tensors: dict with 'y' and optional 'cont','cats'(dict),'hist'; split by a boolean 'val' mask.
    Returns history list of dicts. Keeps best (lowest val NLL) weights in-place."""
    dev = device()
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def move(t):
        return t.to(dev) if torch.is_tensor(t) else t

    y = move(tensors["y"])
    cont = move(tensors.get("cont")) if tensors.get("cont") is not None else None
    cats = {k: move(v) for k, v in tensors.get("cats", {}).items()} or None
    hist = move(tensors.get("hist")) if tensors.get("hist") is not None else None
    val = tensors["val"].to(dev)
    tr = ~val

    def slice_all(mask):
        idx = mask.nonzero(as_tuple=True)[0]
        return idx

    tr_idx, va_idx = slice_all(tr), slice_all(val)

    def gather(idx, t):
        return None if t is None else t[idx]

    def gather_cats(idx):
        return None if cats is None else {k: v[idx] for k, v in cats.items()}

    hist_all = []
    best, best_state, bad = float("inf"), None, 0
    n = tr_idx.shape[0]
    for ep in range(epochs):
        model.train()
        perm = tr_idx[torch.randperm(n, device=dev)]
        tot = 0.0
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            loss = model.nll(y[b], cont=gather(b, cont), cats=gather_cats(b), hist=gather(b, hist))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        model.eval()
        with torch.no_grad():
            vnll = float(model.nll(y[va_idx], cont=gather(va_idx, cont),
                                   cats=gather_cats(va_idx), hist=gather(va_idx, hist)))
        hist_all.append({"epoch": ep, "train_nll": tot / n, "val_nll": vnll})
        if verbose and ep % 20 == 0:
            print(f"  ep {ep:3d} train {tot/n:7.3f} val {vnll:7.3f}")
        if vnll < best - 1e-4:
            best, best_state, bad = vnll, {k: v.detach().cpu().clone()
                                          for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return hist_all, best
