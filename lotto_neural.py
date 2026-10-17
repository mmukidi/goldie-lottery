"""Neural sequence test: a small neural network reads the last K draws (as 0/1 grids) plus recency features and
predicts each number's chance of appearing next. Walk-forward on the full history, compared with random histories.
  python lotto_neural.py powerball [--nulls 2] [--lookback 12]"""
import sys, math, time, json, argparse, os, warnings
import numpy as np
from scipy import stats
from sklearn.neural_network import MLPClassifier
import lotto_fusion as LF
warnings.filterwarnings("ignore")

def feats(X, t, K):
    T, N = X.shape
    seq = np.zeros((K, N), np.float32)
    for k in range(1, K + 1):
        if t - k >= 0: seq[k - 1] = X[t - k]
    f50 = X[max(0, t - 50):t].mean(0) if t else np.zeros(N); f200 = X[max(0, t - 200):t].mean(0) if t else np.zeros(N)
    return np.concatenate([seq.ravel(), f50, f200]).astype(np.float32)

def run(X, K=12, t0=250, step=100, hidden=(64,), seed=0):
    T, N = X.shape
    F = np.stack([feats(X, t, K) for t in range(T)])
    P = np.zeros((T - t0, N), np.float32)
    for b0 in range(t0, T, step):
        clf = MLPClassifier(hidden_layer_sizes=hidden, alpha=0.05, learning_rate_init=0.002, max_iter=60, early_stopping=True,
                            validation_fraction=0.15, n_iter_no_change=6, random_state=seed)
        clf.fit(F[K:b0], X[K:b0].astype(int))
        pr = clf.predict_proba(F[b0:min(T, b0 + step)])
        P[b0 - t0:b0 - t0 + len(pr)] = pr
    a = LF.auc_rows(P, X[t0:]); m = LF.matches_top5(P, X[t0:])
    return float(a.mean()), float((a.mean() - .5) / (LF.AUC_SD / math.sqrt(len(a)))), float(m.mean())

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("game", choices=list(LF.WHITES)); ap.add_argument("--nulls", type=int, default=2)
    ap.add_argument("--lookback", type=int, default=12); ap.add_argument("--data-dir", default="data"); a = ap.parse_args()
    N = LF.WHITES[a.game]; dates, W, B = LF.load_long(a.game, a.data_dir); X = LF.to_X(W, N); T = len(X)
    t0 = time.time(); print(f"{a.game}: {T} draws; neural net (lookback {a.lookback} draws), walk-forward from draw 250, chance AUC 0.500, top-5 matches {25 / N:.3f}")
    res = {}
    for name, h in (("small net (64)", (64,)), ("deeper net (128,64)", (128, 64))):
        auc, z, m = run(X, a.lookback, hidden=h); res[name] = (auc, z, m)
        print(f"  {name:22s} AUC {auc:.4f} (z={z:+.1f})  top-5 picks avg matches {m:.3f}")
    rng = np.random.default_rng(5); nul = []
    for s in range(a.nulls):
        Xn = LF.to_X(LF.random_history(T, N, rng, 24)[0], N)
        r = [run(Xn, a.lookback, hidden=h, seed=s)[0] for h in ((64,), (128, 64))]; nul.append(r)
        print(f"  random history {s + 1}: AUC small {r[0]:.4f}, deeper {r[1]:.4f}")
    print(f"  ({time.time() - t0:.0f}s)")
    json.dump({"game": a.game, "real": res, "null": nul}, open(f"output/{a.game}_neural.json", "w"), indent=1)
