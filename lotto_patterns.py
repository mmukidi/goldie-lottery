"""
Pattern hunter: throws a statistical randomness battery AND machine-learning models at the draw history.
Everything is judged against what pure chance produces, because with this few draws a "pattern" is
easy to imagine and hard to prove. Anything that survives multiple-testing correction is flagged.

  python lotto_patterns.py powerball            # or megamillions
  python lotto_patterns.py powerball --sims 6   # more null simulations (slower)
"""
import sys, os, math, time, datetime as dt, argparse, json, warnings
import numpy as np
from scipy import stats
import lotto_predictor as L
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ helpers
def to_matrix(whites_list, N):
    X = np.zeros((len(whites_list), N), dtype=np.int8)
    for t, w in enumerate(whites_list):
        for n in w: X[t, n - 1] = 1
    return X

def bh(pvals, q=0.05):
    """Benjamini-Hochberg: number of discoveries at false-discovery rate q."""
    p = np.array(sorted(pvals)); m = len(p)
    ok = np.where(p <= q * (np.arange(1, m + 1) / m))[0]
    return (ok.max() + 1) if len(ok) else 0

def simulate(T, N, rng):
    return [sorted(rng.choice(N, 5, replace=False) + 1) for _ in range(T)]

# ------------------------------------------------------------------ A. statistical battery
def battery(W, B, N, NB, dates, rng, nsim=2000):
    T = len(W); X = to_matrix(W, N); XB = np.zeros((T, NB), dtype=np.int8)
    for t, b in enumerate(B): XB[t, b - 1] = 1
    rows = []                                   # (name, statistic text, p-value, family)
    add = lambda name, s, p, fam="single": rows.append((name, s, float(p), fam))

    # 1 uniformity
    c = X.sum(0); chi = ((c - 5 * T / N) ** 2 / (5 * T / N)).sum()
    add("White balls equally likely (chi-square)", f"chi2={chi:.1f}, df={N - 1}", stats.chi2.sf(chi, N - 1))
    cb = XB.sum(0); chib = ((cb - T / NB) ** 2 / (T / NB)).sum()
    add("Bonus ball equally likely (chi-square)", f"chi2={chib:.1f}, df={NB - 1}", stats.chi2.sf(chib, NB - 1))
    # 2 decades / low-high
    dec = np.array([(n - 1) // 10 for n in range(1, N + 1)]); dc = np.array([c[dec == d].sum() for d in np.unique(dec)])
    de = np.array([(dec == d).sum() for d in np.unique(dec)]) / N * 5 * T
    add("Decade spread (chi-square)", f"chi2={((dc - de) ** 2 / de).sum():.1f}", stats.chi2.sf(((dc - de) ** 2 / de).sum(), len(dc) - 1))
    # 3 repeats from previous draw / lag overlap
    for lag in (1, 2, 3, 5, 10):
        ov = np.array([len(set(W[t]) & set(W[t - lag])) for t in range(lag, T)]); e = 25 / N
        v = 5 * (5 / N) * (1 - 5 / N) * (N - 5) / (N - 1)        # hypergeometric variance
        z = (ov.mean() - e) / math.sqrt(v / len(ov))
        add(f"Numbers repeated from {lag} draw(s) ago", f"avg {ov.mean():.3f} vs chance {e:.3f}", 2 * stats.norm.sf(abs(z)), "lag")
    # 4 serial correlation per number (Ljung-Box lags 1-3), pooled
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        ps = []
        for n in range(N):
            if X[:, n].std() == 0: continue
            ps.append(float(acorr_ljungbox(X[:, n].astype(float), lags=[3], return_df=True)["lb_pvalue"].iloc[0]))
        k = bh(ps)
        add("Per-number serial correlation (Ljung-Box, all numbers)", f"{sum(p < .05 for p in ps)} of {len(ps)} p<.05 (chance ~{.05 * len(ps):.1f}); {k} survive correction",
            stats.binomtest(sum(p < .05 for p in ps), len(ps), .05, alternative="greater").pvalue)
    except Exception as e:
        add("Per-number serial correlation", f"skipped ({e})", 1.0)
    # 5 pair affinity
    pc = {}
    for w in W:
        for i in range(5):
            for j in range(i + 1, 5): pc[(w[i], w[j])] = pc.get((w[i], w[j]), 0) + 1
    ppair = 20 / (N * (N - 1)); npairs = N * (N - 1) // 2
    pv = [stats.binom.sf(pc.get((a, b), 0) - 1, T, ppair) for a in range(1, N + 1) for b in range(a + 1, N + 1)]
    mx = max(pc.values()); sims = []
    for _ in range(300):
        S = simulate(T, N, rng); d = {}
        for w in S:
            for i in range(5):
                for j in range(i + 1, 5): d[(w[i], w[j])] = d.get((w[i], w[j]), 0) + 1
        sims.append(max(d.values()))
    add("Pairs that appear together unusually often", f"most frequent pair {max(pc, key=pc.get)} x{mx}; {bh(pv)} survive correction of {npairs} pairs",
        (np.array(sims) >= mx).mean() + 1 / 301)
    # 6 gaps vs geometric
    def gap_stat(M):
        g = np.concatenate([np.diff(np.where(M[:, n] == 1)[0]) for n in range(N)])
        return g.mean(), np.mean(g == 1), np.percentile(g, 90)
    obs = gap_stat(X); sims_g = np.array([gap_stat(to_matrix(simulate(T, N, rng), N)) for _ in range(300)])
    # same statistic on random histories of the same length (corrects for the window cutting off long gaps)
    zg = [(obs[i] - sims_g[:, i].mean()) / (sims_g[:, i].std() + 1e-9) for i in range(3)]
    pg = (np.abs((sims_g - sims_g.mean(0)) / (sims_g.std(0) + 1e-9)).max(1) >= max(abs(z) for z in zg)).mean() + 1 / 301
    add("Gap between repeats of a number (vs random histories of same length)", f"avg gap {obs[0]:.1f}, immediate-repeat share {obs[1]:.3f}, 90th pct {obs[2]:.0f}", pg)
    # 7 runs tests
    def runs_p(x):
        x = np.asarray(x); b = x > np.median(x)
        n1, n2 = b.sum(), (~b).sum()
        if n1 == 0 or n2 == 0: return 1.0
        r = 1 + (b[1:] != b[:-1]).sum(); mu = 2 * n1 * n2 / (n1 + n2) + 1
        sd = math.sqrt((mu - 1) * (mu - 2) / (n1 + n2 - 1)); return 2 * stats.norm.sf(abs((r - mu) / sd))
    sums = np.array([sum(w) for w in W]); odd = np.array([sum(n % 2 for n in w) for w in W]); low = np.array([sum(n <= N // 2 for n in w) for w in W])
    add("Draw sums: runs above/below median", f"{len(sums)} draws", runs_p(sums))
    add("Odd-number count: runs test", "", runs_p(odd)); add("Low-number count: runs test", "", runs_p(low))
    add(f"Bonus ball: runs test", "", runs_p(np.array(B)))
    # 8 spectrum / cycles: Fisher g on every number + draw sums, MC p-value
    def fisher_g(x):
        x = np.asarray(x, float) - np.mean(x)
        if x.std() == 0: return 0
        P = np.abs(np.fft.rfft(x)[1:]) ** 2; return P.max() / P.sum()
    gs = np.array([fisher_g(X[:, n]) for n in range(N)])
    null_max = []
    for _ in range(300):
        S = to_matrix(simulate(T, N, rng), N); null_max.append(max(fisher_g(S[:, n]) for n in range(N)))
    add("Hidden cycles in any number (FFT peak vs random)", f"strongest peak g={gs.max():.3f} (number {gs.argmax() + 1})", (np.array(null_max) >= gs.max()).mean() + 1 / 301)
    gsum = fisher_g(sums); nm = [fisher_g([sum(w) for w in simulate(T, N, rng)]) for _ in range(300)]
    add("Hidden cycle in draw sums (FFT)", f"g={gsum:.3f}", (np.array(nm) >= gsum).mean() + 1 / 301)
    # 9 sum distribution
    ref = np.array([sum(w) for w in simulate(20000, N, rng)])
    ksum = stats.ks_2samp(sums, ref); add("Sum of the 5 numbers has the chance-expected shape (KS)", f"KS={ksum.statistic:.3f}", ksum.pvalue)
    # 10 consecutive numbers
    cons = np.array([sum(1 for i in range(4) if w[i + 1] - w[i] == 1) for w in W])
    cs = np.array([sum(1 for i in range(4) if w[i + 1] - w[i] == 1) for w in simulate(20000, N, rng)])
    z = (cons.mean() - cs.mean()) / (cs.std() / math.sqrt(T)); add("Back-to-back numbers (e.g. 23,24) per draw", f"avg {cons.mean():.3f} vs chance {cs.mean():.3f}", 2 * stats.norm.sf(abs(z)))
    # 11 weekday effect on sum / on bonus
    wd = np.array([d.weekday() for d in dates]); groups = [sums[wd == k] for k in np.unique(wd) if (wd == k).sum() > 5]
    if len(groups) > 1: add("Weekday affects the sum (Kruskal-Wallis)", f"{len(groups)} weekdays", stats.kruskal(*groups).pvalue)
    return rows

# ------------------------------------------------------------------ B. machine learning
def make_features(X, t):
    """Features for every number, using only draws before t."""
    T0, N = X.shape; H = X[:t]; f = []
    for w in (5, 10, 25, 50, 100): f.append(H[max(0, t - w):].mean(0))
    gap = np.where(H.any(0), np.argmax(H[::-1], axis=0), t + 1).astype(float); f.append(gap); f.append(np.minimum(gap, 60))
    for lag in (1, 2, 3): f.append(X[t - lag].astype(float) if t - lag >= 0 else np.zeros(N))
    n = np.arange(1, N + 1); f += [n / N, (n % 2).astype(float), ((n - 1) // 10).astype(float)]
    ema = np.zeros(N)
    for s in range(max(0, t - 60), t): ema = 0.8 * ema + 0.2 * X[s]
    f.append(ema)
    f.append(np.full(N, X[t - 1].sum() if t > 0 else 0))
    return np.column_stack(f)

def model_zoo():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    return {"Logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=300)),
            "Random forest": lambda: RandomForestClassifier(n_estimators=120, max_depth=5, min_samples_leaf=60, n_jobs=2, random_state=1),
            "Gradient boosting": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=80, min_samples_leaf=80, random_state=1)}

def walk_forward(X, k, start, block=8, models=None):
    """Blocked walk-forward: retrain every `block` draws on everything earlier, predict the next block."""
    models = models or model_zoo(); T, N = X.shape
    cache = {t: make_features(X, t) for t in range(10, T)}
    res = {m: dict(auc=[], rank=[], match=[]) for m in models}
    for b0 in range(start, T, block):
        tr = [t for t in range(10, b0)]
        Xtr = np.vstack([cache[t] for t in tr]); ytr = np.concatenate([X[t] for t in tr])
        for name, mk in models.items():
            clf = mk().fit(Xtr, ytr)
            for t in range(b0, min(b0 + block, T)):
                p = clf.predict_proba(cache[t])[:, 1]; y = X[t]
                res[name]["auc"].append(_auc(p, y))
                order = np.argsort(-p); pos = np.empty(N); pos[order] = np.arange(N)
                res[name]["rank"].append(pos[y == 1].mean() / (N - 1))
                res[name]["match"].append(int(y[order[:k]].sum()))
    return res

def _auc(p, y):
    r = stats.rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

# ------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("game", choices=["powerball", "megamillions"])
    ap.add_argument("--sims", type=int, default=4, help="null simulations of the whole ML pipeline"); ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="output"); a = ap.parse_args()
    cfg = L.GAMES[a.game]; draws = sorted([d for d in L.load_draws(a.game, a.data_dir) if d[0] >= cfg["matrix_start"]], key=lambda x: x[0])
    dates = [d[0] for d in draws]; W = [d[1] for d in draws]; B = [d[2] for d in draws]; N, NB, T = cfg["whites"], cfg["bonus"], len(draws)
    rng = np.random.default_rng(7); t0 = time.time()
    print(f"\n{'=' * 78}\n{cfg['label']}: {T} draws ({dates[0]} to {dates[-1]})\n{'=' * 78}")
    print("\nA. STATISTICAL RANDOMNESS BATTERY  (p < 0.05 means 'unlikely under pure chance')")
    rows = battery(W, B, N, NB, dates, rng)
    for name, s, p, _ in rows: print(f"  {'*' if p < .05 else ' '} p={p:6.3f}  {name}  [{s}]")
    singles = [p for _, _, p, _ in rows]
    print(f"  -> {sum(p < .05 for p in singles)} of {len(singles)} tests below 0.05 (about {.05 * len(singles):.1f} expected by chance); "
          f"{bh(singles)} survive Benjamini-Hochberg correction.")
    print(f"\nB. MACHINE LEARNING, walk-forward (retrained every 8 draws, only past data). Chance: AUC 0.500, rank 50%, matches {25 / N:.3f}")
    start = 60 if T > 150 else 40
    res = walk_forward(to_matrix(W, N), 5, start); n_eval = len(next(iter(res.values()))["auc"])
    ml = []
    for name, r in res.items():
        a_, s_ = np.mean(r["auc"]), np.std(r["auc"]) / math.sqrt(n_eval); z = (a_ - .5) / s_
        ml.append((name, a_, s_, z, np.mean(r["rank"]), np.mean(r["match"])))
        print(f"  {name:22s} AUC {a_:.4f} (+/-{s_:.4f}, z={z:+.1f})  winners rank top {100 * np.mean(r['rank']):4.1f}%  top-5 picks avg matches {np.mean(r['match']):.3f}")
    XB = np.zeros((T, NB), dtype=np.int8)
    for t, b in enumerate(B): XB[t, b - 1] = 1
    rb = walk_forward(XB, 1, start, models={"Gradient boosting": model_zoo()["Gradient boosting"]})["Gradient boosting"]
    print(f"  Bonus ball, gradient boosting: AUC {np.mean(rb['auc']):.4f} (+/-{np.std(rb['auc']) / math.sqrt(len(rb['auc'])):.4f}), top-1 hit {100 * np.mean(rb['match']):.1f}% (chance {100 / NB:.1f}%)")
    print(f"\nC. NULL CHECK: same ML pipeline on {a.sims} simulated random histories of identical size (what 'no pattern' looks like)")
    nulls = {m: [] for m in res}
    for s in range(a.sims):
        S = to_matrix(simulate(T, N, rng), N); r = walk_forward(S, 5, start)
        for m in r: nulls[m].append(np.mean(r[m]["auc"]))
    for m in (res if a.sims else {}):
        print(f"  {m:22s} random-data AUC: mean {np.mean(nulls[m]):.4f}, range {min(nulls[m]):.4f}-{max(nulls[m]):.4f}   | real data {np.mean(res[m]['auc']):.4f}")
    flagged = [r for r in rows if r[2] < .05]
    best = max(ml, key=lambda r: r[3])
    print(f"\nVERDICT: strongest ML signal = {best[0]} at z={best[3]:+.1f}; {bh(singles)} statistical tests survive correction. "
          + ("Something may be worth a closer look." if (best[3] > 3 or bh(singles) > 0) else "No pattern distinguishable from randomness."))
    print(f"(done in {time.time() - t0:.0f}s)")
    os.makedirs(a.out, exist_ok=True)
    json.dump({"game": a.game, "draws": T, "tests": [[n, s, p] for n, s, p, _ in rows], "ml": [list(map(lambda v: v if isinstance(v, str) else float(v), r)) for r in ml],
               "null_auc": {m: [float(x) for x in v] for m, v in nulls.items()}}, open(os.path.join(a.out, f"{a.game}_patterns.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
