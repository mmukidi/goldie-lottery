"""
FUSION PATTERN ENGINE
=====================
Instead of trusting one detector, this builds ~45 independent "experts" that each look for a different kind of
structure, then fuses them four different ways (equal vote, running-z weighting, stacked logistic regression,
stacked gradient boosting). Everything is walk-forward (only past draws), uses the FULL history (Powerball 2015+,
Mega Millions 2017+ for the white balls), and is compared with the same pipeline run on simulated random draws.

Experts (what each one believes):
  frequency windows, overdue gap, echo of draw t-k (k=1..12), exponential memory, Markov transitions
  (number m in last draw -> number n now), analog/kNN (what followed the most similar past stretches),
  shape-state (what follows a high/low-sum or odd/even-heavy draw), residue classes (last digit, mod 7, decade),
  cross-game (the other lottery's latest numbers), bonus-ball link, calendar (date = number), neighbours (+/-1).

  python lotto_fusion.py powerball [--nulls 3]
"""
import sys, os, math, time, json, argparse, bisect, datetime as dt, warnings
import numpy as np
from scipy import stats
warnings.filterwarnings("ignore")

WHITES = {"powerball": 69, "megamillions": 70}

def load_long(game, data_dir):
    import csv
    rows = list(csv.DictReader(open(os.path.join(data_dir, f"long_{game}.csv"))))
    d = [dt.date.fromisoformat(r["date"]) for r in rows]
    W = [[int(r[f"n{i}"]) for i in range(1, 6)] for r in rows]; B = [int(r["bonus"]) for r in rows]
    return d, W, B

def to_X(W, N):
    X = np.zeros((len(W), N), np.float32)
    for t, w in enumerate(W):
        for n in w:
            if n <= N: X[t, n - 1] = 1
    return X

def other_game_matrix(dates, other_dates, other_W, N):
    """For each draw date, the other game's most recent draw strictly before it (1 draw and union of last 3)."""
    Xo = to_X(other_W, 70)[:, :N]
    g1 = np.zeros((len(dates), N), np.float32); g3 = np.zeros_like(g1)
    for t, d in enumerate(dates):
        k = bisect.bisect_left(other_dates, d) - 1
        if k >= 0: g1[t] = Xo[k]; g3[t] = np.clip(Xo[max(0, k - 2):k + 1].sum(0), 0, 1)
    return g1, g3

def build_experts(X, B, dates, g1, g3, t0):
    """Returns names, S (T, E, N): each expert's score for every number at every draw, using only draws before t."""
    T, N = X.shape; n = np.arange(1, N + 1)
    names = []; cols = {}
    def reg(name):
        names.append(name); cols[name] = np.zeros((T, N), np.float32); return cols[name]
    F = {w: reg(f"freq_{w}") for w in (20, 50, 100, 300)}; Fall = reg("freq_all")
    GAP = reg("gap_overdue"); GAPR = reg("gap_vs_own_mean")
    ECHO = {k: reg(f"echo_{k}") for k in range(1, 13)}; E510 = reg("echo_5to10_avg")
    EMA = {h: reg(f"memory_hl{h}") for h in (3, 10, 30)}
    MK1 = reg("markov_lag1"); MK2 = reg("markov_lag2")
    KN3 = reg("analog_knn_win3"); KN1 = reg("analog_knn_win1")
    SUMS = reg("state_after_sum_level"); ODDS = reg("state_after_odd_count")
    R10 = reg("class_last_digit"); R7 = reg("class_mod7"); RDEC = reg("class_decade")
    XG1 = reg("cross_game_last"); XG3 = reg("cross_game_last3")
    BL1 = reg("bonus_link_exact"); BL2 = reg("bonus_link_lag2"); BLN = reg("bonus_link_near")
    CD = reg("calendar_day"); CM = reg("calendar_month"); CA = reg("calendar_any")
    NB1 = reg("neighbour_of_last"); NB510 = reg("neighbour_5to10")
    cs = np.vstack([np.zeros((1, N)), np.cumsum(X, 0)]); last = np.full(N, -1.0)
    ema = {h: np.zeros(N) for h in EMA}; alpha = {h: 1 - 0.5 ** (1 / h) for h in EMA}
    k = 25.0; p0 = 5.0 / N
    C1 = np.zeros((N, N)); c1 = np.zeros(N); C2 = np.zeros((N, N)); c2 = np.zeros(N)
    sig = math.sqrt(5 * (N * N - 1) / 12 * (N - 5) / (N - 1)); mu = 5 * (N + 1) / 2
    def sum_level(row): s = (row * n).sum(); return 0 if s < mu - .43 * sig else (2 if s > mu + .43 * sig else 1)
    def odd_level(row): o = int((row * (n % 2)).sum()); return 0 if o <= 1 else (3 if o >= 4 else o - 1)
    sc = np.zeros((3, N)); st = np.zeros(3); oc = np.zeros((4, N)); ot = np.zeros(4)
    M3 = np.zeros((T, N), np.float32)
    for t in range(T): M3[t] = np.clip(X[max(0, t - 3):t].sum(0), 0, 1)
    G3 = M3 @ M3.T; G1 = np.zeros((T, T), np.float32)
    X1 = np.vstack([np.zeros((1, N), np.float32), X[:-1]]); G1 = X1 @ X1.T
    cls = {"d": (n % 10), "7": (n % 7), "dec": ((n - 1) // 10)}
    for t in range(T):
        if t >= t0:
            for w, a in F.items(): a[t] = (cs[t] - cs[max(0, t - w)]) / max(1, min(w, t))
            Fall[t] = cs[t] / max(1, t)
            gap = np.where(last >= 0, t - last, t + 1.0); GAP[t] = gap
            meang = t / np.maximum(1, cs[t]); GAPR[t] = gap / meang
            for kk, a in ECHO.items(): a[t] = X[t - kk] if t - kk >= 0 else 0
            E510[t] = X[max(0, t - 10):max(0, t - 4)].mean(0) if t > 10 else 0
            for h, a in EMA.items(): a[t] = ema[h]
            prev = X[t - 1]; prev2 = X[t - 2]
            MK1[t] = prev @ ((C1 + p0 * k) / (c1[:, None] + k)); MK2[t] = prev2 @ ((C2 + p0 * k) / (c2[:, None] + k))
            for G, out, KK in ((G3, KN3, 15), (G1, KN1, 15)):
                s = G[t, 4:t]; idx = np.argsort(-s, kind="stable")[:KK] + 4
                out[t] = X[idx].mean(0)
            a_ = sum_level(prev); SUMS[t] = (sc[a_] + p0 * k) / (st[a_] + k)
            b_ = odd_level(prev); ODDS[t] = (oc[b_] + p0 * k) / (ot[b_] + k)
            for out, cl in ((R10, cls["d"]), (R7, cls["7"]), (RDEC, cls["dec"])):
                rate = np.zeros(N)
                for c in np.unique(cl): m = cl == c; rate[m] = cs[t][m].sum() / (m.sum() * max(1, t))
                out[t] = rate
            XG1[t] = g1[t]; XG3[t] = g3[t]
            BL1[t] = (n == B[t - 1]); BL2[t] = (n == B[t - 2]); BLN[t] = (np.abs(n - B[t - 1]) <= 2)
            d = dates[t]; CD[t] = (n == d.day); CM[t] = (n == d.month); CA[t] = ((n == d.day) | (n == d.month) | (n == d.year % 100))
            nb = np.zeros(N); pr = prev.astype(bool); nb[1:] += pr[:-1]; nb[:-1] += pr[1:]; NB1[t] = np.clip(nb, 0, 1)
            u = np.clip(X[max(0, t - 10):max(0, t - 4)].sum(0), 0, 1).astype(bool) if t > 10 else np.zeros(N, bool)
            nb2 = np.zeros(N); nb2[1:] += u[:-1]; nb2[:-1] += u[1:]; NB510[t] = np.clip(nb2, 0, 1)
        # update stats with draw t
        row = X[t]; last[row > 0] = t
        for h in EMA: ema[h] = ema[h] * (1 - alpha[h]) + alpha[h] * row
        if t >= 1: C1 += np.outer(X[t - 1], row); c1 += X[t - 1]
        if t >= 2: C2 += np.outer(X[t - 2], row); c2 += X[t - 2]
        if t >= 1: a_ = sum_level(X[t - 1]); sc[a_] += row; st[a_] += 1; b_ = odd_level(X[t - 1]); oc[b_] += row; ot[b_] += 1
    S = np.stack([cols[nm] for nm in names], axis=1)
    return names, S

def zscore(S):
    m = S.mean(-1, keepdims=True); sd = S.std(-1, keepdims=True)
    return np.where(sd > 1e-9, (S - m) / np.where(sd > 1e-9, sd, 1), 0).astype(np.float32)

def auc_rows(P, Y):
    """P, Y: (T, N). Per-draw AUC with tie handling."""
    R = stats.rankdata(P, axis=1); n1 = Y.sum(1); n0 = Y.shape[1] - n1
    return ((R * Y).sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)

def matches_top5(P, Y):
    idx = np.argsort(-P, axis=1, kind="stable")[:, :5]; return np.take_along_axis(Y, idx, 1).sum(1)

AUC_SD = 0.135   # per-draw AUC sd under pure chance: sqrt((N+1)/(12*5*(N-5)))

def fusion_models(Z, X, t_eval, t0):
    """Walk-forward fused predictions for draws t_eval..T-1. Z: (T, E, N) z-scored experts."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    T, E, N = Z.shape; out = {}
    A = np.full((T, E), np.nan)
    for e in range(E): A[:, e] = auc_rows(Z[:, e, :], X)
    # 1 equal vote
    out["FUSE equal vote"] = Z[t_eval:].mean(1)
    # 2 running-z weighting (signed, soft-threshold at 1 sigma): only trust experts with a track record
    P = np.zeros((T - t_eval, N), np.float32)
    for i, t in enumerate(range(t_eval, T)):
        hist = A[t0:t]; cnt = len(hist)
        zrun = (np.nanmean(hist, 0) - .5) / (AUC_SD / math.sqrt(cnt))
        w = np.sign(zrun) * np.maximum(0, np.abs(zrun) - 1.0)
        P[i] = np.tensordot(w, Z[t], axes=(0, 0))
    out["FUSE running-z weights"] = P
    # 3/4 stacking models retrained every 50 draws on all earlier draws
    for name, mk, step in (("FUSE stacked logistic", lambda: LogisticRegression(C=0.003, max_iter=200), 50),
                           ("FUSE stacked boosting", lambda: HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=60, min_samples_leaf=300, l2_regularization=5, random_state=1), 100)):
        P = np.zeros((T - t_eval, N), np.float32)
        for b0 in range(t_eval, T, step):
            tr = np.arange(t0, b0)
            Xtr = Z[tr].transpose(0, 2, 1).reshape(-1, E); ytr = X[tr].reshape(-1)
            clf = mk().fit(Xtr, ytr)
            for t in range(b0, min(T, b0 + step)):
                P[t - t_eval] = clf.predict_proba(Z[t].T)[:, 1]
        out[name] = P
    return out, A

def run_pipeline(X, B, dates, g1, g3, t0=60, t_eval=400, verbose=False):
    T, N = X.shape
    names, S = build_experts(X, B, dates, g1, g3, t0)
    Z = zscore(S); res = {}
    A = np.full((T, len(names)), np.nan)
    for e in range(len(names)): A[t0:, e] = auc_rows(Z[t0:, e, :], X[t0:])
    # expert table on the evaluation region, split into discovery / holdout halves
    mid = (t_eval + T) // 2
    rows = []
    for e, nm in enumerate(names):
        a = A[t_eval:, e]; z = (a.mean() - .5) / (AUC_SD / math.sqrt(len(a)))
        za = (A[t_eval:mid, e].mean() - .5) / (AUC_SD / math.sqrt(mid - t_eval)); zb = (A[mid:, e].mean() - .5) / (AUC_SD / math.sqrt(T - mid))
        rows.append((nm, float(a.mean()), float(z), float(za), float(zb)))
    fused, _ = fusion_models(Z, X, t_eval, t0)
    frows = []
    for nm, P in fused.items():
        a = auc_rows(P, X[t_eval:]); m = matches_top5(P, X[t_eval:])
        frows.append((nm, float(a.mean()), float((a.mean() - .5) / (AUC_SD / math.sqrt(len(a)))), float(m.mean()), float(((a - .5) > 0).mean())))
    return rows, frows, mid

def bh_count(z, q=.05):
    p = np.sort(2 * stats.norm.sf(np.abs(z))); m = len(p); ok = np.where(p <= q * np.arange(1, m + 1) / m)[0]
    return int(ok.max() + 1) if len(ok) else 0

def random_history(T, N, rng, bonus_n):
    W = [sorted(rng.choice(N, 5, replace=False) + 1) for _ in range(T)]
    return W, [int(rng.integers(1, bonus_n + 1)) for _ in range(T)]

def sync_long(game, data_dir):
    """Merge the auto-updated data/<game>_draws.csv into data/long_<game>.csv. Returns number of draws added."""
    import csv
    p = os.path.join(data_dir, f"long_{game}.csv"); cur = {r["date"]: r for r in csv.DictReader(open(p))}
    n0 = len(cur); src = os.path.join(data_dir, f"{game}_draws.csv")
    if os.path.exists(src):
        for r in csv.DictReader(open(src)):
            cur.setdefault(r["date"], {"date": r["date"], **{f"n{i}": r[f"n{i}"] for i in range(1, 6)}, "bonus": r["bonus"]})
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["date", "n1", "n2", "n3", "n4", "n5", "bonus"])
        for d in sorted(cur): w.writerow([d] + [cur[d][f"n{i}"] for i in range(1, 6)] + [cur[d]["bonus"]])
    return len(cur) - n0

def next_picks(game, data_dir, target=None, look=400):
    """Fused picks for the next draw (equal vote of all 42 experts, and the track-record-weighted version) + evidence line."""
    import lotto_predictor as L
    N = WHITES[game]; other = "megamillions" if game == "powerball" else "powerball"
    dates, W, B = load_long(game, data_dir); od, oW, _ = load_long(other, data_dir)
    target = target or L.next_draw_date(game, L.default_start())
    d2, W2, B2 = dates + [target], W + [[1, 2, 3, 4, 5]], B + [B[-1]]
    X = to_X(W2, N); g1, g3 = other_game_matrix(d2, od, oW, N); T = len(X)
    names, S = build_experts(X, B2, d2, g1, g3, 60); Z = zscore(S)
    Xr = X[:-1]; A = np.array([auc_rows(Z[60:T - 1, e, :], Xr[60:]) for e in range(len(names))]).T
    eq = Z[T - 1].mean(0); look = min(look, T - 61)
    z_eq = (auc_rows(Z[T - 1 - look:T - 1].mean(1), Xr[-look:]).mean() - .5) / (AUC_SD / math.sqrt(look))
    zrun = (np.nanmean(A, 0) - .5) / (AUC_SD / math.sqrt(len(A))); w = np.sign(zrun) * np.maximum(0, np.abs(zrun) - 1.0)
    wz = np.tensordot(w, Z[T - 1], axes=(0, 0))
    top = lambda v: sorted(int(i) + 1 for i in np.argsort(-v, kind="stable")[:5])
    return target, top(eq), top(wz), float(z_eq), int((w != 0).sum())

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("game", choices=list(WHITES)); ap.add_argument("--nulls", type=int, default=2)
    ap.add_argument("--sync-long", action="store_true", help="merge newly downloaded draws into the long-history file")
    ap.add_argument("--picks", action="store_true", help="print fused picks + evidence for the next draw and exit")
    ap.add_argument("--data-dir", default="data"); ap.add_argument("--out", default="output"); a = ap.parse_args()
    if a.sync_long: print(f"[{a.game}] long history: added {sync_long(a.game, a.data_dir)} new draw(s)")
    if a.picks:
        tgt, eq, wz, z, nw = next_picks(a.game, a.data_dir)
        print(f"FUSION {a.game} for {tgt}: equal-vote {' '.join(f'{x:02d}' for x in eq)} | track-record-weighted {' '.join(f'{x:02d}' for x in wz)} "
              f"| evidence: equal-vote skill over last 400 draws z={z:+.1f} (needs z>3.5 to mean anything), {nw} experts currently trusted")
        return
    N = WHITES[a.game]; other = "megamillions" if a.game == "powerball" else "powerball"
    dates, W, B = load_long(a.game, a.data_dir); od, oW, _ = load_long(other, a.data_dir)
    X = to_X(W, N); g1, g3 = other_game_matrix(dates, od, oW, N); T = len(dates)
    t_eval = 400 if T > 700 else 250
    print(f"\n{'=' * 80}\n{a.game}: {T} draws, {dates[0]} to {dates[-1]}; experts learn from draw 60+, fused models are scored on the last {T - t_eval} draws\n{'=' * 80}")
    t_start = time.time()
    rows, frows, mid = run_pipeline(X, B, dates, g1, g3, t_eval=t_eval)
    zs = np.array([r[2] for r in rows]); ex = 25 / N
    print(f"\n1. EXPERTS: each one's ability to rank the winning numbers (AUC 0.500 = chance). z = distance from chance in standard errors.")
    print(f"   {'expert':28s} {'AUC':>7s} {'z':>6s}   {'z 1st half':>10s} {'z 2nd half':>10s}")
    for r in sorted(rows, key=lambda r: -abs(r[2]))[:10]:
        print(f"   {r[0]:28s} {r[1]:7.4f} {r[2]:+6.1f}   {r[3]:+10.1f} {r[4]:+10.1f}")
    nsig = int((np.abs(zs) > 1.96).sum())
    print(f"   -> {nsig} of {len(rows)} experts beyond 1.96 sigma (chance: ~{.05 * len(rows):.1f});  {bh_count(zs)} survive multiple-testing correction")
    sel = [r for r in rows if r[3] > 2.0]
    if sel:
        print(f"   Discovery/holdout test: {len(sel)} experts looked good (z>2) in the first half; their average z in the unseen second half = {np.mean([r[4] for r in sel]):+.2f} (0 = no real signal)")
    else:
        print("   Discovery/holdout test: no expert reached z>2 in the first half, so nothing to carry forward.")
    print(f"\n2. FUSED MODELS (all experts combined, walk-forward): chance AUC 0.500, top-5 picks chance matches {ex:.3f}")
    for f in frows: print(f"   {f[0]:26s} AUC {f[1]:.4f} (z={f[2]:+.1f})   top-5 picks avg matches {f[3]:.3f}")
    nullres = []
    if a.nulls:
        print(f"\n3. NULL CHECK: identical pipeline on {a.nulls} simulated random histories (what 'no pattern at all' looks like)")
        rng = np.random.default_rng(11)
        for s in range(a.nulls):
            Wn, Bn = random_history(T, N, rng, 26 if a.game == "powerball" else 24)
            Xn = to_X(Wn, N); Wo, _ = random_history(len(od), N, rng, 24); g1n, g3n = other_game_matrix(dates, od, Wo, N)
            rn, fn, _ = run_pipeline(Xn, Bn, dates, g1n, g3n, t_eval=t_eval)
            zn = np.array([r[2] for r in rn]); nullres.append((float(np.abs(zn).max()), [f[1] for f in fn], [f[2] for f in fn]))
            print(f"   random history {s + 1}: strongest single expert |z|={np.abs(zn).max():.1f}; fused AUCs " + ", ".join(f"{f[1]:.4f}" for f in fn))
    best_f = max(frows, key=lambda f: f[2]); best_e = max(rows, key=lambda r: abs(r[2]))
    print(f"\nSUMMARY: strongest expert = {best_e[0]} (z={best_e[2]:+.1f}); strongest fusion = {best_f[0]} (AUC {best_f[1]:.4f}, z={best_f[2]:+.1f}); "
          f"{bh_count(zs)} experts survive correction.  ({time.time() - t_start:.0f}s)")
    os.makedirs(a.out, exist_ok=True)
    json.dump({"game": a.game, "draws": T, "experts": rows, "fused": frows, "nulls": nullres}, open(os.path.join(a.out, f"{a.game}_fusion.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
