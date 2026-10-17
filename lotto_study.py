"""
Train/test study: learn from one year, test on the next (default: train on 2025, test on 2026).

Questions it answers, per game:
  1. If the model is trained only on the training year and FROZEN, how close are its probabilities to what
     happened in the test year? (rank of winning numbers, log-likelihood vs uniform, match counts)
  2. If the model keeps updating (rolling 365-day window), does it do better?
  3. Can we tweak the weights to improve it? Settings are tuned on the training year only, then judged on the
     test year they never saw. If in-sample gains vanish out-of-sample, they were luck.
Writes <game>_study.xlsx (all draws + results) and prints a summary.
"""
import datetime as dt, math, os, random, statistics, itertools
from collections import Counter
import lotto_predictor as L

NEG = [(-1, 0, 0), (0, -1, 0), (0, 0, -1)]            # "contrarian" variants, to test inverse signals too
TAUS = [0.25, 0.5, 1.0, 2.0, 4.0, None]               # temperature for turning scores into probabilities (None = uniform)

def param_grid():
    seen = set()
    for wts in list(L.GRID_WEIGHTS) + NEG:
        for half in L.GRID_HALF:
            p = (*wts, half)
            if p not in seen: seen.add(p); yield p

def mean_se(vals):
    n = len(vals)
    if n < 2: return (vals[0] if vals else float("nan")), float("nan")
    return statistics.mean(vals), statistics.pstdev(vals) / math.sqrt(n)

def llr(scores, winners, tau, k):
    """Average log(model probability / uniform probability) over the winning numbers (0 = same as uniform)."""
    if tau is None: return 0.0
    m = max(scores.values())
    ex = {n: math.exp((s - m) / tau) for n, s in scores.items()}
    tot = sum(ex.values()); u = k / len(scores)
    return statistics.mean(math.log(k * ex[w] / tot / u) for w in winners)

def metrics(params, tests, cfg):
    """Per-draw rank percentiles and probability scores for a model using `params`, history per test draw."""
    wf, wd, wh, half = params
    out = dict(rw=[], rb=[], lw={t: [] for t in TAUS}, lb={t: [] for t in TAUS})
    for H, w, b in tests:
        sw = L.score_pool([x for x, _ in H], cfg["whites"], wf, wd, wh, half)
        sb = L.score_pool([[bb] for _, bb in H], cfg["bonus"], wf, wd, wh, half)
        out["rw"].append(L.rank_pct(sw, w)); out["rb"].append(L.rank_pct(sb, [b]))
        for t in TAUS:
            out["lw"][t].append(llr(sw, w, t, 5)); out["lb"][t].append(llr(sb, [b], t, 1))
    return out

def frozen_metrics(params, Htrain, test_draws, cfg):
    wf, wd, wh, half = params
    sw = L.score_pool([x for x, _ in Htrain], cfg["whites"], wf, wd, wh, half)
    sb = L.score_pool([[bb] for _, bb in Htrain], cfg["bonus"], wf, wd, wh, half)
    out = dict(rw=[], rb=[], lw={t: [] for t in TAUS}, lb={t: [] for t in TAUS})
    for d, w, b in test_draws:
        out["rw"].append(L.rank_pct(sw, w)); out["rb"].append(L.rank_pct(sb, [b]))
        for t in TAUS:
            out["lw"][t].append(llr(sw, w, t, 5)); out["lb"][t].append(llr(sb, [b], t, 1))
    return out

def spearman(a, b):
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i]); r = [0] * len(x)
        for pos, i in enumerate(order): r[i] = pos
        return r
    ra, rb = ranks(a), ranks(b); n = len(a)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float("nan")

def hypergeom(N, k):
    return math.comb(5, k) * math.comb(N - 5, 5 - k) / math.comb(N, 5)

def run(game, draws, days, train_year, test_year, out_dir, data_dir):
    cfg = GAMES = L.GAMES[game]
    label = cfg["label"]; N = cfg["whites"]
    train = [x for x in draws if x[0].year == train_year and x[0] >= cfg["matrix_start"]]
    test = sorted([x for x in draws if x[0].year == test_year], key=lambda x: x[0])
    if len(train) < 40 or len(test) < 20:
        print(f"[{label}] not enough data (train {len(train)}, test {len(test)} draws)"); return
    print(f"\n{'=' * 72}\n{label}: train on {train_year} ({len(train)} draws) -> test on {test_year} ({len(test)} draws)\n{'=' * 72}")
    random.seed(2026)

    # ---------- test sets
    tests_train = L.build_tests(train, cfg, dt.date(test_year, 1, 1), days, window=len(train))   # walk-forward inside train year
    tests_test = L.build_tests(draws, cfg, dt.date(test_year + 1, 1, 1), days, window=len(test))  # walk-forward through test year
    Htrain, _ = L.history_for(draws, dt.date(test_year, 1, 1), 366, cfg)

    # ---------- 1. tuning grid: in-sample (train year) vs out-of-sample (test year)
    rows = []
    for p in param_grid():
        mi = metrics(p, tests_train, cfg); mo = metrics(p, tests_test, cfg)
        rows.append(dict(p=p, in_w=statistics.mean(mi["rw"]), oos_w=statistics.mean(mo["rw"]),
                         in_b=statistics.mean(mi["rb"]), oos_b=statistics.mean(mo["rb"]),
                         oos_w_se=mean_se(mo["rw"])[1], oos_llr=statistics.mean(mo["lw"][1.0]),
                         in_llr=statistics.mean(mi["lw"][1.0])))
    best = min(rows, key=lambda r: r["in_w"])
    default = next(r for r in rows if r["p"] == L.DEFAULT_PARAMS)
    rho = spearman([r["in_w"] for r in rows], [r["oos_w"] for r in rows])
    n_in, n_out = len(tests_train), len(tests_test)
    print(f"\n[1] Tuning on {train_year} only ({n_in} walk-forward tests), then judged on {test_year} ({n_out} draws)")
    print("    Metric = average rank of the 5 winning numbers as % of the pool (50% = random, lower is better)")
    print(f"    Default formula:            {train_year} {100 * default['in_w']:.1f}%   ->  {test_year} {100 * default['oos_w']:.1f}%")
    print(f"    Best-looking on {train_year} {tuple(round(x, 2) for x in best['p'])}: "
          f"{train_year} {100 * best['in_w']:.1f}%   ->  {test_year} {100 * best['oos_w']:.1f}% (+/-{100 * best['oos_w_se']:.1f})")
    print(f"    Rank correlation of settings between years: {rho:+.2f} (0 = {train_year} performance says nothing about {test_year})")
    better = sum(1 for r in rows if r["oos_w"] < 0.5)
    print(f"    Settings that beat 50% in {test_year}: {better} of {len(rows)} (about half expected from luck alone)")

    # ---------- 2. frozen model: trained on train year, never updated
    fz = {}
    for name, p in [("Default formula", L.DEFAULT_PARAMS), ("Tuned on train year", best["p"]),
                    ("Frequency only", (1, 0, 0, 20)), ("Due only", (0, 1, 0, 20)), ("Hot streak only", (0, 0, 1, 20))]:
        fz[name] = frozen_metrics(p, Htrain, test, cfg)
    print(f"\n[2] Frozen model: trained on {train_year}, never updated, tested on all {len(test)} {test_year} draws")
    print("    Model                      winners rank (whites)   winners rank (bonus)   probability skill* (whites)")
    for name, m in fz.items():
        rw, se = mean_se(m["rw"]); rb, _ = mean_se(m["rb"]); ll, lse = mean_se(m["lw"][1.0])
        print(f"    {name:26s} top {100 * rw:5.1f}% (+/-{100 * se:.1f})     top {100 * statistics.mean(m['rb']):5.1f}%            {ll:+.3f} (+/-{lse:.3f})")
    print("    *log of (model probability / uniform probability) for the winning numbers; 0 = same as uniform, negative = worse.")
    print("     For truly random draws no model can score above 0 on average.")
    print("    Temperature check (default formula): probability skill by how strongly it commits to its favourites:")
    print("      " + "  ".join(f"tau={'inf' if t is None else t}: {statistics.mean(fz['Default formula']['lw'][t]):+.3f}" for t in TAUS))

    # ---------- 3. rolling model: keeps learning from each new draw
    ro = {"Default formula": metrics(L.DEFAULT_PARAMS, tests_test, cfg), "Tuned on train year": metrics(best["p"], tests_test, cfg)}
    print(f"\n[3] Rolling model: re-learns from the previous 365 days before every {test_year} draw")
    for name, m in ro.items():
        rw, se = mean_se(m["rw"]); ll, lse = mean_se(m["lw"][1.0])
        print(f"    {name:26s} winners rank top {100 * rw:5.1f}% (+/-{100 * se:.1f}); probability skill {ll:+.3f} (+/-{lse:.3f})")

    # ---------- 4. actual guesses: frozen picks and rolling picks, matches vs random
    strategies = list(L.STRATEGIES)
    frozen_picks = {name: f(Htrain, cfg) for name, f in strategies}
    frozen_m = {name: [len(set(c) & set(w)) for _, w, _ in test] for name, (c, _) in frozen_picks.items()}
    frozen_b = {name: [int(pb == b) for _, _, b in test] for name, (_, pb) in frozen_picks.items()}
    rolling_picks, rolling_m, rolling_b = {n: [] for n, _ in strategies}, {n: [] for n, _ in strategies}, {n: [] for n, _ in strategies}
    for (H, w, b), (d, _, _) in zip(tests_test, test[-len(tests_test):]):
        for name, f in strategies:
            c, pb = f(H, cfg)
            rolling_picks[name].append((c, pb)); rolling_m[name].append(len(set(c) & set(w))); rolling_b[name].append(int(pb == b))
    exp_m = 25 / N; n = len(test)
    sims = sorted(statistics.mean(len(set(random.sample(range(1, N + 1), 5)) & set(w)) for _, w, _ in test) for _ in range(3000))
    pct = lambda v: 100 * sum(s <= v for s in sims) / len(sims)
    print(f"\n[4] Actual guesses, {test_year}: average white-ball matches per draw (random expectation {exp_m:.3f}; "
          f"random luck ranges {sims[int(.025 * len(sims))]:.2f}-{sims[int(.975 * len(sims))]:.2f})")
    print(f"    {'Strategy':34s} frozen on {train_year}   rolling")
    for name, _ in strategies:
        fm, rm = statistics.mean(frozen_m[name]), statistics.mean(rolling_m[name])
        print(f"    {name:34s} {fm:8.3f} (p{pct(fm):.0f})      {rm:8.3f} (p{pct(statistics.mean(rolling_m[name])):.0f})")
    print("    (pNN = percentile among 3,000 simulated years of pure random picks; ~50 is typical, >97 would be notable)")
    print(f"    Bonus-ball hit rate: " + ", ".join(f"{nm.split(' (')[0]} {100 * statistics.mean(rolling_b[nm]):.1f}%" for nm, _ in strategies)
          + f"  (random {100 / cfg['bonus']:.1f}%)")

    # ---------- 5. match distribution vs theory (frozen default formula)
    name0 = strategies[0][0]; obs = Counter(frozen_m[name0])
    print(f"\n[5] Match distribution, frozen formula pick vs theory (hypergeometric):")
    for k in range(4):
        print(f"    {k} matches: observed {obs.get(k, 0):3d}   expected {n * hypergeom(N, k):6.1f}")
    print(f"    4+ matches: observed {sum(v for kk, v in obs.items() if kk >= 4)}   expected {n * sum(hypergeom(N, kk) for kk in (4, 5)):.2f}")

    # ---------- verdict
    sig = (0.5 - best["oos_w"]) / best["oos_w_se"] if best["oos_w_se"] else 0
    print(f"\nVerdict: tuned model out-of-sample = {100 * best['oos_w']:.1f}% (z={sig:+.1f} vs random). "
          f"Default frozen = {100 * statistics.mean(fz['Default formula']['rw']):.1f}%. "
          f"Any gain on {train_year} {'persisted' if sig > 2.5 else 'did not persist'} on {test_year}.")

    write_study_workbook(game, cfg, draws, train, test, tests_test, rows, best, default, rho, fz, ro,
                         frozen_picks, frozen_m, frozen_b, rolling_picks, rolling_m, rolling_b, strategies,
                         sims, train_year, test_year, out_dir, data_dir)

def write_study_workbook(game, cfg, draws, train, test, tests_test, rows, best, default, rho, fz, ro, frozen_picks,
                         frozen_m, frozen_b, rolling_picks, rolling_m, rolling_b, strategies, sims, ty, sy, out_dir, data_dir):
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter as CL
    except ImportError:
        print("  (openpyxl not installed - skipping Excel file. Run: pip install openpyxl)"); return
    F = "Arial"; label = cfg["label"]; N = cfg["whites"]; bn = cfg["bonus_name"]
    hf = Font(name=F, bold=True, color="FFFFFF"); hfill = PatternFill("solid", fgColor="1F3864")
    body = Font(name=F); bold = Font(name=F, bold=True); blue = Font(name=F, color="0000FF")
    ctr = Alignment(horizontal="center", vertical="center", wrap_text=True)
    bfill = PatternFill("solid", fgColor={"powerball": "F8CBAD", "megamillions": "FFE699"}[game])
    wb = Workbook(); wb.remove(wb.active)

    def header(ws, row, heads):
        for c, h in enumerate(heads, start=1):
            cell = ws.cell(row, c, h); cell.font = hf; cell.fill = hfill; cell.alignment = ctr

    # ---- Draws: every draw, with year and format
    wd = wb.create_sheet("All Draws")
    old_path = os.path.join(data_dir, "megamillions_old_format_2025.csv")
    allrows = [(d, w, b, "Current") for d, w, b in draws]
    if game == "megamillions" and os.path.exists(old_path):
        import csv
        for r in csv.DictReader(open(old_path)):
            allrows.append((dt.date.fromisoformat(r["date"]), [int(r[f"n{i}"]) for i in range(1, 6)], int(r["bonus"]), "Old (Mega Ball 1-25)"))
    allrows.sort(key=lambda x: x[0], reverse=True)
    header(wd, 1, ["Draw Date", "Year", "Day", "Ball 1", "Ball 2", "Ball 3", "Ball 4", "Ball 5", bn, "Sum of 5", "Format"])
    for i, (d, w, b, fmt) in enumerate(allrows, start=2):
        wd.cell(i, 1, d).number_format = "yyyy-mm-dd"; wd.cell(i, 2, f"=YEAR(A{i})"); wd.cell(i, 3, f'=TEXT(A{i},"ddd")')
        for c, v in enumerate(sorted(w), start=4): wd.cell(i, c, v)
        wd.cell(i, 9, b); wd.cell(i, 10, f"=SUM(D{i}:H{i})"); wd.cell(i, 11, fmt)
        for c in range(1, 12):
            wd.cell(i, c).font = body
            if c != 1: wd.cell(i, c).alignment = Alignment(horizontal="center")
        wd.cell(i, 9).fill = bfill; wd.cell(i, 9).font = bold
    last_all = len(allrows) + 1
    wd.freeze_panes = "A2"; wd.auto_filter.ref = f"A1:K{last_all}"
    for c, w_ in zip("ABCDEFGHIJK", [13, 7, 7, 8, 8, 8, 8, 8, 12, 10, 22]): wd.column_dimensions[c].width = w_

    # ---- Test detail
    td = wb.create_sheet(f"Test {sy} Detail")
    first = strategies[0][0]
    heads = ["Draw Date", "Win 1", "Win 2", "Win 3", "Win 4", "Win 5", f"Win {bn}",
             f"Frozen-{ty} pick 1", "2", "3", "4", "5", f"{bn}", "White matches", "Bonus hit",
             "Rolling pick 1", "2", "3", "4", "5", f"{bn}", "White matches", "Bonus hit"]
    td["A1"] = (f"{label}: {first} guess on every {sy} draw. 'Frozen' = trained on {ty} only and never updated "
                f"(same numbers every draw). 'Rolling' = re-trained on the prior 365 days before each draw. Matches are formulas.")
    td["A1"].font = Font(name=F, italic=True); td.merge_cells("A1:W1")
    header(td, 3, heads); td.row_dimensions[3].height = 34
    fc, fpb = frozen_picks[first]
    offset = len(test) - len(tests_test)
    for i, (d, w, b) in enumerate(test):
        r = 4 + i
        td.cell(r, 1, d).number_format = "yyyy-mm-dd"
        for c, v in enumerate(sorted(w), start=2): td.cell(r, c, v)
        td.cell(r, 7, b)
        for c, v in enumerate(fc, start=8): td.cell(r, c, v)
        td.cell(r, 13, fpb)
        td.cell(r, 14, f"=SUMPRODUCT(COUNTIF(B{r}:F{r},H{r}:L{r}))"); td.cell(r, 15, f"=IF(M{r}=G{r},1,0)")
        j = i - offset
        if j >= 0:
            rc, rpb = rolling_picks[first][j]
            for c, v in enumerate(rc, start=16): td.cell(r, c, v)
            td.cell(r, 21, rpb)
            td.cell(r, 22, f"=SUMPRODUCT(COUNTIF(B{r}:F{r},P{r}:T{r}))"); td.cell(r, 23, f"=IF(U{r}=G{r},1,0)")
        for c in range(1, 24):
            td.cell(r, c).font = blue if (8 <= c <= 13 or 16 <= c <= 21) else body
            td.cell(r, c).alignment = ctr
        td.cell(r, 7).fill = bfill; td.cell(r, 13).fill = bfill; td.cell(r, 21).fill = bfill
    last_t = 3 + len(test)
    td.freeze_panes = "B4"
    for c in range(1, 24): td.column_dimensions[CL(c)].width = 11
    td.column_dimensions["A"].width = 13

    # ---- Match distribution: observed (formula) vs expected (formula)
    md = wb.create_sheet("Match Distribution")
    md["A1"] = f"How often the {first} pick matched k white balls in {sy}, vs what pure chance predicts"; md["A1"].font = Font(name=F, bold=True, size=12)
    header(md, 3, ["Matches", "Frozen: observed draws", "Frozen: expected draws", "Rolling: observed draws", "Rolling: expected draws", "Chance of this many matches (one draw)"])
    tq = f"'{td.title}'"
    for k in range(6):
        r = 4 + k
        md.cell(r, 1, k)
        md.cell(r, 2, f"=COUNTIF({tq}!$N$4:$N${last_t},A{r})")
        md.cell(r, 3, f"=COUNT({tq}!$N$4:$N${last_t})*F{r}")
        md.cell(r, 4, f"=COUNTIF({tq}!$V$4:$V${last_t},A{r})")
        md.cell(r, 5, f"=COUNT({tq}!$V$4:$V${last_t})*F{r}")
        md.cell(r, 6, f"=COMBIN(5,A{r})*COMBIN({N - 5},5-A{r})/COMBIN({N},5)")
        for c in range(1, 7): md.cell(r, c).font = body; md.cell(r, c).alignment = ctr
        for c in (3, 5): md.cell(r, c).number_format = "0.00"
        md.cell(r, 6).number_format = "0.000000"
    md.cell(11, 1, "Average matches").font = bold
    md.cell(11, 2, f"=AVERAGE({tq}!$N$4:$N${last_t})"); md.cell(11, 4, f"=AVERAGE({tq}!$V$4:$V${last_t})")
    md.cell(11, 3, f"=25/{N}"); md.cell(11, 5, f"=25/{N}")
    for c in (2, 3, 4, 5): md.cell(11, c).number_format = "0.000"; md.cell(11, c).font = bold; md.cell(11, c).alignment = ctr
    md.cell(12, 1, f"Bonus ball hit rate").font = bold
    md.cell(12, 2, f"=AVERAGE({tq}!$O$4:$O${last_t})"); md.cell(12, 4, f"=AVERAGE({tq}!$W$4:$W${last_t})")
    md.cell(12, 3, f"=1/{cfg['bonus']}"); md.cell(12, 5, f"=1/{cfg['bonus']}")
    for c in (2, 3, 4, 5): md.cell(12, c).number_format = "0.0%"; md.cell(12, c).font = bold; md.cell(12, c).alignment = ctr
    md.cell(14, 1, "Observed should track expected within normal noise. Large gaps in the 3+ rows would be the only sign of real skill.").font = Font(name=F, italic=True)
    for c, w_ in zip("ABCDEF", [18, 22, 22, 22, 22, 28]): md.column_dimensions[c].width = w_
    md.row_dimensions[3].height = 32

    # ---- Strategy comparison (all 5 strategies)
    sc = wb.create_sheet("Strategy Results")
    sc["A1"] = f"Average white-ball matches per draw in {sy} (random expectation {25 / N:.3f}) and bonus hit rate"; sc["A1"].font = Font(name=F, bold=True, size=12)
    header(sc, 3, ["Strategy", f"Frozen on {ty}: avg matches", "Frozen: bonus hit rate", "Rolling: avg matches", "Rolling: bonus hit rate",
                   "Frozen picks (white)", f"Frozen {bn}"])
    for i, (name, _) in enumerate(strategies):
        r = 4 + i
        sc.cell(r, 1, name); sc.cell(r, 2, round(statistics.mean(frozen_m[name]), 4)); sc.cell(r, 3, statistics.mean(frozen_b[name]))
        sc.cell(r, 4, round(statistics.mean(rolling_m[name]), 4)); sc.cell(r, 5, statistics.mean(rolling_b[name]))
        sc.cell(r, 6, " ".join(f"{x:02d}" for x in frozen_picks[name][0])); sc.cell(r, 7, frozen_picks[name][1])
        for c in range(1, 8): sc.cell(r, c).font = body; sc.cell(r, c).alignment = ctr
        sc.cell(r, 1).alignment = Alignment(horizontal="left")
        for c in (3, 5): sc.cell(r, c).number_format = "0.0%"
        for c in (2, 4): sc.cell(r, c).number_format = "0.000"
    r = 4 + len(strategies)
    sc.cell(r, 1, "Random (theory)").font = bold; sc.cell(r, 2, f"=25/{N}"); sc.cell(r, 4, f"=25/{N}")
    sc.cell(r, 3, f"=1/{cfg['bonus']}"); sc.cell(r, 5, f"=1/{cfg['bonus']}")
    for c in (2, 4): sc.cell(r, c).number_format = "0.000"
    for c in (3, 5): sc.cell(r, c).number_format = "0.0%"
    for c in range(2, 6): sc.cell(r, c).font = bold; sc.cell(r, c).alignment = ctr
    sc.cell(r + 2, 1, f"Random luck over {len(test)} draws typically lands between {sims[int(.025 * len(sims))]:.2f} and {sims[int(.975 * len(sims))]:.2f} (95% range, simulated).").font = Font(name=F, italic=True)
    sc.cell(r + 3, 1, "Values on this sheet were computed by the script (rolling picks re-train before every draw); the Test Detail sheet shows the first strategy draw-by-draw with live formulas.").font = Font(name=F, italic=True)
    for c, w_ in zip("ABCDEFG", [34, 22, 18, 22, 18, 26, 14]): sc.column_dimensions[c].width = w_
    sc.row_dimensions[3].height = 34

    # ---- Tuning grid
    tg = wb.create_sheet("Tuning Grid")
    tg["A1"] = (f"Settings tuned on {ty} only, then judged on {sy}. Metric = average rank of the winning numbers as % of the pool "
                f"(50% = random, lower is better).")
    tg["A1"].font = Font(name=F, bold=True)
    header(tg, 3, ["Frequency weight", "Due weight", "Hot-streak weight", "Hot half-life (draws)", f"{ty} rank % (whites)", f"{sy} rank % (whites)",
                   f"{ty} rank % ({bn})", f"{sy} rank % ({bn})", f"Change {ty}->{sy}", "Best on " + str(ty) + "?"])
    srt = sorted(rows, key=lambda r: r["in_w"])
    for i, rr in enumerate(srt):
        r = 4 + i
        for c, v in enumerate(rr["p"], start=1): tg.cell(r, c, v)
        tg.cell(r, 5, rr["in_w"]); tg.cell(r, 6, rr["oos_w"]); tg.cell(r, 7, rr["in_b"]); tg.cell(r, 8, rr["oos_b"])
        tg.cell(r, 9, f"=F{r}-E{r}"); tg.cell(r, 10, "yes" if rr["p"] == best["p"] else ("default" if rr["p"] == L.DEFAULT_PARAMS else ""))
        for c in range(1, 11): tg.cell(r, c).font = body; tg.cell(r, c).alignment = ctr
        for c in (5, 6, 7, 8): tg.cell(r, c).number_format = "0.0%"
        tg.cell(r, 9).number_format = "+0.0%;-0.0%"
    lt = 3 + len(srt)
    tg.cell(lt + 2, 1, "Average across all settings").font = bold
    for c in (5, 6, 7, 8): tg.cell(lt + 2, c, f"=AVERAGE({CL(c)}4:{CL(c)}{lt})"); tg.cell(lt + 2, c).number_format = "0.0%"; tg.cell(lt + 2, c).font = bold
    tg.cell(lt + 3, 1, f"Rank correlation of settings between years").font = bold
    tg.cell(lt + 3, 5, f"=CORREL(E4:E{lt},F4:F{lt})"); tg.cell(lt + 3, 5).number_format = "+0.00;-0.00"; tg.cell(lt + 3, 5).font = bold
    tg.cell(lt + 4, 1, "Near 0 means a setting's results in one year say nothing about the next year. Rows are sorted by best performance in the training year.").font = Font(name=F, italic=True)
    for c in range(1, 11): tg.column_dimensions[CL(c)].width = 17
    tg.row_dimensions[3].height = 34

    # ---- Probability scoring
    ps = wb.create_sheet("Probability Scores")
    ps["A1"] = (f"Probability skill: average log(model probability / uniform probability) of the winning numbers in {sy}. "
                "0 = no better than uniform; negative = worse. Random draws cannot be beaten on average.")
    ps["A1"].font = Font(name=F, bold=True)
    header(ps, 3, ["Model", "Type", "Winners rank % (whites)", "Std error", "Rank % (bonus)", "Skill @ tau 0.25", "tau 0.5", "tau 1", "tau 2", "tau 4", "tau inf (uniform)"])
    r = 4
    for kind, bundle in (("Frozen on " + str(ty), fz), ("Rolling", ro)):
        for name, m in bundle.items():
            ps.cell(r, 1, name); ps.cell(r, 2, kind)
            ps.cell(r, 3, statistics.mean(m["rw"])); ps.cell(r, 4, mean_se(m["rw"])[1]); ps.cell(r, 5, statistics.mean(m["rb"]))
            for c, t in zip(range(6, 12), TAUS): ps.cell(r, c, statistics.mean(m["lw"][t]))
            for c in range(1, 12): ps.cell(r, c).font = body; ps.cell(r, c).alignment = ctr
            for c in (3, 4, 5): ps.cell(r, c).number_format = "0.0%"
            for c in range(6, 12): ps.cell(r, c).number_format = "+0.000;-0.000;0.000"
            r += 1
    for c, w_ in zip("ABCDEFGHIJK", [26, 18, 20, 12, 14, 16, 10, 10, 10, 10, 16]): ps.column_dimensions[c].width = w_
    ps.row_dimensions[3].height = 34

    # ---- Frequency by year (formulas over All Draws)
    fq = wb.create_sheet("Frequency by Year")
    fq["A1"] = f"{label}: how many times each number came up, by year (live formulas over the All Draws tab; current-format draws only)"; fq["A1"].font = Font(name=F, bold=True)
    years = sorted({d.year for d, _, _ in draws})
    heads = ["Number"] + [f"{y} white count" for y in years] + [f"{y} {bn} count" for y in years]
    header(fq, 3, heads); fq.row_dimensions[3].height = 34
    aq = "'All Draws'"
    for n in range(1, N + 1):
        r = 3 + n; fq.cell(r, 1, n)
        for j, y in enumerate(years):
            fq.cell(r, 2 + j, f'=SUMPRODUCT(({aq}!$B$2:$B${last_all}={y})*({aq}!$K$2:$K${last_all}="Current")*({aq}!$D$2:$H${last_all}=$A{r}))')
            if n <= cfg["bonus"]:
                fq.cell(r, 2 + len(years) + j, f'=SUMPRODUCT(({aq}!$B$2:$B${last_all}={y})*({aq}!$K$2:$K${last_all}="Current")*({aq}!$I$2:$I${last_all}=$A{r}))')
        for c in range(1, 2 + 2 * len(years)): fq.cell(r, c).font = body; fq.cell(r, c).alignment = Alignment(horizontal="center")
    for c in range(1, 2 + 2 * len(years)): fq.column_dimensions[CL(c)].width = 16
    fq.freeze_panes = "B4"

    # ---- Summary (first sheet)
    sm = wb.create_sheet("Summary", 0)
    sm["A1"] = f"{label}: trained on {ty}, tested on {sy}"; sm["A1"].font = Font(name=F, bold=True, size=14)
    lines = [
        ("Training draws", len(train), "Draws in the training year used to build the model"),
        ("Test draws", len(test), "Draws in the test year the model never saw"),
        ("Frozen formula: avg white matches / draw", f"='Match Distribution'!B11", f"Chance expectation {25 / N:.3f}"),
        ("Rolling formula: avg white matches / draw", f"='Match Distribution'!D11", "Re-trains before every draw"),
        ("Random expectation (matches / draw)", f"=25/{N}", "5 numbers chosen at random from the pool"),
        (f"Frozen: {bn} hit rate", "='Match Distribution'!B12", f"Chance expectation {100 / cfg['bonus']:.1f}%"),
        ("Winners rank, default formula frozen (whites)", statistics.mean(fz["Default formula"]["rw"]), "% of pool; 50% = random, lower is better"),
        (f"Winners rank, tuned on {ty}, in {ty}", best["in_w"], f"Best-looking setting {tuple(round(x, 2) for x in best['p'])}"),
        (f"Winners rank, same tuned setting, in {sy}", best["oos_w"], "Out-of-sample: what happened to the 'improvement'"),
        ("Setting correlation between years", rho, "Near 0 = last year's best says nothing about this year"),
        ("Probability skill, default frozen (tau 1)", statistics.mean(fz["Default formula"]["lw"][1.0]), "0 = same as uniform; negative = worse than uniform"),
    ]
    header(sm, 3, ["Measure", "Value", "What it means"])
    for i, (a, b, c) in enumerate(lines):
        r = 4 + i; sm.cell(r, 1, a); sm.cell(r, 2, b); sm.cell(r, 3, c)
        for cc in range(1, 4): sm.cell(r, cc).font = body
        sm.cell(r, 2).alignment = Alignment(horizontal="center")
    for r in (6, 7, 8): sm.cell(r, 2).number_format = "0.000"
    sm.cell(9, 2).number_format = "0.0%"
    for r in (10, 11, 12): sm.cell(r, 2).number_format = "0.0%"
    sm.cell(13, 2).number_format = "+0.00;-0.00"; sm.cell(14, 2).number_format = "+0.000;-0.000"
    sig = (0.5 - best["oos_w"]) / (mean_se([x for x in [0]])[0] + 1e-9) if False else None
    verdict = (f"Verdict: best setting found on {ty} ranked winners at {100 * best['in_w']:.1f}% in {ty} but {100 * best['oos_w']:.1f}% in {sy} "
               f"(50% = pure chance). Improvements found by tuning on one year did not carry to the next, as expected for random draws.")
    sm.cell(16, 1, verdict).font = bold; sm.merge_cells("A16:C16"); sm.cell(16, 1).alignment = Alignment(wrap_text=True, vertical="top"); sm.row_dimensions[16].height = 48
    sm.cell(18, 1, "Tabs: All Draws (every draw with dates) | Test Detail (draw-by-draw) | Match Distribution | Strategy Results | Tuning Grid | Probability Scores | Frequency by Year").font = Font(name=F, italic=True)
    sm.cell(19, 1, f"Source: NY Open Data lottery datasets (data.ny.gov). Generated {dt.datetime.now():%Y-%m-%d}. For-fun experiment; odds are 1 in {292201338 if game == 'powerball' else 290472336:,}.").font = Font(name=F, italic=True)
    for c, w_ in zip("ABC", [52, 16, 70]): sm.column_dimensions[c].width = w_

    path = os.path.join(out_dir, f"{game}_study_{ty}_train_{sy}_test.xlsx")
    wb.save(path); print(f"\nStudy workbook written: {path}")
