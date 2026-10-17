"""Which history window (how many past days/draws to learn from) and which strategy works best, per game.
Walk-forward: every draw is predicted using only earlier draws. Windows are picked on the FIRST half of the test draws
and then judged on the SECOND half, so a lucky window can't masquerade as skill."""
import datetime as dt, statistics, math, itertools, random, json, sys
import lotto_predictor as L

WINDOWS = [14, 30, 45, 60, 90, 120, 180, 270, 365]
CORE = {"Formula (45/25/30)": (.45, .25, .30, 20), "Frequency only": (1, 0, 0, 20), "Due only": (0, 1, 0, 20),
        "Hot streak (half-life 8)": (0, 0, 1, 8), "Hot streak (half-life 40)": (0, 0, 1, 40)}

def rank_sweep(game, draws, start):
    cfg = L.GAMES[game]
    test = sorted([x for x in draws if x[0] >= start], key=lambda x: x[0])
    res = {w: {k: {"w": [], "b": []} for k in CORE} for w in WINDOWS}
    for d, win, b in test:
        for w in WINDOWS:
            H, _ = L.history_for(draws, d, w, cfg)
            if len(H) < 8: continue
            for k, (wf, wd, wh, hl) in CORE.items():
                sw = L.score_pool([x for x, _ in H], cfg["whites"], wf, wd, wh, hl)
                sb = L.score_pool([[bb] for _, bb in H], cfg["bonus"], wf, wd, wh, hl)
                res[w][k]["w"].append(L.rank_pct(sw, win)); res[w][k]["b"].append(L.rank_pct(sb, [b]))
    return test, res

def combo_sweep(game, draws, start, windows):
    cfg = L.GAMES[game]
    test = sorted([x for x in draws if x[0] >= start], key=lambda x: x[0])
    out = {(n, w): {"m": [], "b": []} for n, _ in L.STRATEGIES for w in windows}
    for d, win, b in test:
        for w in windows:
            H, _ = L.history_for(draws, d, w, cfg)
            if len(H) < 8: continue
            for n, f in L.STRATEGIES:
                c, pb = f(H, cfg)
                if not c: continue
                out[(n, w)]["m"].append(len(set(c) & set(win))); out[(n, w)]["b"].append(int(pb == b))
    return test, out

def se(v): return statistics.pstdev(v) / math.sqrt(len(v)) if len(v) > 1 else float("nan")

if __name__ == "__main__":
    game = sys.argv[1]
    draws = L.load_draws(game, "data"); cfg = L.GAMES[game]
    start = dt.date(2025, 7, 1) if game == "powerball" else dt.date(2025, 7, 1)
    test, rs = rank_sweep(game, draws, start)
    n = len(test); h = n // 2
    print(f"\n=== {cfg['label']}: {n} walk-forward draws from {start}; first half {h} (choose window), second half {n - h} (judge) ===")
    summary = {}
    for k in CORE:
        print(f"\n{k}: winners' rank % (50 = random) by history window")
        print("  window(days)   all draws     1st half   2nd half")
        best = None
        for w in WINDOWS:
            v = rs[w][k]["w"]
            if len(v) < n * 0.9: continue
            a, b2 = v[:h], v[h:]
            print(f"  {w:5d}          {100*statistics.mean(v):5.1f}%        {100*statistics.mean(a):5.1f}%     {100*statistics.mean(b2):5.1f}%")
            if best is None or statistics.mean(a) < best[1]: best = (w, statistics.mean(a), statistics.mean(b2))
        print(f"  -> window chosen on 1st half: {best[0]}d ({100*best[1]:.1f}%), on 2nd half it scores {100*best[2]:.1f}%")
        summary[k] = best
    # window averaged across components
    print("\nAverage across the 5 scorers, by window (all draws):")
    for w in WINDOWS:
        vals = [statistics.mean(rs[w][k]["w"]) for k in CORE if len(rs[w][k]["w"]) >= n * .9]
        if vals: print(f"  {w:5d}d  {100*statistics.mean(vals):5.2f}%")
    # combo strategies
    ws = [60, 90, 180, 365]
    t2, co = combo_sweep(game, draws, dt.date(2026, 1, 1), ws)
    exp = 25 / cfg["whites"]
    print(f"\nActual picks, 2026 walk-forward ({len(t2)} draws): avg white matches (random {exp:.3f}, 1 s.e. ~ {math.sqrt(exp*0.9)/math.sqrt(len(t2)):.2f}); bonus hit % (random {100/cfg['bonus']:.1f}%)")
    print("  " + f"{'Strategy':34s}" + "".join(f"{w:>14d}d" for w in ws))
    for nme, _ in L.STRATEGIES:
        print("  " + f"{nme:34s}" + "".join(f"{statistics.mean(co[(nme,w)]['m'] or [0]):8.3f}/{100*statistics.mean(co[(nme,w)]['b'] or [0]):4.1f}%" for w in ws))
    json.dump({"rank": {str(w): {k: statistics.mean(rs[w][k]['w']) for k in CORE} for w in WINDOWS if all(rs[w][k]['w'] for k in CORE)},
               "combo": {f"{a}|{w}": [statistics.mean(v['m']), statistics.mean(v['b'])] for (a, w), v in co.items() if v['m']}},
              open(f"output/{game}_windows.json", "w"))
