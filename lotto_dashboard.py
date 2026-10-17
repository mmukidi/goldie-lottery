"""Builds the data behind the daily dashboard (3 small JSON files) from the latest draws.
  python lotto_dashboard.py [--no-fusion] [--out output/dash]
Files: games.json (one row per game), numbers.json (one row per number), draws.json (recent results)."""
import os, json, argparse, datetime as dt, statistics
from zoneinfo import ZoneInfo
import lotto_predictor as L

def draw_ts(game, d):
    cfg = L.GAMES[game]; hh, mm = (22, 59) if game == "powerball" else (23, 0)
    return dt.datetime(d.year, d.month, d.day, hh, mm, tzinfo=ZoneInfo("America/New_York")).isoformat()

def build(data_dir, fusion=True, days=365):
    games, numbers, recent = [], [], []
    start = L.default_start()
    for g, cfg in L.GAMES.items():
        draws = L.load_draws(g, data_dir); target = L.next_draw_date(g, start)
        H, wstart = L.history_for(draws, target, days, cfg)
        N, NB = cfg["whites"], cfg["bonus"]
        picks = {}
        for name, f in L.STRATEGIES:
            c, b = f(H, cfg); picks[name] = (sorted(c), b)
        main_name = L.STRATEGIES[0][0]; main, mb = picks[main_name]
        W = [w for w, _ in H]; B = [[b] for _, b in H]
        zf, zd, zh = L.score_components(W, N); zfb, zdb, zhb = L.score_components(B, NB)
        fz = None
        if fusion:
            try:
                import lotto_fusion as LF
                if os.path.exists(os.path.join(data_dir, f"long_{g}.csv")):
                    LF.sync_long(g, data_dir); tgt, eq, wz, z, nw = LF.next_picks(g, data_dir, target); fz = (eq, wz, z, nw)
            except Exception as e:
                print("fusion skipped:", e)
        sums = sorted(sum(w) for w in W); lo, hi = L.sum_band(H)
        row = dict(game=g, label=cfg["label"], bonus_name=cfg["bonus_name"], draw_date=target.isoformat(), draw_time=cfg["draw_time"], draw_ts=draw_ts(g, target),
                   odds=292201338 if g == "powerball" else 290472336, window_draws=len(H), window_start=wstart.isoformat(), pool=N, bonus_pool=NB,
                   p1=main[0], p2=main[1], p3=main[2], p4=main[3], p5=main[4], bonus=mb, total=sum(main), odd=sum(x % 2 for x in main), low=sum(x <= N // 2 for x in main),
                   consec=sum(1 for i in range(4) if main[i + 1] - main[i] == 1), band_lo=lo, band_hi=hi, median_sum=statistics.median(sums))
        for key, (nm, (c, b)) in zip(["balanced", "overdue", "hot", "pairs"], [(n, picks[n]) for n, _ in L.STRATEGIES[1:]]):
            row[key] = " ".join(f"{x:02d}" for x in c) + f" + {b:02d}"; row[key + "_name"] = nm
        if fz:
            row["fusion"] = " ".join(f"{x:02d}" for x in fz[0]); row["fusion_weighted"] = " ".join(f"{x:02d}" for x in fz[1]); row["fusion_z"] = round(fz[2], 2); row["fusion_trusted"] = fz[3]
        games.append(row)
        alt = {}
        for nm, (c, b) in picks.items():
            for x in c: alt[("white", x)] = alt.get(("white", x), 0) + 1
            alt[("bonus", b)] = alt.get(("bonus", b), 0) + 1
        fus = set(fz[0]) if fz else set()
        for kind, pool, series, z1, z2, z3, expn in (("white", N, W, zf, zd, zh, 5.0), ("bonus", NB, B, zfb, zdb, zhb, 1.0)):
            for n in range(1, pool + 1):
                flags = [1 if n in s else 0 for s in series]            # newest first
                first = flags.index(1) if 1 in flags else len(flags)
                tot = sum(flags); hits20 = sum(flags[:20]); hits50 = sum(flags[:50])
                numbers.append(dict(game=g, kind=kind, number=n, hits20=hits20, hits50=hits50, hits_all=tot, exp20=round(20 * expn / pool, 2), exp50=round(50 * expn / pool, 2),
                                    exp_all=round(len(flags) * expn / pool, 1), since_last=first, avg_gap=round(len(flags) / tot, 1) if tot else None,
                                    z_freq=round(z1[n], 2), z_due=round(z2[n], 2), z_hot=round(z3[n], 2), strip="".join(map(str, flags[:20])),
                                    picked=int((kind == "white" and n in main) or (kind == "bonus" and n == mb)), fusion_pick=int(kind == "white" and n in fus),
                                    strategies=alt.get((kind, n), 0)))
        for d, w, b in draws[:14]:
            recent.append(dict(game=g, date=d.isoformat(), n1=w[0], n2=w[1], n3=w[2], n4=w[3], n5=w[4], bonus=b, overlap=len(set(w) & set(main))))
    return games, numbers, recent

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", default="data"); ap.add_argument("--out", default="output/dash"); ap.add_argument("--no-fusion", action="store_true"); a = ap.parse_args()
    games, numbers, recent = build(a.data_dir, not a.no_fusion); os.makedirs(a.out, exist_ok=True)
    for nm, rows in (("games", games), ("numbers", numbers), ("draws", recent)):
        json.dump(rows, open(os.path.join(a.out, nm + ".json"), "w"), separators=(",", ":"))
    print(f"wrote {len(games)} games, {len(numbers)} numbers, {len(recent)} recent draws to {a.out}")
