"""分析 snapshot_screen.csv：按归档把快照与 best_model.zip 比功能量。

预注册规则（与 screen_snapshots.py 一致）：
  保留条件：min_adv >= best_min_adv + 0.01 且 swap_rate >= best_swap - 0.05
  其中 min_adv = min(adv_L, adv_R)，swap_rate 取两个种子的中位数。
另报 Pareto 改进：min_adv 更高且 swap 不低于 best-0.05，或 swap 更高且 min_adv 不低于 best-0.01。
"""
import csv, collections, os, sys

def med(xs):
    xs = sorted(x for x in xs if x == x)
    if not xs: return float("nan")
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2

def main(path="debug_probes/snapshot_screen.csv"):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    print("rows:", len(rows))
    by = collections.defaultdict(list)
    for r in rows:
        if r.get("error"): continue
        by[(r["archive"], r["file"], r["kind"])].append(r)
    agg = {}
    for (a, f, k), rs in by.items():
        agg[(a, f, k)] = dict(
            min_adv=med([float(x["min_adv"]) for x in rs]),
            swap=med([float(x["swap_rate"]) for x in rs]),
            ep=med([float(x["ep_len"]) for x in rs]),
            land=med([float(x["land_per_s"]) for x in rs]),
            ds=med([float(x["ds"]) for x in rs]),
            speed=med([float(x["speed"]) for x in rs]))
    keeps = []
    for (a, f, k), m in sorted(agg.items()):
        if k != "snapshot": continue
        best = agg.get((a, os.path.join(a, "best_model.zip"), "best"))
        if not best: continue
        rule = (m["min_adv"] >= best["min_adv"] + 0.01 and m["swap"] >= best["swap"] - 0.05)
        pareto = ((m["min_adv"] >= best["min_adv"] + 0.01 and m["swap"] >= best["swap"] - 0.05) or
                  (m["swap"] >= best["swap"] + 0.05 and m["min_adv"] >= best["min_adv"] - 0.01))
        if rule:
            keeps.append((a, f, m, best, "RULE"))
        elif pareto:
            keeps.append((a, f, m, best, "PARETO"))
    print(f"\n值得保留的快照: {len(keeps)}")
    for a, f, m, b, tag in keeps:
        print(f"[{tag}] {a}\n    {os.path.basename(f)}: min_adv {m['min_adv']:.3f} (best {b['min_adv']:.3f}), "
              f"swap {m['swap']:.3f} (best {b['swap']:.3f}), ep {m['ep']:.0f}, land/s {m['land']:.2f}, ds {m['ds']:.3f}")
    print("\n各归档 best 概览:")
    for (a, f, k), m in sorted(agg.items()):
        if k == "best":
            print(f"  {a:60s} min_adv {m['min_adv']:+.3f}  swap {m['swap']:.3f}  ep {m['ep']:.0f}")

if __name__ == "__main__":
    main(*sys.argv[1:])
