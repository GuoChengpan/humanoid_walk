"""历史 det eval 全库检查：**双支撑 `ds` 到底是"走路"的指标，还是"快摔"的指标？**

动机（批次⑧ 定价时撞见的事实）：把 dense 收入挂到 ds 上之前，必须知道 ds>0 的 Episode 是什么样的。
本次用 `eval_logs/eval_gait.jsonl`（append-only，跨轮累计 **315 集**，每集一行真统计）按 ds 分箱，
直接量每箱的：存活长度、摔率(`ep_len<1500`)、回报、速度、腾空、左右摆幅。
判据（写在前）：若 ds 升高与**摔率升高/存活缩短**同向，那么"付钱买 ds"= 付钱买瘫倒姿势，
批次⑧ 这条改动属于恶性激励，不该做；反之若高 ds 箱大多能跑满 1500 步，才值得继续。
"""
import json

import numpy as np

PATH = "/home/guo/workspace/humanoid_walk/eval_logs/eval_gait.jsonl"
BINS = [0.0, 0.01, 0.03, 0.08, 0.20, 0.50, 1.01]


def main():
    rows = [json.loads(l) for l in open(PATH) if l.strip()]
    ds = np.array([float(r["double_support_frac"]) for r in rows])
    ln = np.array([float(r["ep_len"]) for r in rows])
    ret = np.array([float(r["ret"]) for r in rows])
    sp = np.array([float(r["speed"]) for r in rows])
    fl = np.array([float(r["flight_frac"]) for r in rows])
    rl = np.array([float(r["ev_rom_L_mean"]) for r in rows])
    rr = np.array([float(r["ev_rom_R_mean"]) for r in rows])
    full = ln >= 1499.0                      # 跑满 1500 步 = 没摔
    print("样本：eval_gait.jsonl 共 %d 集（跨轮 append-only，det 评测）" % len(rows))
    print("  ds 分布：p50=%.4f p90=%.4f p99=%.4f max=%.4f｜ds>0.03 的集数 = %d（%.1f%%）" % (
        np.percentile(ds, 50), np.percentile(ds, 90), np.percentile(ds, 99), ds.max(),
        int((ds > 0.03).sum()), 100.0 * (ds > 0.03).mean()))

    print("\n%-16s %5s %8s %8s %9s %7s %7s %8s %8s" % (
        "ds 区间", "n", "跑满率", "摔率", "平均存活", "回报", "速度", "腾空", "romL/romR"))
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (ds >= lo) & (ds < hi)
        if m.sum() == 0:
            print("%-16s %5d" % ("[%.2f,%.2f)" % (lo, hi), 0))
            continue
        print("%-16s %5d %7.0f%% %7.0f%% %9.0f %7.0f %7.2f %8.3f %5.1f/%.0f" % (
            "[%.2f,%.2f)" % (lo, hi), m.sum(), 100 * full[m].mean(), 100 * (1 - full[m].mean()),
            ln[m].mean(), ret[m].mean(), sp[m].mean(), fl[m].mean(), rl[m].mean(), rr[m].mean()))

    # 相关与逻辑回归式的单变量判据
    print("\n=== 相关性（全部 %d 集）===" % len(rows))
    for name, v in (("ep_len", ln), ("ret", ret), ("speed", sp), ("flight", fl),
                    ("rom_L", rl), ("rom_R", rr)):
        print("  corr(ds, %-7s) = %+0.3f" % (name, float(np.corrcoef(ds, v)[0, 1])))
    print("  ds>0.03 的集里摔的比例 = %.0f%%（全体摔率 %.0f%%）" % (
        100 * (1 - full[ds > 0.03].mean()) if (ds > 0.03).sum() else float("nan"),
        100 * (1 - full.mean())))
    hi = ds > 0.03
    if hi.sum():
        print("  高 ds 集（n=%d）里：平均存活 %.0f 步、回报 %.0f、速度 %.2f、腾空 %.3f" % (
            hi.sum(), ln[hi].mean(), ret[hi].mean(), sp[hi].mean(), fl[hi].mean()))
    ok = ds.max()
    print("\n=== 判读 ===")
    print("  这条改动的否决条件（预先声明）：ds 与「跑满 1500 步」负相关，或 ds>%.3f 的箱子平均存活 <600 步。" % 0.03)
    print("  历史里出现过的最大 det ds = %.4f ⇒ 就算门控做成 clip(ds/0.05)，" % ok)
    print("     **存活集**几乎全部落在 0 附近（见 [0.00,0.01) 箱的跑满率），门控等价于把全部收入清零。")


if __name__ == "__main__":
    main()
