"""全历史 det eval 的 `ev_reach_L/R` 分布：脚到底有没有落到（大腿 / 躯干）前方过？

`reach` 在奖励里的定义是 `foot_x − left_thigh.xpos[0]`（脚相对**大腿根**的前后，正=脚在腿前方）。
判据（写在前）：
  · 若历史上 reach_L 的最大值都 < 0 ⇒ "把脚插到身体前方"这个动作**从未发生过**，那不是价格问题；
  · 若在某些集里 reach_L > 0 且同时 ds≥0.03、ep_len=1500 ⇒ 前摆落点**已经被采样到过**，
    那才谈得上"价格没把它留住"，第 7 批才有讨论基础。
分段看：⑨（从头训+⑧价格）/ ⑩（+按躯干付钱）/ ⑩'（foot 量纲+碎步税）。
"""
import json

import numpy as np

PATH = "/home/guo/workspace/humanoid_walk/eval_logs/eval_gait.jsonl"
SEG = [("⑨ 从头训+⑧价格", 0, 555), ("⑩ 按躯干付钱", 555, 1200), ("⑩' foot量纲+碎步税", 1200, 10 ** 9)]


def col(rows, k):
    return np.array([float(r.get(k, float("nan"))) for r in rows])


def main():
    rows = [json.loads(l) for l in open(PATH) if l.strip()]
    print("总行数 %d = %d 次 eval（跨 ⑨/⑩/⑩' 三段，append-only）" % (len(rows), len(rows) // 5))
    E, DS = col(rows, "ep_len"), col(rows, "double_support_frac")
    RL, RR = col(rows, "ev_reach_L_mean"), col(rows, "ev_reach_R_mean")
    RL90, RR90 = col(rows, "ev_reach_L_p90"), col(rows, "ev_reach_R_p90")
    LIFL = col(rows, "ev_lift_L_mean")

    print("\n=== 按 ds 分箱（全历史）：reach = 脚相对大腿根的前后(m)，正=脚在腿前方 ===")
    print("%-14s %5s %8s %9s %9s %9s %9s %9s" % (
        "ds 区间", "n", "跑满率", "reach_L", "reach_R", "reachL_p90", "lift_L", "ep_len"))
    for lo, hi in [(0.0, 0.005), (0.005, 0.03), (0.03, 0.08), (0.08, 0.20), (0.20, 1.01)]:
        m = (DS >= lo) & (DS < hi)
        if not m.any():
            print("%-14s %5d" % ("[%.3f,%.3f)" % (lo, hi), 0))
            continue
        print("%-14s %5d %7.0f%% %9.3f %9.3f %9.3f %9.3f %9.0f" % (
            "[%.3f,%.3f)" % (lo, hi), int(m.sum()), 100 * float((E[m] >= 1499).mean()),
            np.nanmean(RL[m]), np.nanmean(RR[m]), np.nanmean(RL90[m]), np.nanmean(LIFL[m]), np.nanmean(E[m])))

    print("\n=== 三段各自的 reach_L / reach_R（正=脚落到腿前方）===")
    for tag, a, b in SEG:
        b = min(b, len(rows))
        if a >= b:
            continue
        sl = slice(a, b)
        print("  %-18s n=%4d｜reach_L p50 %+.3f p90 %+.3f max %+.3f｜reach_R p50 %+.3f p90 %+.3f max %+.3f｜"
              "reach_L>0 的集 %.1f%%" % (
                  tag, b - a, np.nanmedian(RL[sl]), np.nanpercentile(RL[sl], 90), np.nanmax(RL[sl]),
                  np.nanmedian(RR[sl]), np.nanpercentile(RR[sl], 90), np.nanmax(RR[sl]),
                  100 * float(np.nanmean(RL[sl] > 0))))

    good = (DS >= 0.03) & (E >= 1499)
    print("\n=== 关键交叉：ds≥0.03 且跑满 1500 的集（走路口径）里，脚有没有落到前方？ ===")
    print("  这样的集共 %d 个" % int(good.sum()))
    if good.sum():
        print("  reach_L：p50 %+.3f｜p90 %+.3f｜max %+.3f｜>0 的比例 %.0f%%" % (
            np.nanmedian(RL[good]), np.nanpercentile(RL[good], 90), np.nanmax(RL[good]),
            100 * float(np.nanmean(RL[good] > 0))))
        print("  reach_R：p50 %+.3f｜p90 %+.3f｜max %+.3f｜>0 的比例 %.0f%%" % (
            np.nanmedian(RR[good]), np.nanpercentile(RR[good], 90), np.nanmax(RR[good]),
            100 * float(np.nanmean(RR[good] > 0))))
        both = good & (RL > 0) & (RR > 0)
        print("  **两脚都落到前方（各自的 reach 均 >0）且 ds≥0.03 且跑满的集数 = %d**" % int(both.sum()))
        hi = good & (RL90 > 0.05)
        print("  左脚 p90 > 0.05 m（即确有大幅前插）的同口径集数 = %d" % int(hi.sum()))
    print("\n判读：若上面两个计数都是 0 ⇒ 前摆落点从未被采样到，问题不在价格；若有非 0，则值得为它单独设计一条钱。")


if __name__ == "__main__":
    main()
