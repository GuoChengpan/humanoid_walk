"""决定性检验：左腿摆动驱动尖峰 dL 是否转化为左落地步幅？

背景：SAC_18 跑到 ~1M 步，dL 中位仍贴地板 0.006，左步幅 med 0.06 vs 右 0.39。
要判断瓶颈在"激励不够"还是"机制下游（相位/离地间隙）"，唯一干净的证据是
条件对比：dL 高时左步幅是否显著变大。

同时澄清 swing_drive_reward 的稀疏结构（只在落地帧非零），
避免把"逐帧 med=0"误读为"几乎从不发钱"。
"""
import glob
import os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

FLOOR, TOP, WEIGHT = 0.006, 0.03, 2.0


def load(path):
    ea = EventAccumulator(path, size_guidance={"scalars": 0})
    ea.Reload()
    return {t: np.array([x.value for x in ea.Scalars(t)])
            for t in ea.Tags()["scalars"]}


def main():
    os.chdir(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    p = sorted(glob.glob("sac_humanoid_walk/SAC_18/events.out.tfevents.*"),
               key=os.path.getmtime)[-1]
    d = load(p)
    dl, dr = d["gait/last_drive_L"], d["gait/last_drive_R"]
    sl, sr = d["gait/last_left_stride"], d["gait/last_right_stride"]
    rw = d["gait/swing_drive_reward"]
    neg = d["gait/negative_stride_rate"]

    print(f"file={os.path.basename(p)}")
    print(f"\n[1] swing_drive_reward 稀疏结构（逐帧）")
    print(f"    >0 占比 = {np.mean(rw > 0)*100:.2f}%  (落地帧才发钱)")
    nz = rw[rw > 0]
    print(f"    发钱时的值: n={len(nz)} med={np.median(nz):.3f} "
          f"满值({WEIGHT})占比={np.mean(nz >= WEIGHT*0.999)*100:.1f}%")
    print(f"    每步平均贡献 = {rw.mean():.4f}")

    print(f"\n[2] 左右驱动不对称（1M 步后）")
    print(f"    dL med={np.median(dl):.4f}  p90={np.percentile(dl,90):.4f}")
    print(f"    dR med={np.median(dr):.4f}  p90={np.percentile(dr,90):.4f}")
    print(f"    比值 dR/dL (med) = {np.median(dr)/np.median(dl):.1f}x")
    print(f"    dL < floor 占比 = {np.mean(dl < FLOOR)*100:.1f}%  "
          f"→ 这些结算点奖励恒为 0（死区，无梯度）")

    print(f"\n[3] 关键：dL 分位 → 左步幅（驱动尖峰有没有换成步幅？）")
    qs = np.quantile(dl, [0, .25, .5, .75, .9, .95, .99])
    edges = np.unique(np.concatenate([[0.0], qs, [np.inf]]))
    for a, b in zip(edges[:-1], edges[1:]):
        m = (dl >= a) & (dl < b) & np.isfinite(sl)
        if m.sum() < 5:
            continue
        print(f"    dL∈[{a:.4f},{b if np.isfinite(b) else 99:.4f}) n={m.sum():4d} "
              f"左步幅 med={np.median(sl[m]):.4f} p90={np.percentile(sl[m],90):.4f} "
              f"负步幅率={np.mean(neg[m]>0)*100:.1f}%")

    # 明确阈值对比：贴地板 vs 超过 top
    lo = (dl < FLOOR) & np.isfinite(sl)
    hi = (dl >= TOP) & np.isfinite(sl)
    mid = (dl >= FLOOR) & (dl < TOP) & np.isfinite(sl)
    print(f"\n[4] 三档对比（决定性）")
    for name, m in (("dL<floor(0.006)", lo), ("floor<=dL<top", mid), ("dL>=top(0.03)", hi)):
        if m.sum() == 0:
            continue
        print(f"    {name:18s} n={m.sum():4d}  左步幅 med={np.median(sl[m]):.4f} "
              f"mean={np.mean(sl[m]):.4f}  p75={np.percentile(sl[m],75):.4f}")
    if lo.sum() > 5 and hi.sum() > 5:
        print(f"    → 高驱动档 / 死区档 左步幅比 = "
              f"{np.median(sl[hi])/max(np.median(sl[lo]),1e-9):.2f}x")

    r = np.corrcoef(dl[np.isfinite(sl)], sl[np.isfinite(sl)])[0, 1]
    rs = np.corrcoef(dr[np.isfinite(sr)], sr[np.isfinite(sr)])[0, 1]
    print(f"    Pearson dL~左步幅 = {r:.3f}   dR~右步幅 = {rs:.3f}")

    print(f"\n[5] 步幅不对称的当前刻度")
    print(f"    左 med={np.median(sl):.4f}  右 med={np.median(sr):.4f}  "
          f"比值 右/左={np.median(sr)/np.median(sl):.1f}x")
    print(f"    min(L,R) med={np.median(np.minimum(sl,sr)):.4f}  "
          f"STRIDE_FLOOR=0.10 → 满分需 min>=0.32")

    print(f"\n[6] 近/远期对照（最后 100K vs 最早的 100K 之后稳定段）")
    n = len(dl)
    for name, seg in (("early(100-300K)", slice(int(n*0.1), int(n*0.3))),
                      ("late(700K-end)", slice(int(n*0.7), n))):
        a, b, c = dl[seg], sl[seg], sr[seg]
        print(f"    {name}: dL med={np.median(a):.4f} >top={np.mean(a>=TOP)*100:.1f}% "
              f"左步幅 med={np.median(b):.4f} 右步幅 med={np.median(c):.4f} "
              f"不对称={np.median(c)/max(np.median(b),1e-9):.1f}x")


if __name__ == "__main__":
    main()
