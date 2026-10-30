"""(c) 问题的事实答案：把"任意落地"下的两脚间距分布量出来。

用 debug_probes/trace_944k.npz（弹跳者 det rollout，含 lx/rz 等原始物理量）离线重算：
  - 用脚 body z 的低位/高位阈值重建接触状态（比旧的 z<0.08 更贴近真实触地）
  - 对每条腿的每次落地帧：spacing = 落地脚x − 另一脚x；按"另一脚是否在支撑"分组
  - 统计 spacing 的 mean/p10/p50/p90、负值占比，以及 NEGATIVE 罚的期望强度
只读，不加载模型、不碰训练进程。
"""
import numpy as np

T = np.load("/home/guo/workspace/humanoid_walk/debug_probes/trace_944k.npz")
ep, lx, rx, lz, rz, tx = (T["ep"], T["lx"], T["rx"], T["lz"], T["rz"], T["torso_x"])
bd = T["bd"]
keep = ~bd.astype(bool)
ep, lx, rx, lz, rz, tx = ep[keep], lx[keep], rx[keep], lz[keep], rz[keep], tx[keep]
n = len(lx)
print(f"样本 {n} 步 / {ep.max()+1} 集")

for name, z in (("lz", lz), ("rz", rz)):
    print(f"{name}: p1={np.percentile(z,1):+.3f} p10={np.percentile(z,10):+.3f} "
          f"p50={np.percentile(z,50):+.3f} p90={np.percentile(z,90):+.3f}")

# 双阈值重建接触：进入 z<LO、离开 z>HI（消抖）
for LO, HI in ((0.00, 0.03), (-0.02, 0.01), (0.02, 0.05)):
    def contact(z):
        c = np.zeros(n, bool); on = False
        for i in range(n):
            if not on and z[i] < LO:
                on = True
            elif on and z[i] > HI:
                on = False
            c[i] = on
        return c
    cL, cR = contact(lz), contact(rz)
    flight = (~cL) & (~cR)
    ds = cL & cR
    print(f"\n[LO={LO:+.2f} HI={HI:+.2f}] duty_L={cL.mean():.3f} duty_R={cR.mean():.3f} "
          f"ds={ds.mean():.3f} flight={flight.mean():.3f} duty_sum={cL.mean()+cR.mean():.3f}")

    # 每条腿的落地帧（本腿 off->on，且同一集内）
    for side, c, fx, ofx, oc in (("L", cL, lx, rx, cR), ("R", cR, rx, lx, cL)):
        land = np.where(np.array([c[i] and not c[i-1] and ep[i] == ep[i-1] for i in range(1, n)] + [False]))[0]
        if len(land) < 4:
            continue
        sp_all = fx[land] - ofx[land]
        sp_ds = sp_all[oc[land]]
        sp_air = sp_all[~oc[land]]
        q = lambda a, p: float(np.percentile(a, p)) if len(a) else float("nan")
        print(f"  {side}: 落地{len(land)}次 支撑中落地占{100*np.mean(oc[land]):.0f}% | "
              f"spacing(全部) mean={sp_all.mean():+.3f} p10={q(sp_all,10):+.3f} p50={q(sp_all,50):+.3f} 负占={100*np.mean(sp_all<0):.0f}%")
        print(f"       {'':14s} spacing(双撑落地) n={len(sp_ds)} p50={q(sp_ds,50):+.3f} 负占={100*np.mean(sp_ds<0):.0f}%"
              f" | spacing(腾空落地) n={len(sp_air)} mean={sp_air.mean():+.3f} p10={q(sp_air,10):+.3f} p50={q(sp_air,50):+.3f} 负占={100*np.mean(sp_air<0):.0f}%")
        # 若按 NEGATIVE_STRIDE_WEIGHT=10, SCALE=0.08 逐次落地结算，每集扣多少
        pen = np.clip(-sp_all / 0.08, 0, 1) * 10.0
        print(f"       若『任意落地』启用负步幅罚: 单次均值={pen.mean():.2f} 分/落地, "
              f"落地频率={len(land)/max((ep==ep[0]).sum(),1):.3f}/步 → ≈{pen.mean()*len(land)/max((ep==ep[0]).sum(),1)*1500:.0f} 分/1500步集")
    # reach 对照（脚x − 髋x 需要 thigh 位置，trace 里没有；用躯干近似做量级对照）
    print(f"  对照：落地帧 脚相对躯干 (lx-torso)/rx-torso: L={np.mean([lx[i]-tx[i] for i in np.where(np.diff(cL.astype(int)) == 1)[0] if ep[i]==ep[i-1] if i < n]):+.3f}")
