"""接触相位真值校验：body_z<0.08 判据 vs MuJoCo 真接触，以及真实滑移。

动机（humanoid.xml 第61-62行）：
    <body name="left_foot" pos="0 0 -0.45">
        <geom name="left_foot" type="sphere" size="0.075" pos="0 0 0.1"/>
  脚球心在 foot body 原点上方 0.10 m、球半径 0.075 →
  球底触地时 body 原点 z = 0.075 - 0.10 = -0.025。
  而 WalkRewardWrapper 用 `left_z = body("left_foot").xpos[2] < 0.08` 当接触判定，
  等价于"球底离地 < 0.105 m"即算着地——会把大量真实空中时间判成支撑。

本探针只读，输出：
 1) 几何事实与 clearance 真值
 2) z 判据 vs 真接触的混淆矩阵 + 落地时序偏差
 3) wrapper 认定的摆动窗口里，实际仍在接触的时间占比
 4) 真双腾空占比 vs wrapper 声称的双腾空占比
 5) 真实接触点滑移速度（=脚底对地的切向速度）→ 每支撑段真实滑行距离
"""
import sys
import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
from probe_left_leg_chain import build_env, load_model, DT, EP_HORIZON, N_EP

CONTACT_Z = 0.08          # 现奖励用的判据


def main():
    vec = build_env()
    model = load_model(vec)
    raw = vec.envs[0].unwrapped
    m, d = raw.model, raw.data
    fl, fr = m.body("left_foot").id, m.body("right_foot").id
    trs = m.body("torso").id
    gl = [g for g in range(m.ngeom) if m.geom_bodyid[g] == fl]
    gr = [g for g in range(m.ngeom) if m.geom_bodyid[g] == fr]
    print(f"left_foot geoms={[(m.geom(g).name, m.geom(g).type, m.geom(g).size[0]) for g in gl]}")
    r_l = m.geom(gl[0]).size[0] if gl else 0.075

    rows = []
    ep = 0
    obs = vec.reset()
    for _ in range(8000):
        act, _ = model.predict(obs, deterministic=True)
        obs, rew, done, info = vec.step(act)
        if done[0]:
            ep += 1
            if ep >= N_EP:
                break
            continue
        # 真接触：世界与脚 geom 的 contact 对
        cl_true = cr_true = False
        slipL = slipR = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            gid = None
            if g1 in gl or g2 in gl:
                gid = "L"
            elif g1 in gr or g2 in gr:
                gid = "R"
            if gid is None:
                continue
            other = g2 if (g1 in gl or g1 in gr) else g1
            if m.geom_bodyid[other] != 0:          # 只认与世界的接触
                continue
            bid = fl if gid == "L" else fr
            if gid == "L":
                cl_true = True
            else:
                cr_true = True
            # 脚上该物质点的速度。cvel 表达在 com 处：v = v_com + ω × (p - xipos)
            om = d.cvel[bid][:3]
            vv = d.cvel[bid][3:6]
            p = c.pos
            vp = vv + np.cross(om, p - d.xipos[bid])
            if gid == "L":
                slipL = float(np.hypot(vp[0], vp[1]))
            else:
                slipR = float(np.hypot(vp[0], vp[1]))
        zc_l = d.geom_xpos[gl[0]][2] - r_l if gl else 0.0
        rows.append(dict(
            ep=ep, bodyLz=d.xpos[fl][2], bodyRz=d.xpos[fr][2],
            clearL=zc_l, clearR=(d.geom_xpos[gr[0]][2] - r_l if gr else 0.0),
            zL=bool(d.xpos[fl][2] < CONTACT_Z), zR=bool(d.xpos[fr][2] < CONTACT_Z),
            tL=cl_true, tR=cr_true, slipL=slipL, slipR=slipR,
            torsoz=d.xpos[trs][2],
        ))
    print(f"采集 {len(rows)} 步 / {ep} 集")

    bodyLz = np.array([r["bodyLz"] for r in rows])
    clearL = np.array([r["clearL"] for r in rows])
    clearR = np.array([r["clearR"] for r in rows])
    zL = np.array([r["zL"] for r in rows]); zR = np.array([r["zR"] for r in rows])
    tL = np.array([r["tL"] for r in rows]); tR = np.array([r["tR"] for r in rows])
    slipL = np.array([r["slipL"] for r in rows]); slipR = np.array([r["slipR"] for r in rows])

    print("\n=== 1) 几何事实 ===")
    print(f"foot body 原点 z:      p5={np.percentile(bodyLz,5):+.3f} p50={np.median(bodyLz):+.3f} p95={np.percentile(bodyLz,95):+.3f}")
    print(f"脚球底离地 clearance:  L p5={np.percentile(clearL,5):+.3f} p50={np.median(clearL):+.3f} p95={np.percentile(clearL,95):+.3f}")
    print(f"判据 body_z<0.08 等价于 clearance < {CONTACT_Z + 0.10 - 0.075:+.3f} m  （球心在原点上方0.10、半径0.075）")
    print(f"真实离地间隙最大时 body_z = {bodyLz.max():+.3f} ；clearance 最小值 = {min(clearL.min(), clearR.min()):+.4f}")

    print("\n=== 2) z 判据 vs 真接触（左腿）===")
    tp = np.mean(zL & tL); fp = np.mean(zL & ~tL); fn = np.mean(~zL & tL); tn = np.mean(~zL & ~tL)
    print(f"判据说接触且真接触={tp:.3f} | 判据说接触但**实际在空中**={fp:.3f} | 判据说空中但真接触={fn:.3f} | 都无={tn:.3f}")
    print(f"→ z 判据的\"支撑\"里有 {100*fp/max(tp+fp,1e-9):.0f}% 其实是空中")
    print("右腿: 判据支撑中实际空中占比 = "
          f"{100*np.mean(zR & ~tR)/max(np.mean(zR),1e-9):.0f}%")

    print("\n=== 3) wrapper 的\"摆动窗口\"(not z) 里真实状态 ===")
    swL = ~zL
    print(f"摆动占比={swL.mean():.3f} 其中真在地面接触={np.mean(swL & tL)/max(swL.mean(),1e-9):.3f}")
    airL = zL
    print(f"支撑占比={airL.mean():.3f} 其中真腾空={np.mean(airL & ~tL)/max(airL.mean(),1e-9):.3f}")

    print("\n=== 4) 腾空 ===")
    print(f"真双腾空(两脚都无接触)占比={np.mean(~tL & ~tR):.3f}")
    print(f"wrapper 声称双腾空(z 都不接触)占比={np.mean(~zL & ~zR):.3f}")
    print(f"真单脚支撑占比={np.mean(tL & ~tR):.3f}(L) {np.mean(tR & ~tL):.3f}(R)")
    print(f"wrapper 声称单腿={np.mean(zL & ~zR):.3f}(L) {np.mean(zR & ~zL):.3f}(R)")

    print("\n=== 5) 真实滑移（接触点切向速度）===")
    print(f"左: 接触时滑移速度 mean={slipL[tL].mean():.3f} p50={np.median(slipL[tL]):.3f} p90={np.percentile(slipL[tL],90):.3f} m/s")
    print(f"右: 接触时滑移速度 mean={slipR[tR].mean():.3f} p50={np.median(slipR[tR]):.3f} p90={np.percentile(slipR[tR],90):.3f} m/s")
    # 每个真支撑段积分滑移距离 与踝位移对比
    for name, tt, bb, sl in (("L", tL, bodyLz, slipL), ("R", tR, bodyLz, slipR)):
        pass
    for name, tt, sl in (("L", tL, slipL), ("R", tR, slipR)):
        segs, i = [], 0
        while i < len(tt) - 1:
            if not tt[i] and tt[i + 1]:
                j = i + 1
                while j < len(tt) - 1 and tt[j]:
                    j += 1
                if j - i > 3:
                    segs.append((sl[i+1:j] * DT).sum())
                i = j
            else:
                i += 1
        a = np.array(segs)
        print(f"{name}: 真支撑段 n={len(a)} 每段积分滑移 mean={a.mean():.4f} p50={np.median(a):.4f} p90={np.percentile(a,90):.4f} m")
    vec.close()


if __name__ == "__main__":
    main()
