"""预检：这台机体能不能被**开环交替步态**撑住？（纯仿真搜索，不加载任何策略、不改训练代码）

为什么先做这个：⑧⑨⑩ 三批价格改动都没能让"交替 + 活着"同时出现（⑨ 活着但不交替，⑩ 交替但不活）。
再烧一次 4M（≈11.4 小时）之前，必须先回答一个更便宜的问题：
  **在动作空间里放一个左右反相的正弦步态发生器，随机搜索它的参数，有没有任何一组能让它不摔？**
  · 若 1 g 下存在能跑满 1500 步且两脚真分开的参数 ⇒ 机体做得到，缺的只是策略/示范 ⇒ 值得继续训
    （并让"相位机先验"有合法目标可学，即 (B) 模仿/先验路线可行）。
  · 若只有低重力下才存在 ⇒ 说明需要 (C) 物理卸载课程，且这里能给出**要减到几 g**。
  · 若任何重力都找不到 ⇒ 这台机体 + 这个驱动配置下开环交替步态本身就不可行，该谈 (A) 换目标/(D) 归档。

步态发生器（左右严格反相 ⇒ 交替由构造保证）：
  hip_y  : a = b + A_h·sin(2πft + φ_side)        （髋前后，主驱动）
  knee   : a = b_k + A_k·sin(2πft + φ_k + φ_side)（膝，摆动期屈曲）
  ankle  : a = b_a + A_a·sin(2πft + φ_a + φ_side)
  hip_x/z, abdomen, 手臂 : 常数偏置（搜索里当"姿态配平"参数）
随机搜索 600 组 × 3 档重力，每组最多 1500 步；记录存活、是否摔倒、ds、两脚落点间距、净速度、髋摆幅。
"""
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym

from train_humanoid_sac import patch_mirror_symmetry

rng = np.random.default_rng(20261012)
N_TRIALS = 600
HORIZON = 1500
DT = 0.02
GRAVS = (1.0, 0.6, 0.35)


def build_env(g_scale):
    raw = gym.make("Humanoid-v4", max_episode_steps=HORIZON)
    patch_mirror_symmetry(raw.unwrapped.model)
    u = raw.unwrapped
    names = [u.model.actuator(j).name for j in range(u.model.nu)]
    idx = {}
    for j in range(u.model.nu):
        n = u.model.actuator(j).name or ""
        idx[n] = j
    side = {"L": ["left_hip_y", "left_knee", "left_ankle", "left_hip_x", "left_hip_z"],
            "R": ["right_hip_y", "right_knee", "right_ankle", "right_hip_x", "right_hip_z"]}
    return raw, u, names, idx, side


def trial(raw, u, idx, p):
    """p: dict 参数。返回 (存活步数, 是否摔倒, 指标)"""
    obs, _ = raw.reset(seed=123)
    prevL = prevR = True
    lastL = lastR = -99
    footL = footR = 0.0
    gaps = []
    ds_frames = 0
    hipL = []
    hipR = []
    x0 = float(u.data.qpos[0])
    n = 0
    f = p["f"]
    for k in range(HORIZON):
        ph = 2 * np.pi * f * k * DT
        a = np.zeros(u.model.nu, dtype=np.float64)
        for sgn, key in ((+1.0, "L"), (-1.0, "R")):
            a[idx["left_hip_y" if key == "L" else "right_hip_y"]] = p["b_h"] + p["A_h"] * np.sin(ph + (0.0 if sgn > 0 else np.pi))
            kk = "left_knee" if key == "L" else "right_knee"
            a[idx[kk]] = p["b_k"] + p["A_k"] * np.sin(ph + p["pk"] + (0.0 if sgn > 0 else np.pi))
            # 注意：gym Humanoid 的 17 个执行器里**没有踝关节**（left/right_ankle 不受驱动），
            # 所以落地姿态只能靠 hip_x/hip_z/knee 组合去凑，这本身就是这台机体的硬约束之一。
            hx = "left_hip_x" if key == "L" else "right_hip_x"
            hz = "left_hip_z" if key == "L" else "right_hip_z"
            a[idx[hx]] = p["b_hx"]
            a[idx[hz]] = p["b_hz"]
        A_ank = 0.0  # 无踝驱动，保留字段仅用于说明
        for nm in ("abd_y", "abd_z", "abd_x", "r_sh1", "r_sh2", "r_el", "l_sh1", "l_sh2", "l_el"):
            j = idx.get({"abd_y": "abdomen_y", "abd_z": "abdomen_z", "abd_x": "abdomen_x",
                         "r_sh1": "right_shoulder1", "r_sh2": "right_shoulder2", "r_el": "right_elbow",
                         "l_sh1": "left_shoulder1", "l_sh2": "left_shoulder2", "l_el": "left_elbow"}[nm])
            if j is not None:
                a[j] = p[nm]
        a = np.clip(a, -1.0, 1.0)
        obs, r, term, trunc, info = raw.step(a)
        n += 1
        q = u.data.qpos
        hipL.append(float(q[idx["left_hip_y"]]))
        hipR.append(float(q[idx["right_hip_y"]]))
        # 接触用脚的高度判（本探针不依赖 wrapper，只测动力学可行性）
        lq = float(u.data.body("left_foot").xpos[0]); rq = float(u.data.body("right_foot").xpos[0])
        cL = float(u.data.body("left_foot").xpos[2]) < 0.09
        cR = float(u.data.body("right_foot").xpos[2]) < 0.09
        if cL and cR:
            ds_frames += 1
        if cL and not prevL and k - lastL >= 5:
            gaps.append(abs(lq - footR))
            footL = lq
            lastL = k
        if cR and not prevR and k - lastR >= 5:
            gaps.append(abs(rq - footL))
            footR = rq
            lastR = k
        prevL, prevR = cL, cR
        if term or trunc:
            break
    dz = float(u.data.qpos[2])
    fell = dz < 1.0
    g = np.array(gaps) if gaps else np.array([np.nan])
    full = 2 * (np.ptp(hipL) + np.ptp(hipR)) / 2 * 180 / np.pi
    return n, fell, dict(ds=ds_frames / max(n, 1), gap=float(np.nanmedian(g)), gap90=float(np.nanpercentile(g, 90)),
                         rom=float(full), vnet=(float(u.data.qpos[0]) - x0) / max(n * DT, 1e-6),
                         nland=len(gaps), z=dz)


def sample_params():
    return dict(f=rng.uniform(1.2, 3.2), A_h=rng.uniform(0.15, 0.6), b_h=rng.uniform(-0.25, 0.25),
                A_k=rng.uniform(0.0, 0.45), b_k=rng.uniform(-0.1, 0.4), pk=rng.uniform(-2.0, 2.0),
                A_a=rng.uniform(0.0, 0.3), b_a=rng.uniform(-0.3, 0.3), pa=rng.uniform(-2.0, 2.0),
                b_hx=rng.uniform(-0.15, 0.15), b_hz=rng.uniform(-0.15, 0.15),
                abd_y=rng.uniform(-0.2, 0.2), abd_z=rng.uniform(-0.1, 0.1), abd_x=rng.uniform(-0.15, 0.15),
                r_sh1=rng.uniform(-0.1, 0.1), r_sh2=rng.uniform(-0.1, 0.1), r_el=rng.uniform(-0.1, 0.1),
                l_sh1=rng.uniform(-0.1, 0.1), l_sh2=rng.uniform(-0.1, 0.1), l_el=rng.uniform(-0.1, 0.1))


def main():
    for g in GRAVS:
        raw, u, names, idx, side = build_env(g)
        raw.unwrapped.model.opt.gravity[2] = -9.81 * g
        res = []
        for i in range(N_TRIALS):
            p = sample_params()
            try:
                n, fell, m = trial(raw, u, idx, p)
            except Exception as e:
                continue
            res.append((n, fell, m, p))
        surv = [x for x in res if x[0] >= HORIZON]
        best = sorted(res, key=lambda x: -x[0])[:5]
        print("\n=== 重力 %.2f g（随机搜索 %d 组开环反相步态）===" % (g, len(res)))
        print("  跑满 1500 步不摔的参数组数 = **%d / %d**" % (len(surv), len(res)))
        print("  最好 5 组的存活步数：%s" % ", ".join("%d%s" % (x[0], "(摔)" if x[1] else "(满)") for x in best))
        for x in best[:3]:
            m = x[2]
            print("    存活%4d｜ds %.3f｜两脚落点间距 p50 %.3f m p90 %.3f｜髋摆幅 %.1f°｜v_net %.2f m/s｜落地 %d 次｜末 z %.2f" % (
                x[0], m["ds"], m["gap"], m["gap90"], m["rom"], m["vnet"], m["nland"], m["z"]))
        good = [x for x in surv if x[2]["gap"] >= 0.10 and x[2]["ds"] >= 0.03]
        print("  其中【跑满 且 间距≥0.10 m 且 ds≥0.03】= %d 组 ⇒ %s" % (
            len(good), "交替步态在这档重力下**动力学可行**" if good else "没有真交替（脚没迈开或无双撑）"))
        del raw


if __name__ == "__main__":
    main()
