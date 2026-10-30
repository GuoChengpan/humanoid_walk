"""⑰ 定价（预算中性版）：把同一笔钱换到不同操作量上，原地高抬腿盆(M)与慢速真走盆(W)各挣多少。

标本说明：
  M = ⑯ 现权重 det rollout，通道面值取 info（**实测**），操作量自己逐帧算。
  H = 弹跳标本：⑧ 的权重在当前模板(k=1.0 覆盖髋)下 25-30 帧即塌，这类对比已在 ⑯ 的 P2 判过
      "不可用于设计判断"，所以**这里不用 H**，只比 M vs W。
  W = 人体参考的运动学反事实。每个通道都按文件里的真公式算（:873 phase / :1725 stride /
      :1827 walk_gate / :1568 dense_ds / :1391-1394 height+survival 的 speed_ramp），假设逐条写明：
        v=0.84 m/s，2.13 landings/s，duty 0.573/腿，flight 0.000，
        ds = 0.146（=2×0.573−1 的算术值；池内实测 0.27 另跑一遍做敏感性），
        单步 0.393 m，触地 lead 0.233 m，髋的骨盆相对前摆与世界系大腿角都 ≥28.6° ⇒ 两种 phase 口径都饱和。

变体（都不新增收入轴，只是把已有的钱换目标，避免又开一条漏洞）：
  V0 现价
  VA `phase_reward` 操作量：骨盆相对髋角 → 世界系大腿角（公式/clip/权重不动）
  VB 把 phase 那一整笔预算改付给**支撑前脚时长**：clip((脚x−CoMx)/0.20) × 该脚正在支撑
  VC VB + `stride_bonus`/`position_swap` 回到"另一脚仍在支撑"的落地才结算
"""
import importlib.util
import os
import sys
import gc
import warnings

import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")

spec = importlib.util.spec_from_file_location("b18g2_main", "train_humanoid_sac.py")
W = importlib.util.module_from_spec(spec)
sys.modules["b18g2_main"] = W
spec.loader.exec_module(W)

MODEL = os.environ.get("PROBE_MODEL", "best_model/best_model.zip")
SEEDS = tuple(int(x) for x in os.environ.get("S18_SEEDS", "123,2026,7,31,55,77").split(","))
FR = 1500
DT = 0.015
PH_CLIP = 0.5            # :900 activity / 0.5 rad
PH_BUF = 50              # leg_activity_buffer_size (:721)

model = SAC.load(MODEL)
del model.replay_buffer
gc.collect()

PAY = ["forward_gauss_gated", "reach_reward_gated", "swing_min_reward_gated", "stride_bonus_gated",
       "net_balance", "height_reward_gated", "phase_reward_gated", "survival_bonus_gated",
       "position_swap_gated", "alternate_paid", "swing_forward_reward", "contact_phase_reward"]
TAX = ["flight_phase_penalty", "short_stride_penalty", "lateral_penalty", "slip_penalty",
       "support_sym_penalty", "hip_xz_angle_penalty", "ctrl_cost", "stride_balance_penalty",
       "negative_stride_penalty", "activity_diff_paid"]


def roll(seed):
    raw = gym.make("Humanoid-v4", max_episode_steps=FR)
    W.patch_mirror_symmetry(raw.unwrapped.model)
    w = W.WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=True)
    w.reset(seed=seed)
    m, d = w.unwrapped.model, w.unwrapped.data
    al = m.jnt_qposadr[m.joint("left_hip_y").id]
    ar = m.jnt_qposadr[m.joint("right_hip_y").id]
    buf = {k: [] for k in PAY + TAX}
    jl, jr, wl, wr, com, fxL, fxR, hL, hR = [], [], [], [], [], [], [], [], []
    land = ds_land = 0
    prev = (False, False)
    for _ in range(FR):
        a, _ = model.predict(w.unwrapped._get_obs(), deterministic=True)
        _, r, ter, tru, info = w.step(a)
        if ter or tru:
            break
        for k in PAY + TAX:
            buf[k].append(float(info.get(k, 0.0) or 0.0))
        hitL, hitR, _, _ = w._foot_contact_state(m, d)
        L = ((not prev[0]) and hitL) or ((not prev[1]) and hitR)
        land += int(L)
        ds_land += int(L and hitL and hitR)          # 本帧落地时**两脚都接触** = wrapper 的 ds 落地口径
        prev = (hitL, hitR)
        jl.append(d.qpos[al]); jr.append(d.qpos[ar])
        for s_, st in (("left", wl), ("right", wr)):
            v = d.xpos[m.body(f"{s_}_thigh").id]
            u = d.xpos[m.body(f"{s_}_shin").id] - v
            st.append(float(np.arctan2(u[0], -u[2])))   # arctan2 已是弧度；再套 np.radians 会把值压 57 倍
        com.append(d.qpos[0])
        fxL.append(d.body("left_foot").xpos[0]); fxR.append(d.body("right_foot").xpos[0])
        hL.append(int(hitL)); hR.append(int(hitR))
    n = len(com)

    def act(arr, fwd):
        vals = [min(1.0, max(0.0, fwd(np.array(arr[i - PH_BUF:i]))) / PH_CLIP)
                for i in range(PH_BUF, n + 1, 10)]
        return float(np.mean(vals)) if vals else 0.0
    joint_act = 0.5 * (act(jl, lambda s: -s.min()) + act(jr, lambda s: -s.min()))
    world_act = 0.5 * (act(wl, lambda s: s.max()) + act(wr, lambda s: s.max()))
    fore = float(np.sum(np.clip((np.array(fxL) - np.array(com)) / 0.20, 0, 1) * np.array(hL)
                  + np.clip((np.array(fxR) - np.array(com)) / 0.20, 0, 1) * np.array(hR)))
    per = {k: float(np.sum(v)) for k, v in buf.items()}
    return dict(n=n, per=per, joint_act=joint_act, world_act=world_act, fore=fore,
                ds_frac=ds_land / max(land, 1), land=land)


R = [roll(s) for s in SEEDS]
med = lambda f: float(np.median([f(r) for r in R]))
N = med(lambda r: r["n"])
M = {k: med(lambda r: r["per"][k]) for k in PAY + TAX}
ratio = med(lambda r: r["world_act"]) / max(med(lambda r: r["joint_act"]), 1e-9)
print(f"M 原地高抬腿：存活 {N:.0f} 帧   落地 {med(lambda r: r['land']):.0f} 次/集   "
      f"ds 落地占比 {med(lambda r: r['ds_frac']):.4f}")
print(f"  phase activity：关节口径 {med(lambda r: r['joint_act']):.3f} → 世界口径 {med(lambda r: r['world_act']):.3f}"
      f"（×{ratio:.3f}）   支撑前脚时长(clip 累加) = {med(lambda r: r['fore']):.1f}"
      f"（W 盆按 30% 帧估算 ≈ {0.30*N:.0f}）")
print(f"  现价分/集：毛 {sum(max(0,M[k]) for k in PAY):.0f}  税 {sum(max(0,M[k]) for k in TAX):.0f}"
      f"  净 {sum(max(0,M[k]) for k in PAY)-sum(max(0,M[k]) for k in TAX):.0f}")
for k in PAY:
    if M[k] > 0.5:
        print(f"    +{k:26s} {M[k]:7.0f}")
for k in TAX:
    if M[k] > 0.5:
        print(f"    -{k:26s} {M[k]:7.0f}")

PH_M = M["phase_reward_gated"]
PLACE_BUDGET = PH_M                       # VB 把 phase 的预算整体搬走，不加钱


def w_basin(DS_W):
    wg = 0.20 + 0.80 * np.clip((DS_W + 0.20) / (0.10 + 0.20), 0, 1)     # :1827 walk_gate(ds−flight)
    dsf = 0.55 + 0.45 * min(1.0, DS_W / 0.25)                            # :1568 dense_ds (floor .55 ref .25)
    land = 2.13 * N * DT
    V = dict.fromkeys(PAY + TAX, 0.0)
    V["phase_reward_gated"] = N * 1.0                                    # 两种口径都饱和，swap_gate≈1
    V["height_reward_gated"] = M["height_reward_gated"] / max(dsf * 1.0, 1e-9) * dsf  # 面值同，门更大
    V["survival_bonus_gated"] = M["survival_bonus_gated"]
    V["stride_bonus_gated"] = land * min(1.0, (0.393 - 0.10) / 0.10) * 35.0 * min(1.0, 2.5 / 2.13)
    V["reach_reward_gated"] = land * 2.0                                 # lead 0.233 > 0.20 ⇒ 每落地饱和
    V["swing_min_reward_gated"] = M["swing_min_reward_gated"] * wg / 0.20 * 0.20
    V["position_swap_gated"] = M["position_swap_gated"]
    V["forward_gauss_gated"] = N * 0.90 * wg                             # 高斯在 v=0.84 处 ≈0.9 面值
    V["net_balance"] = M["net_balance"]
    V["alternate_paid"] = -0.05 * N                                      # 真交替 ⇒ 拿奖（面值按实测比例放大）
    V["lateral_penalty"] = M["lateral_penalty"] * 0.3
    V["slip_penalty"] = M["slip_penalty"] * 0.5
    V["support_sym_penalty"] = M["support_sym_penalty"] * 0.5
    V["hip_xz_angle_penalty"] = M["hip_xz_angle_penalty"]
    V["flight_phase_penalty"] = 0.0
    V["short_stride_penalty"] = 0.0
    V["ctrl_cost"] = M["ctrl_cost"] * 0.7
    return V, wg, dsf, land


def net(V):
    return sum(max(0.0, V[k]) for k in PAY) - sum(max(0.0, V[k]) for k in TAX)


for DS_W in (0.146, 0.27):
    Wv, wg, dsf, land = w_basin(DS_W)
    ds_M = med(lambda r: r["ds_frac"])
    rows = []
    # M 侧
    m0 = net(M)
    mA = dict(M, phase_reward_gated=PH_M * ratio); m1 = net(mA)
    mB = dict(mA, phase_reward_gated=0.0)
    fore_scale_M = med(lambda r: r["fore"]) / (0.30 * N)                 # 归一到"人应得满预算"
    m2 = net(mB) + PLACE_BUDGET * fore_scale_M
    mC = m2 - (M["stride_bonus_gated"] + M["position_swap_gated"]) * (1 - ds_M)
    # W 侧
    w0 = net(Wv)
    w1 = net(dict(Wv))                                                   # 世界口径也饱和 ⇒ 不变
    w2 = net(dict(Wv, phase_reward_gated=0.0)) + PLACE_BUDGET            # 前脚在支撑 ⇒ 整笔预算拿回
    wC = w2                                                              # 人的落地几乎全是 ds 落地 ⇒ 不减
    print(f"\n=== ds={DS_W:.3f}（walk_gate={wg:.2f} dense_ds={dsf:.2f} 落地={land:.0f}/集）净分/集 ===")
    print(f"{'价格':8s} {'M 原地高抬腿':>14s} {'W 慢速真走':>12s} {'W−M':>10s} {'W/M':>7s}")
    for nm, a, b in (("V0 现价", m0, w0), ("VA 世界口径", m1, w1),
                     ("VB 支撑前脚", m2, w2), ("VC +ds结算", mC, wC)):
        print(f"{nm:10s} {a:14.0f} {b:12.0f} {b - a:+10.0f} {b / a:7.2f}")
print("\n读法：只有 W/M > 1 的那一列才可能把策略推开 M 盆；W/M ≈ 1 等于没改，< 1 是在奖励原地踏步。")
