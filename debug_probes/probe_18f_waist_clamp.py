"""⑰ 决策前的第 4 件测量：撅着腚到底是不是"承重的"（load-bearing）。

动机（用户判断）："撅着腚是为了保持平衡才做原地高抬腿的。" 上一支探针里我用
`corr(躯干俯仰, CoM−支撑脚误差)` 去判，**结果是不可判定**——支撑误差整段只跨 0.02–0.07 m，
4 cm 的窗口没有回归杠杆。所以这次不用相关，用**干预**：把腰的控制力矩换成"把 abdomen_y
拉到目标角"的 PD，看这个盆会不会塌。

读法（先把话讲清楚，免得事后编故事）：
  · 目标角 0°（中立）若**存活帧数暴跌** ⇒ 对这个已收敛的控制器，撅腚是承重的，
    任何"给 |abdomen_y| 上税"的改法都是在跟平衡打架 ⇒ ⑰ 只能改付款口径，不能改姿势。
  · 若存活/前进基本不变 ⇒ 姿势是三条"只看 pelvis 相对关节角"的通道买来的，可以放手。
  · 这条干预测的是**当前权重对姿势的依赖**，不是"重训后的策略会不会需要它"。
    后者要训练才知道（memory 里"assist-on-converged-weights 的门槛无效"是同一类警告）。

实现要点：gymnasium 的 humanoid.xml 用的是 **motor**（gear 100–300，ctrlrange ±0.4 = ±40~120 N·m），
不是位置执行器 ⇒ ctrl 不能直接当角度目标，必须自己写 PD。另外 `osc_k_start=1.0` ⇒ 模板在驱动帧
**完全覆盖** hip_y 的 ctrl，所以"把髋摆幅调小"这个变体只在非驱动帧起作用，读数时要记住。
控制项：先复现 abdomen_y 目标 = 实测 +29°，这一行的存活/前进应当与不加干预几乎相同。
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

spec = importlib.util.spec_from_file_location("b18f_main", "train_humanoid_sac.py")
W = importlib.util.module_from_spec(spec)
sys.modules["b18f_main"] = W
spec.loader.exec_module(W)

MODEL = os.environ.get("PROBE_MODEL", "best_model/best_model.zip")
SEEDS = tuple(int(x) for x in os.environ.get("S18_SEEDS", "123,2026,7,31,55,77,91,101").split(","))
FR = int(os.environ.get("S18_FRAMES", "1500"))

model = SAC.load(MODEL)
del model.replay_buffer
gc.collect()

ABD_Y_CTRL, HIPL_Y_CTRL, HIPR_Y_CTRL = 0, 9, 5


def roll(seed, clamp_to=None, kp=45.0, kd=6.0, hip_scale=None):
    """clamp_to 不为 None 时，abdomen_y 的 ctrl 被替换成把它拉到该角度(度)的 PD 力矩。"""
    raw = gym.make("Humanoid-v4", max_episode_steps=FR)
    W.patch_mirror_symmetry(raw.unwrapped.model)
    w = W.WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=True)
    w.reset(seed=seed)
    m, d = w.unwrapped.model, w.unwrapped.data
    adr = m.jnt_qposadr[m.joint("abdomen_y").id]
    vel = m.jnt_dofadr[m.joint("abdomen_y").id]
    n = cL = cR = ds = land = 0
    r_sum = 0.0
    ph_sum = 0.0
    jl = []; jrr = []; wl = []; wr = []; rr_l = []; rr_r = []
    x_prev = d.qpos[0]
    v_sum = 0.0
    abd_err = []
    sup_ahead = []      # 支撑帧里"脚在 CoM 前方"的 clip((脚−CoM+0.06)/0.26) 累加
    land_off = []       # 触地瞬间 脚x − CoMx（带符号，正=迈到身前）
    prev = (False, False)
    for _ in range(FR):
        a, _ = model.predict(w.unwrapped._get_obs(), deterministic=True)
        a = np.array(a, dtype=np.float64).copy()
        if hip_scale is not None:
            for c in (HIPL_Y_CTRL, HIPR_Y_CTRL):
                a[c] *= hip_scale
        if clamp_to is not None:
            tgt = np.radians(clamp_to)
            tau = kp * (tgt - d.qpos[adr]) - kd * d.qvel[vel]
            a[ABD_Y_CTRL] = float(np.clip(tau / 100.0, -0.4, 0.4))   # gear=100 ⇒ ctrl=τ/100
            abd_err.append(d.qpos[adr])
        _, r, ter, tru, info = w.step(a.astype(np.float32))
        if ter or tru:
            break
        ph_sum += float(info.get("phase_reward_gated", 0.0) or 0.0)
        adr_l = m.jnt_qposadr[m.joint("left_hip_y").id]; adr_r = m.jnt_qposadr[m.joint("right_hip_y").id]
        jl.append(np.degrees(d.qpos[adr_l])); jrr.append(np.degrees(d.qpos[adr_r]))
        for s_, store in (("left", wl), ("right", wr)):
            v = d.xpos[m.body(f"{s_}_thigh").id]
            u = d.xpos[m.body(f"{s_}_shin").id] - v
            store.append(np.degrees(np.arctan2(u[0], -u[2])))
        if not prev[0] and True:
            pass
        hitL, hitR, _, _ = w._foot_contact_state(m, d)
        com = d.qpos[0]
        for s_, h_ in (("left", hitL), ("right", hitR)):
            if h_:
                sup_ahead.append(max(0.0, min(1.0, (d.body(f"{s_}_foot").xpos[0] - com + 0.06) / 0.26)))
        if (not prev[0] and hitL) or (not prev[1] and hitR):
            s__ = "left" if (not prev[0] and hitL) else "right"
            land_off.append(d.body(f"{s__}_foot").xpos[0] - com)
        ds += int(hitL and hitR)
        land += int((not prev[0] and hitL) + (not prev[1] and hitR))
        prev = (hitL, hitR)
        cL += int(hitL); cR += int(hitR); n += 1
        rr_l.append(d.body("left_foot").xpos[0] - d.body("left_thigh").xpos[0])
        rr_r.append(d.body("right_foot").xpos[0] - d.body("right_thigh").xpos[0])
        r_sum += r
        v_sum += (d.qpos[0] - x_prev) / 0.015
        x_prev = d.qpos[0]
    return dict(n=n, ret=r_sum, ahead=float(np.sum(sup_ahead)),
                lo_p50=float(np.median(land_off)) if land_off else np.nan,
                lo_max=float(np.max(land_off)) if land_off else np.nan, phase=ph_sum, jointL=float(np.median(jl)) if jl else np.nan,
                worldL=float(np.median(wl)) if wl else np.nan, worldR=float(np.median(wr)) if wr else np.nan,
                reachL=float(np.median(rr_l)) if rr_l else np.nan, reachR=float(np.median(rr_r)) if rr_r else np.nan,
                v=v_sum / max(n, 1), duty=(cL + cR) / max(2 * n, 1),
                ds=ds / max(n, 1), land_s=land / max(n * 0.015, 1e-9),
                abd=float(np.degrees(np.median(abd_err))) if abd_err else np.nan,
                land=land)


print("=== A. 把腰按到指定角度，看这个盆塌不塌（6 seeds 中位；存活帧是主读数）===")
print(f"{'干预':26s} {'存活帧':>7s} {'腹部y':>7s} {'髋关节':>7s} {'世界大腿L/R':>13s} {'reachL/R':>13s} "
      f"{'v':>6s} {'duty':>6s} {'land/s':>7s} {'place梯子':>9s} {'触地脚−CoM p50/max':>18s} {'phase分/集':>10s} {'ret/集':>8s}")
for label, kw in (("不加干预（基线）", dict(clamp_to=None)),
                  ("腰按到 +29°(复现控制)", dict(clamp_to=29.0)),
                  ("腰按到 +15°", dict(clamp_to=15.0)),
                  ("腰按到 0°(中立)", dict(clamp_to=0.0)),
                  ("腰按到 −15°(后仰)", dict(clamp_to=-15.0)),
                  ("腰按到 +15° 且髋 ctrl×0.6", dict(clamp_to=15.0, hip_scale=0.6)),
                  ("髋 ctrl×0.6（腰不动）", dict(clamp_to=None, hip_scale=0.6))):
    rs = [roll(s, **kw) for s in SEEDS]
    med = lambda k: float(np.median([r[k] for r in rs]))
    print(f"{label:26s} {med('n'):7.0f} {med('abd'):7.1f} {med('jointL'):7.1f} "
          f"{med('worldL'):5.1f}/{med('worldR'):5.1f}  {med('reachL'):+6.3f}/{med('reachR'):+6.3f} "
          f"{med('v'):6.3f} {med('duty'):6.3f} {med('land_s'):7.2f} {med('ahead'):9.0f} "
          f"  {med('lo_p50'):+6.3f}/{med('lo_max'):+6.3f}  {med('phase'):10.0f} {med('ret'):8.0f}")

print("\n=== B. 顺带：这个盆在 det 口径下的分通道收入（不加干预，6 seeds 合并）===")
w = None
keys = None
acc = {}
cnt = 0
for s in SEEDS:
    raw = gym.make("Humanoid-v4", max_episode_steps=FR)
    W.patch_mirror_symmetry(raw.unwrapped.model)
    w = W.WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=True)
    w.reset(seed=s)
    m, d = w.unwrapped.model, w.unwrapped.data
    for _ in range(FR):
        a, _ = model.predict(w.unwrapped._get_obs(), deterministic=True)
        _, r, ter, tru, info = w.step(a)
        if ter or tru:
            break
        if keys is None:
            keys = [k for k, v in info.items() if isinstance(v, (int, float, np.floating, np.integer))]
        cnt += 1
        for k in keys:
            acc[k] = acc.get(k, 0.0) + float(info.get(k, 0.0) or 0.0)
print(f"  共 {cnt} 帧。先验台账恒等式 Σ(求和式里的通道) == r ：")
CH = ["forward_gauss_gated", "swing_forward_reward", "reach_reward_gated", "swing_min_reward_gated",
      "stride_bonus_gated", "net_balance", "height_reward_gated", "fall_penalty", "phase_reward_gated",
      "survival_bonus_gated", "position_swap_gated", "contact_phase_reward", "activity_diff_paid",
      "support_sym_penalty", "alternate_paid", "lateral_penalty", "hip_xz_angle_penalty",
      "flight_phase_penalty", "ctrl_cost", "short_stride_penalty", "negative_stride_penalty",
      "stride_balance_penalty", "slip_penalty", "speed_penalty"]
tot = 0.0
for k in CH:
    if k in acc:
        tot += acc[k] / max(cnt, 1)
print(f"    Σ通道/帧 = {tot:+.4f}   （对照 gait/reward 之类若不在 info 里，就用逐帧 r 的均值核对）")
for k in CH:
    if k in acc:
        print(f"    {k:26s} {acc[k]/cnt:+.5f} 分/帧")
    else:
        print(f"    {k:26s} 【info 里没有这个键，台账不完整】")
