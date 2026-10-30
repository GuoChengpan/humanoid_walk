"""批次⑮（N1 reach 带重挂 / N2 stride 摘 walk_gate / N3 速度门连续化）的验证：
**同一轨迹、两套价格逐帧对比** —— 把 ⑬⑭ 的备份模块和改后的模块都 import 进来，
用同一组权重、同一个 seed、同一个动作序列喂两个独立的 env，先证明轨迹逐帧相同，
再看每一项的实付钱各自动了多少。

为什么必须做成对回放而不是算术：⑮ 改的全是"求和式与门怎么乘"，本项目吃过 `stall_penalty`
算了却没进求和式、两个因子恒等于 1 这类接线错；单项算术证明不了整体（见 §53 的同类教训）。

四件事：
  T0 轨迹恒等：max‖obs_new − obs_old‖ < 1e-9，且每集帧数相同 ⇒ 否则整张表作废
  T1 台账闭合：**改后的 NEW 必须仍然满足 Σinfo_paid == r**（N2 同时改了求和式与
     `info["stride_bonus_gated"]`，只改一边就会在这里现形）
  T2 逐项差额：height/survival/reach/stride/碎步税/总和，按"分/帧"给 old、new、Δ
  T3 断点：把 _speed_ramp 与旧二值门在 v∈[0,0.6] 上扫，报最大跳变
"""
import collections
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

DT, FR = 0.015, 1500
SEEDS = tuple(int(x) for x in os.environ.get("A_SEEDS", "123,2026,7,31,55,77,91,101").split(","))
MODEL = os.environ.get("PROBE_MODEL", "best_model/best_model.zip")   # ⑬⑭ 1.86M，md5 42d0a80c…
OLD_SRC = "train_humanoid_sac_batch13_14_1_86M.py"
NEW_SRC = "train_humanoid_sac.py"

SIGN = {
    "+": ["forward_gauss_gated", "swing_forward_reward", "reach_reward_gated",
          "swing_min_reward_gated", "stride_bonus_gated", "net_balance", "height_reward_gated",
          "fall_penalty", "phase_reward_gated", "survival_bonus_gated", "position_swap_gated",
          "contact_phase_reward"],
    "-": ["speed_penalty", "activity_diff_paid", "support_sym_penalty", "alternate_paid",
          "lateral_penalty", "hip_xz_angle_penalty", "flight_phase_penalty", "ctrl_cost",
          "short_stride_penalty", "negative_stride_penalty", "stride_balance_penalty", "slip_penalty"],
}


def r_hat(info):
    v = 0.0
    for k in SIGN["+"]:
        if k in info:
            v += float(info[k])
    for k in SIGN["-"]:
        if k in info:
            v -= float(info[k])
    return v


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


OLD, NEW = load(OLD_SRC, "t_old"), load(NEW_SRC, "t_new")
model = SAC.load(MODEL)


def build(W):
    raw = gym.make("Humanoid-v4", max_episode_steps=FR)
    W.patch_mirror_symmetry(raw.unwrapped.model)
    return W.WalkRewardWrapper(raw, swing_drive_weight=2.0, asym_inject_p=0.0, osc_enable=True)


def run(w, seed):
    acc, frames, resid, ep_len = collections.defaultdict(list), 0, [], None
    obs, _ = w.reset(seed=seed)
    hist = [np.asarray(obs, dtype=np.float64).copy()]
    while True:
        a, _ = model.predict(obs, deterministic=True)
        a = np.asarray(a, dtype=np.float64)
        obs, r, term, trunc, info = w.step(a)
        frames += 1
        hist.append(np.asarray(obs, dtype=np.float64).copy())
        resid.append(r_hat(info) - float(r))
        for k in SIGN["+"] + SIGN["-"] + ["tiny_step_frame_frac", "speed", "walk_gate", "forward_gate",
                                          "swap_gate", "stride_length", "lo_held"]:
            if k in info:
                acc[k].append(float(info[k]))
        if term or trunc:
            ep_len = frames
            break
    return acc, ep_len, np.array(resid), np.array(hist)


tot_old = collections.defaultdict(list)
nframes = 0
resid = {"old": [], "new": []}
max_obs_gap = 0.0
len_gap = []
for sd in SEEDS:
    wo, wn = build(OLD), build(NEW)
    Ao, fo, ro, ho = run(wo, sd)
    An, fn, rn, hn = run(wn, sd)
    if ho.shape == hn.shape:
        max_obs_gap = max(max_obs_gap, float(np.abs(hn - ho).max()))
    else:
        max_obs_gap = max(max_obs_gap, float("inf"))
    len_gap.append(abs(fo - fn))
    resid["old"] += list(ro)
    resid["new"] += list(rn)
    for k, v in Ao.items():
        tot_old[k] = tot_old.get(k, []) + v
    for k, v in An.items():
        tot_old[k + "|new"] = tot_old.get(k + "|new", []) + v
    nframes += max(fo, fn)

print(f"权重={MODEL}  {len(SEEDS)} 集  合计 {nframes} 帧")
print(f"T0 轨迹恒等：末帧 obs 最大差={max_obs_gap:.2e}  每集帧数差={len_gap} ⇒ "
      f"{'同一条轨迹，表可信' if max_obs_gap < 1e-9 and sum(len_gap) == 0 else '**轨迹不同，整张表作废**'}")
print("T1 台账闭合 Σinfo−r（两版都要闭合；NEW 若不闭合而 OLD 闭合 ⇒ 是 ⑮ 接线错）")
for lab in ("old", "new"):
    a = np.array(resid[lab])
    print(f"  {lab}: 中位={np.median(a):+.5f} p95|r|={np.percentile(np.abs(a),95):.5f} 最大={np.abs(a).max():.5f}"
          f" ⇒ {'闭合' if np.percentile(np.abs(a),95) < 0.02 else '**没闭合：有一项没接上或接重了**'}")

KEYS = ["height_reward_gated", "survival_bonus_gated", "reach_reward_gated", "stride_bonus_gated",
        "short_stride_penalty", "phase_reward_gated", "forward_gauss_gated", "position_swap_gated",
        "negative_stride_penalty", "alternate_paid", "ctrl_cost", "fall_penalty", "net_balance"]
print(f"\n=== T2 逐项实付（分/帧，{nframes} 帧）===")
print(f"  {'通道':26s} {'⑬⑭':>9} {'⑮':>9} {'Δ':>9}   Δ/集")
for k in KEYS:
    o = np.mean(tot_old[k]) if tot_old.get(k) else np.nan
    nn = np.mean(tot_old[k + "|new"]) if tot_old.get(k + "|new") else np.nan
    sgn = -1.0 if k in SIGN["-"] else 1.0
    if np.isfinite(o) and np.isfinite(nn):
        print(f"  {k:26s} {sgn*o:+9.4f} {sgn*nn:+9.4f} {sgn*(nn-o):+9.4f}   {sgn*(nn-o)*nframes/len(SEEDS):+8.1f}")
    else:
        print(f"  {k:26s} {'—':>9} {'—':>9} {'—':>9}   (键缺失)")
print(f"  {'每帧总分（对照）':26s}")
for lab, kk in (("⑬⑭", ""), ("⑮", "|new")):
    s = 0.0
    for k in KEYS:
        v = tot_old.get(k + kk)
        if v:
            s += (-1.0 if k in SIGN["-"] else 1.0) * np.mean(v)
    print(f"  {'  ' + lab + ' 合计':24s} {s:+9.4f} 分/帧 = {s*nframes/len(SEEDS):+8.1f} 分/集")

print("\n=== T3 断点扫描 v∈[0,0.6] ===")
vs = np.linspace(0.0, 0.6, 601)
ramp = NEW.WalkRewardWrapper._speed_ramp
probe = object.__new__(NEW.WalkRewardWrapper)
import ast
txt = open(NEW_SRC, encoding="utf-8").read()
tree = ast.parse(txt)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WalkRewardWrapper")
init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
CC = {}
for node in ast.walk(init):
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Attribute):
        try:
            v = node.value
            CC[node.targets[0].attr] = -float(v.operand.value) if (isinstance(v, ast.UnaryOp)
                                                                   and isinstance(v.operand, ast.Constant)) else float(v.value)
        except Exception:
            pass
probe.speed_gate_lo, probe.speed_gate_ref = CC["speed_gate_lo"], CC["speed_gate_ref"]
new_g = np.array([ramp(probe, x) for x in vs])
old_g = (vs > 0.3).astype(float)
jump = lambda arr: float(np.abs(np.diff(arr)).max())
print(f"  ⑬⑭ 二值门：最大跳变 {jump(old_g):.3f}（在 v=0.30）；⑮ 斜坡 lo/ref=({CC['speed_gate_lo']},{CC['speed_gate_ref']})："
      f"最大跳变 {jump(new_g):.4f}")
print(f"  工作点等价性：v=0.427 旧={float((0.427>0.3)*1.0):.3f} 新={float(ramp(probe,0.427)):.3f} ⇒ "
      f"{'发钱量不变' if abs(ramp(probe,0.427)-1.0)<1e-9 else '**变了，N3 不止是抹平断点**'}")
print(f"  站立侧：v=0.05 旧={float((0.05>0.3)*1.0):.3f} 新={float(ramp(probe,0.05)):.3f} ⇒ "
      f"{'站立仍拿不到' if ramp(probe,0.05)==0.0 else '**站立也能拿钱，N3 削弱了防重开站立陷阱的那一侧**'}")
