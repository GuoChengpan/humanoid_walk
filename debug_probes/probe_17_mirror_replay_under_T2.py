"""⑰ 的第 1 步验证（不训练、不改训练代码）：**T2 下镜像整集重放还是不是"错数据"**。

§1175 关掉 `WITH_MIRROR_REPLAY` 的理由是"镜像重放会把脉冲在同一侧重复注入 ⇒ 错数据"。
那句话说的是 **T1**：T1 的驱动窗口挂在绝对时钟上，所以镜像一条轨迹必须把时钟平移半周期，
否则模板还在驱动同一条腿。**⑯/T2 把驱动改成由接触事件决定** ⇒ 镜像轨迹的接触天然左右互换，
"必须靠时钟平移才能对齐模板"这个前提原则上消失了。

注意别把两件事混在一起：**奖励里的 `phase_reward` 至今仍挂在时钟上**（那是奖励，不是驱动），
所以"不平移时钟时逐帧奖励不等"是**预期的、无害的**；真正要判的是**驱动有没有翻边**。
本脚本因此把两者分开报。

四种组合：{T2 当前文件, T1=⑮ 重构件} × {shift=0, shift=半周期}，每个 4 个 seed。

通过线（跑之前写死）：
  P-1 状态等变（物理+观测，与奖励无关）：max‖mirror(obs_orig[t]) − obs_mir[t]‖∞ 在 shift=半周期 时 < 1e-3
  P-2 逐帧奖励等变（同 shift=半周期）：max|r_mir − r_orig| < 1e-3
  P-3 **驱动翻边**：mean|osc_active_L(mir) − osc_active_R(orig)| < 0.02
  P-4 关键差别：T1 在 shift=0 时 P-3 **必须不过**（这就是 §1175 的机理），
      而 T2 在 shift=0 时 P-3 **必须过**（这才证明反对理由随 ⑯ 消失）。
  P-5 长度：两种 shift 都不该让镜像集提前终止（帧数差 < 5%），否则镜像集是废数据。
"""
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

from mirror_map import MirrorMap, patch_mirror_symmetry

FR = 1500
SEEDS = tuple(int(x) for x in os.environ.get("MR_SEEDS", "123,2026,7,31").split(","))
MODEL = os.environ.get("PROBE_MODEL", "best_model/best_model.zip")
T1_SRC = "/tmp/train_b15_recon.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def build(W):
    raw = gym.make("Humanoid-v4", max_episode_steps=FR)
    patch_mirror_symmetry(raw.unwrapped.model)
    w = W.WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=True)
    return w


def record(W, seed, model):
    w = build(W); mj = w.unwrapped
    obs, _ = w.reset(seed=seed)
    q0, v0 = mj.data.qpos.copy(), mj.data.qvel.copy()
    clock = int(w.total_steps)
    out = dict(r=[], obs=[], act=[], aL=[], aR=[], ph=[], sw=[], hi=[])
    while True:
        a, _ = model.predict(obs, deterministic=True)
        a = np.asarray(a, dtype=np.float64)
        obs, r, term, trunc, info = w.step(a)
        out["r"].append(r); out["obs"].append(np.asarray(obs, dtype=np.float64)); out["act"].append(a)
        out["aL"].append(info.get("osc_active_L", 0.0)); out["aR"].append(info.get("osc_active_R", 0.0))
        out["ph"].append(info.get("phase_reward_gated", 0.0))
        out["sw"].append(info.get("swap_rate", 0.0)); out["hi"].append(info.get("height_reward_gated", 0.0))
        if term or trunc:
            break
    return {k: (np.array(v) if k != "act" else v) for k, v in out.items()} | dict(clock=clock, q0=q0, v0=v0)


def replay(W, src, shift, mm):
    w = build(W); mj = w.unwrapped
    w.reset()
    qm, vm = mm.mirror_qpos_qvel(src["q0"], src["v0"])
    mj.set_state(qm, vm)
    w.initial_y = float(mj.data.qpos[1])
    w.prev_body_x = w.prev_body_x_per_frame = float(mj.data.qpos[0])
    w.total_steps = src["clock"] + shift          # ← 唯一的自变量
    out = dict(r=[], obs=[], aL=[], aR=[], ph=[])
    for a_env in src["act"]:
        obs2, r, term, trunc, info = w.step(mm.mirror_action(a_env))
        out["r"].append(r); out["obs"].append(np.asarray(obs2, dtype=np.float64))
        out["aL"].append(info.get("osc_active_L", 0.0)); out["aR"].append(info.get("osc_active_R", 0.0))
        out["ph"].append(info.get("phase_reward_gated", 0.0))
        if term or trunc:
            break
    return {k: np.array(v) for k, v in out.items()}


model = SAC.load(MODEL)
T2, T1 = load("train_humanoid_sac.py", "t2"), load(T1_SRC, "t1")
_probe = build(T2); MM = MirrorMap(_probe.unwrapped.model); GP = _probe.gait_period
print(f"权重={MODEL}  gait_period={GP} 帧（半周期={GP//2}）  seed={SEEDS}")
print(f"\n{'件':4}{'时钟平移':>10}{'P-1 状态等变':>15}{'P-2 奖励等变':>15}{'P-3 驱动翻边':>15}"
      f"{'phase差(解释用)':>16}{'帧数差':>9}")
res = {}
for tag, W in (("T2", T2), ("T1", T1)):
    for shift in (0, GP // 2):
        d1, d2, d3, d4, dl = [], [], [], [], []
        for sd in SEEDS:
            src = record(W, sd, model)
            out = replay(W, src, shift, MM)
            n = min(len(src["r"]), len(out["r"]))
            dl.append(abs(n - len(src["r"])) / max(len(src["r"]), 1))
            mo = np.array([MM.mirror_obs(x) for x in src["obs"][:n]])
            d1.append(float(np.abs(mo - out["obs"][:n]).max()))
            d2.append(float(np.abs(out["r"][:n] - src["r"][:n]).max()))
            d3.append(float(np.abs(out["aL"][:n] - src["aR"][:n]).mean()))
            d4.append(float(np.abs(out["ph"][:n] - src["ph"][:n]).mean()))
        m = lambda v: float(np.mean(v))
        p1, p2, p3, p5 = m(d1) < 1e-3, m(d2) < 1e-3, m(d3) < 0.02, m(dl) < 0.05
        res[(tag, shift)] = (p1, p2, p3, p5)
        print(f"{tag:4}{shift:>10d}{m(d1):15.2e}{m(d2):15.4f}{m(d3):15.4f}{m(d4):16.4f}"
              f"{100*m(dl):8.1f}%  P1{'✓' if p1 else '✗'} P2{'✓' if p2 else '✗'} "
              f"P3{'✓' if p3 else '✗'} P5{'✓' if p5 else '✗'}")

print("\n=== 判读（P-4 是这次真正要的东西）===")
t2_0, t2_h = res[("T2", 0)], res[("T2", GP // 2)]
t1_0, t1_h = res[("T1", 0)], res[("T1", GP // 2)]
print(f"  驱动翻边 P-3：T1 不平移 {'过' if t1_0[2] else '**不过**'} ｜T1 平移 {'过' if t1_h[2] else '不过'}")
print(f"           T2 不平移 {'过' if t2_0[2] else '不过'} ｜T2 平移 {'过' if t2_h[2] else '不过'}")
print(f"  奖励等变 P-2：T1 平移 {'过' if t1_h[1] else '**不过**'} ｜T2 平移 {'过' if t2_h[1] else '**不过**'}"
      f"（不平移时两边都不过是预期的：phase_reward 仍挂时钟）")
if (not t1_0[2]) and t2_0[2]:
    print("  ⇒ **P-4 成立**：§1175 的'同侧重复注入脉冲'确实是 T1 的时钟窗口造成的，T2 已经不依赖平移 ⇒ "
          "⑰ 可以开 WITH_MIRROR_REPLAY（配合半周期平移，让 phase_reward 也对齐）")
elif t1_0[2] and t2_0[2]:
    print("  ⇒ T1 不平移也翻边 ⇒ §1175 当时的措辞不准（另有原因），但结论不变：T2 下重放可用")
elif not t2_h[2]:
    print("  ⇒ T2 平移后仍不翻边 ⇒ 重放机器与 T2 不兼容，⑰ 不得开 WITH_MIRROR_REPLAY，只能走价格那条路")
else:
    print("  ⇒ 驱动没对齐，⑰ 不得开 WITH_MIRROR_REPLAY")
if not t2_h[1]:
    print("  ⚠ 另外：T2 平移后逐帧奖励都不等变 ⇒ 先别谈注入 buffer，"
          "因为喂进去的镜像经验带着错的优势函数（这比'同侧脉冲'更伤）")
