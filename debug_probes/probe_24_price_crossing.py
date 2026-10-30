"""⑱ 的地基：同一份轨迹、两套价格互相打分（V9 的 16 项 vs ⑰ 的 22 项）。

动机（V9 归档实测 + V9.txt 读出来的价格表）：V9 那份权重在**未修改的 XML** 上跑出
`v_net = 2.61 m/s、单步 L/R 0.671/0.672 m、交替率 1.000、左右差 0.1%、1000/1000 帧存活`
⇒ "球足无踝做不到交替前进"是错的，是**我们的价格后来把前进买没了**。V9 的求和式只有 16 项，
关键是 `forward_reward = 0.5 × max(0, v)`（线性、单向、无门控），而 ⑰ 是
`forward_gauss × walk_gate` 实测 **0.0035 分/帧 = 5 分/集**。

方法（为什么这样比才可信）：det 策略不吃奖励 ⇒ 同一 actor + 同一物理 ⇒ **轨迹逐帧相同**，
于是两个 ret 的差**只能**来自价格。我用躯干 x 轨迹逐帧相等（max|Δx| < 1e-12）当控件；
控件不过就整表作废。⑮ 那一行需要模板（T1 时钟驱动），而 V9 的 wrapper 不会注入模板，
所以我在探针里按 T1 的公式自己注入，并用同一个 Δx 控件验证注入与 wrapper 内部一致。

预测（跑之前写下，防事后编故事）：
  P1 在 V9 价格下，慢跑 ≫ 高抬腿（因为 0.5v 线性项在 2.6 m/s 上给 1310 分/集）
  P2 在 ⑰ 价格下，高抬腿 ≥ 慢跑（因为前进只占 0.2%，而 height/stride/phase 在慢速下更便宜）
  若 P2 不成立 ⇒ ⑰ 的价格其实仍偏爱前进，那"价格凶手"就是别的东西（模板/门控/熵），
  我们就不该用"回到 V9 价格"当 ⑱ 的理由。
"""
import importlib.util
import io
import json
import os
import sys
import zipfile
import warnings

import numpy as np
import gymnasium as gym
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")

V9_TXT = "best_model_saved/V9/V9-实现交替慢跑/V9.txt"
V9_W = "best_model_saved/V9/V9-实现交替慢跑/best_model.zip"
B15_W = "best_model/tmp_model_data/SAC_19/19_4_调权重_结果还是原地踏步速度慢了/best_model.zip"
CUR = "train_humanoid_sac.py"
SEEDS = tuple(int(x) for x in os.environ.get("X_SEEDS", "7,31,55,123,2026").split(","))
FR = 1000


def mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def actor(path):
    """只装 actor，不装配 9.14 GB 的 replay buffer。"""
    import torch.nn as nn
    from stable_baselines3.sac.policies import SACPolicy
    with zipfile.ZipFile(path) as z:
        sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=False)
    hid = [int(sd[k].shape[0]) for k in ("actor.latent_pi.0.weight", "actor.latent_pi.2.weight") if k in sd]
    env = gym.make("Humanoid-v4")
    pol = SACPolicy(env.observation_space, env.action_space, lambda x: 1e-4,
                    net_arch=hid, activation_fn=nn.Tanh, use_sde=False)
    pol.load_state_dict(sd, strict=False)
    env.close()
    return pol.to("cpu").eval()


W17 = mod(CUR, "price17")
import shutil
V9_PY = "/tmp/v9mod.py"
_lines = open(V9_TXT, encoding="utf-8").read().splitlines()
_i = next(i for i, l in enumerate(_lines) if l.startswith("import "))      # 头两行是中文说明，不是代码
open(V9_PY, "w", encoding="utf-8").write("\n".join(_lines[_i:]) + "\n")
W9 = mod(V9_PY, "price9")
A9, A15 = actor(V9_W), actor(B15_W)
# T1 注入所需常量（从 ⑰ 模块的 wrapper 默认值取，不自己写死）
_probe = W17.WalkRewardWrapper(gym.make("Humanoid-v4"), swing_drive_weight=2.0, osc_enable=True)
OSC_A, OSC_SF, OSC_T, OSC_K = _probe.osc_A, _probe.osc_swing_frac, _probe.gait_period, _probe.osc_k_start
_m = _probe.unwrapped.model     # osc_aidx 是 step 里惰性创建的，这里直接从执行器名算
OSC_IDX = {s: [u for u in range(_m.nu) if _m.joint(_m.actuator_trnid[u][0]).name == f"{s}_hip_y"][0]
           for s in ("left", "right")}
assert OSC_IDX == ({"left": 9, "right": 5}), f"髋作用器索引与 ⑰ 内部不一致：{OSC_IDX}"
_probe.env.close()
print(f"T1 常量取自 ⑰ wrapper：A={OSC_A} swing_frac={OSC_SF:.4f} period={OSC_T} k={OSC_K} aidx={OSC_IDX}")


def t1_pulse(steps):
    """复刻 ⑰ 的 T1 时钟驱动（右腿晚半周期 ⇒ 结构上保证交替）。"""
    ph = (steps % OSC_T) / float(OSC_T)
    out = {}
    for s in ("left", "right"):
        off = 0.0 if s == "left" else 0.5
        p = (ph - off) % 1.0
        out[s] = (-OSC_A * np.sin(np.pi * p / OSC_SF), p < OSC_SF)
    return out


def roll(pol, Wmod, seed, patch, template, inject, replay=None):
    env = gym.make("Humanoid-v4", max_episode_steps=FR)
    if patch:
        W17.patch_mirror_symmetry(env.unwrapped.model)
    kw = dict(swing_drive_weight=2.0, osc_enable=template) if Wmod is W17 else {}
    w = Wmod.WalkRewardWrapper(env, **kw)
    w.reset(seed=seed)
    xs, ret, n = [], 0.0, 0
    ch = {}
    ctrls = []
    for t in range(FR):
        if replay is not None:
            a = np.asarray(replay[t], dtype=np.float64)      # 回放：直接用另一侧实际施加的 ctrl
        else:
            a, _ = pol.predict(w.unwrapped._get_obs(), deterministic=True)
            a = np.asarray(a, dtype=np.float64)
        if inject:                                    # V9 那一侧自己注入 T1
            pul = t1_pulse(n + 1)
            a = a.copy()
            for s, (v, on) in pul.items():
                if on:
                    a[OSC_IDX[s]] = float(np.clip((1 - OSC_K) * a[OSC_IDX[s]] + OSC_K * v, -1, 1))
        _, r, ter, tru, info = w.step(a.astype(np.float32))
        ctrls.append(w.unwrapped.data.ctrl.copy())           # 实际施加的控制量（模板注入之后）
        ret += float(r); n += 1
        xs.append(float(w.unwrapped.data.qpos[0]))
        for k in ("forward_reward", "forward_gauss_gated", "place_reward", "height_reward_gated",
                  "stride_bonus_gated", "phase_reward", "phase_reward_unpaid", "survival_bonus",
                  "survival_bonus_gated", "position_swap_reward", "stride_bonus", "flight_phase_penalty",
                  "short_stride_penalty", "lateral_penalty", "slip_penalty", "ctrl_cost", "hip_y_angle_penalty",
                  "hip_xz_angle_penalty", "fall_penalty", "alternate_penalty", "swing_forward_reward"):
            if k in info:
                ch[k] = ch.get(k, 0.0) + float(info.get(k, 0.0) or 0.0)
        if ter or tru:
            break
    return dict(ret=ret, n=n, x=np.array(xs), ch=ch, ctrls=ctrls)


def cell(name, pol, patch, template, inj17, inj9):
    # 先在 ⑰ 侧（模板由 wrapper 自己注入）跑一遍并记录实际施加的 ctrl，再用同一串 ctrl 回放给 V9 侧。
    # ⇒ 两侧轨迹由确定性物理保证相同，价格是唯一变量；控件 max|Δx| 用来证明这一点。
    r17 = [roll(pol, W17, s, patch, template, inj17) for s in SEEDS]
    r9 = [roll(pol, W9, s, patch, False, inj9, replay=b["ctrls"]) for s, b in zip(SEEDS, r17)]
    dx = max(float(np.abs(a["x"] - b["x"][:len(a["x"])]).max()) for a, b in zip(r9, r17))
    v9 = float(np.median([a["ret"] / max(a["n"], 1) * 1000 for a in r9]))
    v17 = float(np.median([b["ret"] / max(b["n"], 1) * 1000 for b in r17]))
    medn = float(np.median([a["n"] for a in r9]))
    print(f"\n--- {name} ---   回放控件 max|Δx轨迹| = {dx:.2e}  {'✓ 同一条轨迹' if dx < 1e-9 else '✘ 轨迹不同行，本行作废'}")
    print(f"    存活 {medn:.0f}/{FR} 帧")
    print(f"    V9 价格（16 项）  ret = {v9:+8.0f} 分/集   其中前进项 {np.median([a['ch'].get('forward_reward',np.nan) for a in r9])/max(medn,1)*1000:+7.0f}"
          f" = {abs(np.median([a['ch'].get('forward_reward',0) for a in r9]))/max(abs(v9),1)*100:4.1f}% 收入")
    print(f"    ⑰ 价格（22 项）  ret = {v17:+8.0f} 分/集   前进 {np.median([b['ch'].get('forward_gauss_gated',np.nan) for b in r17])/max(medn,1)*1000:+7.0f}"
          f"  place {np.median([b['ch'].get('place_reward',np.nan) for b in r17])/max(medn,1)*1000:+7.0f}")
    return v9, v17


print("\n=== 2×2：同一轨迹、两套价格 ===")
j9, j17 = cell("慢跑者 = V9 权重（原生接口：无模板、无对称修补）", A9, False, False, False, False)
m9, m17 = cell("高抬腿者 = ⑮ best 权重（原生接口：T1 模板开、对称修补开）", A15, True, True, False, True)
print(f"\n=== 结论 ===")
print(f"  V9 价格：慢跑 {j9:+.0f} vs 高抬腿 {m9:+.0f}  ⇒ 慢跑胜 {j9 > m9}  （P1）")
print(f"   价格：慢跑 {j17:+.0f} vs 高抬腿 {m17:+.0f}  ⇒ 高抬腿胜 {m17 >= j17}  （P2）")
print(f"  价格翻转幅度：慢跑−高抬腿 在 V9 价格下 {j9-m9:+.0f}，在 ⑰ 价格下 {j17-m17:+.0f}")
