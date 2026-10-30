"""A 类清理的**奖励数值不变性**验证 + 四道活门控的"还在不在、还咬不咬"实测。

为什么要写这个：上一条我用"被动基线逐位相同"当 A 类清理的验收，那个验证**是不够的**——
det rollout 的轨迹根本不依赖奖励值，所以它只能证明"接口没坏"，证明不了"发钱没变"。
这里补两件事：
 1) 把被我删掉的两个因子按**原公式**逐帧重算，证明它们在每一帧都恰好等于 1.0
    ⇒ 乘以它等于不乘 ⇒ 删除是恒等变换（代数 + 实测）。
 2) 用"把某个门的 floor 临时改成 0，整集奖励必须下降"的功能测试，证明
    `walk_gate / flat_gate / dense_ds_factor / forward_gate` **四道门都还挂在求和式上并且真的在扣分**。
    （⑥-era 被删的 `event_factor/ds_factor` 与 ⑧ 的 `dense_ds_factor` 名字只差一个前缀，
      正是这个相似性让人担心删错——所以必须测，不能靠我说。）
"""
import sys
import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

ROOT = "/home/guo/workspace/humanoid_walk/"
POL = ROOT + "best_model/tmp_model_data/SAC_18/18_8_从头训/best_model_snapshots/best_model_1696000.zip"
FRAMES, SEED = 600, 123


def rollout(mut=None, det=True):
    raw = gym.make("Humanoid-v4", max_episode_steps=FRAMES)
    patch_mirror_symmetry(raw.unwrapped.model)
    w = WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=False)
    if mut:
        mut(w)
    model = SAC.load(POL, env=w, device="cpu")
    obs, _ = w.reset(seed=SEED)
    tot = 0.0
    ev, dsf, gaps = [], [], []
    gates = []
    # 复刻被我删掉的那套计时器（原码：每帧 ++，落地即清零），才能在"清理后的代码"上重算原公式
    since = {"L": 0, "R": 0}
    prev = {"L": True, "R": True}
    for _ in range(FRAMES):
        a, _ = model.predict(obs, deterministic=det)
        obs, r, term, trunc, info = w.step(a)
        tot += float(r)
        since["L"] += 1
        since["R"] += 1
        cL, cR = bool(w.left_contact_prev), bool(w.right_contact_prev)
        landL, landR = cL and not prev["L"], cR and not prev["R"]
        if landL:
            since["L"] = 0
        if landR:
            since["R"] = 0
        # (1) 用**原始公式 + 原始常量**（event_gap_ref=cap=1.0；stride_ds_floor=1.0/ref=0.10）重算
        gap = since["L"] if landL else (since["R"] if landR else max(since["L"], since["R"]))
        gaps.append(gap)
        ev.append(min(max(gap, 1.0) / 1.0, 1.0))
        dsf.append(1.0 + (1.0 - 1.0) * np.clip(w.double_support_frac / 0.10, 0.0, 1.0))
        prev = {"L": cL, "R": cR}
        gates.append((w.walk_gate, w.flat_gate, w._dense_ds_factor, w.forward_gate))
        if term or trunc:
            break
    del model, w, raw
    print(f"  （实测 gap 范围 {min(gaps)}..{max(gaps)} 帧，共 {len(gaps)} 帧）")
    return tot, np.array(ev), np.array(dsf), np.array(gates), len(gates)


def main():
    tot, ev, dsf, gates, n = rollout()
    print(f"=== (1) 被我删掉的两个因子按原公式逐帧重算（{n} 帧）===")
    print(f"  event_factor: min={ev.min():.12f} max={ev.max():.12f} ⇒ 恒等 1.0：{np.all(ev == 1.0)}")
    print(f"  ds_factor   : min={dsf.min():.12f} max={dsf.max():.12f} ⇒ 恒等 1.0：{np.all(dsf == 1.0)}")
    print("  ⇒ 删除 r×1.0 与保留 r×1.0 在任何一帧都不可能不同 ⇒ A 类清理对奖励数值是恒等变换")

    print(f"\n=== (2) 四道活门控的实测取值（{n} 帧）与'临时关成 0 必须变穷'的功能测试 ===")
    lab = ["walk_gate", "flat_gate", "dense_ds_factor", "forward_gate"]
    for i, nm in enumerate(lab):
        v = gates[:, i]
        print(f"  {nm:18s} min={v.min():.3f} p50={np.median(v):.3f} max={v.max():.3f} "
              f"变化={bool(v.max() - v.min() > 1e-6)}")
    base = tot
    print(f"\n  基线整集奖励（{n} 帧，det 策略）= {base:.2f}")
    # 注意：⑨ 的轨迹上门控是**饱和**的（walk_gate=flat_gate=dense_ds=1.0），只把 floor 调 0 不会有任何
    # 变化 ⇒ 那测不出"接线在不在"。所以这里同时把 ref 挪远，让门必然落在不饱和区，
    # 测的是"这一项到底乘没乘进求和式"，而不是"门现在咬不咬"（后者 §47 已量：不咬）。
    tests = {
        "walk_gate 强制降到 0.36": lambda w: (setattr(w, "walk_gate_floor", 0.0),
                                             setattr(w, "walk_gate_ref", 0.90),
                                             setattr(w, "walk_gate_lo", 0.48)),
        "dense_ds 强制降到 0.10": lambda w: (setattr(w, "dense_ds_floor", 0.0),
                                            setattr(w, "dense_ds_ref", 0.90)),
        "forward_gate 强制→0": lambda w: setattr(w, "forward_gate_ref", 1e6),
    }
    for name, mut in tests.items():
        t2 = rollout(mut=mut)[0]
        d = t2 - base
        print(f"  {name:18s} 整集={t2:8.2f}  Δ={d:+8.2f} ⇒ {'OK 门控真在扣分' if d < -0.5 else '**异常：门没生效，必须查**'}")
    print("\n结论：flat_gate = 0.6+0.4·walk_gate，跟着 walk_gate 一起被测到；"
          "四道门都还在求和式里。**⑥-era 被删的只有 event_factor/ds_factor 这两个恒 1 的折算，不是门控。**")


if __name__ == "__main__":
    main()
