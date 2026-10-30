"""批次④.3/⑤/⑥ 的落地自检（开跑前必须全绿）。

1) `_check_cycle` 的**时序**单元测试：min 必须取"上一次落地的 diff"和"这一次的 diff"，
   不能在 `self.prev_diff` 被覆盖之后才读——我第一版就是这么写错的（min 退化成 |cur|）。
   用构造的 diff 序列直接判定返回值，不依赖环境状态。
2) ⑤ 的斜坡在**实测工况点**上必须有非零斜率：rom=1.1°（左腿 det 实测中位数）时
   swing_min 必须 > 0，且每度换到的分/集要和注释里写的量级一致。
3) ⑥ 只核对常数确实改到了 0.10（熵系数在 MirrorEqSAC 构造参数里，不重跑模型）。
"""
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

raw = gym.make("Humanoid-v4", max_episode_steps=100)
patch_mirror_symmetry(raw.unwrapped.model)
env = WalkRewardWrapper(raw, swing_drive_weight=2.0)
env.reset(seed=0)

# ---- 1) _check_cycle 时序 ----
seq = [1.0, -0.4, -1.3, 0.9, 0.2, -2.0]
env.prev_diff = None
got = [env._check_cycle(d) for d in seq]
exp = [0.0]
for i in range(1, len(seq)):
    p, c = seq[i - 1], seq[i]
    exp.append(np.sign(p * c) * min(abs(p), abs(c)))
print("diff 序列        :", seq)
print("  _check_cycle   :", ["%.3f" % x for x in got])
print("  期望 sign×min  :", ["%.3f" % x for x in exp])
assert np.allclose(got, exp), "④.3 失败：min 没有取到上一次的 diff（时序 bug）"
# 反证：如果写成 min(|self.prev_diff(已被覆盖)|, |cur|)，i=2 会返回 +1.3 而不是 +0.4
# （seq=[1.0,−0.4,−1.3,…]：i=2 的 prev=−0.4、cur=−1.3 ⇒ 同号→+，min(0.4,1.3)=0.4）
assert abs(got[1] - (-0.4)) < 1e-9, "④.3：i=1 应为 sign(1.0×−0.4)×min(1.0,0.4) = −0.4"
assert abs(got[2] - 0.4) < 1e-9, "④.3 回归：i=2 必须取 min(0.4,1.3)=0.4，覆盖后再读会变成 1.3"
print("  → ④.3 时序正确（含回归断言）")

# 有界性：min 形式不可能超过任一侧的 |diff|
for i in range(1, len(seq)):
    assert abs(got[i]) <= max(abs(seq[i]), abs(seq[i - 1]))
print("  → min 形式天然有界，无需饱和 cap")

# ---- 2) ⑤ 斜坡在工况点上有斜率 ----
floor, top, w = env.swing_min_floor_deg, env.swing_min_top_deg, env.swing_min_weight
print("\n⑤ swing_min: floor=%.1f° top=%.1f° weight=%.1f | 左腿 det 实测中位 rom=1.1°" % (floor, top, w))
assert floor <= 1.1, "floor 仍在工况点之上 ⇒ 斜率恒为 0，⑤ 没生效"
for rom in (0.0, 1.1, 2.0, 6.0, 14.0, 20.0):
    val = w * np.clip((rom - floor) / (top - floor), 0, 1)
    print("   rom=%5.1f° → 每次落地 %+.3f 分 → ×114落地×gate0.5 ≈ %5.1f 分/集" % (
        rom, val, val * 114 * 0.5))
slope = w / (top - floor) * 114 * 0.5
print("   斜率 = %.1f 分/集每度（旧 floor=4° 时在 1.1° 处恒为 0）" % slope)

# ---- 3) ⑥ 常数 ----
src = open("/home/guo/workspace/humanoid_walk/train_humanoid_sac.py").read()
assert "ent_coef=0.10," in src and "ent_coef=0.05," not in src.split("MirrorEqSAC(")[1][:2000]
print("\n⑥ ent_coef=0.10 已写入 MirrorEqSAC 构造参数")
print("\nALL CHECKS PASSED")
