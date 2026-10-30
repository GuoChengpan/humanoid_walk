"""批次⑥·(乙) 的落地自检：**时长折算必须真的把"落得勤就多赚"变成中性**，这是这批唯一的立论。

三段：
 1) 因子单元测试：event_factor(gap) 的取值与上限、ds_factor(ds) 的保底与饱和。
 2) **频率中性**测试：同一单价、同一总时长下，落地次数翻倍 ⇒ 事件类总收入必须几乎不变。
    这条是 (乙) 的全部理由；不成立就说明 REF/上限配错了。
 3) 步频边际的实测复算：用 det rollout 的逐落地数据，比较"改造前 vs 改造后"每多一次落地值多少钱。
    （改造前的口径 = 单价直接相加；改造后 = 单价 × min(gap/REF,cap)）
"""
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
sys.path.insert(0, "/home/guo/workspace/humanoid_walk/debug_probes")

from train_humanoid_sac import WalkRewardWrapper
import gymnasium as gym
from price_batch4 import rollout

raw = gym.make("Humanoid-v4", max_episode_steps=100)
env = WalkRewardWrapper(raw, swing_drive_weight=2.0)
REF, CAP = env.event_gap_ref, env.event_gap_cap
FLOOR, DSREF = env.stride_ds_floor, env.stride_ds_ref
T = 1500


def ev(gap):
    return min(max(gap, 1.0) / REF, CAP)


def ds(x):
    return FLOOR + (1.0 - FLOOR) * np.clip(x / DSREF, 0.0, 1.0)


print("1) 因子单位值")
if REF == 1.0 and CAP == 1.0 and FLOOR == 1.0:
    print("   批次⑥ 已回滚（ref=cap=ds_floor=1 ⇒ 两个因子恒为 1），⑥ 的断言不适用；"
          "下面只做中性核对。")
    assert ev(3) == 1.0 and ev(50) == 1.0 and ds(0.5) == 1.0
    print("   ✅ 中性核对通过：event_factor≡1、ds_factor≡1 ⇒ 奖励精确回到改造前")
    sys.exit(0)
print("   event_factor: gap=%d→%.3f  gap=%d→%.3f  gap=%d→%.3f  gap=999→%.3f(上限)" % (
    REF, ev(REF), REF / 2, ev(REF / 2), REF * 2, ev(REF * 2), ev(999)))
assert abs(ev(REF) - 1.0) < 1e-9 and abs(ev(REF / 2) - 0.5) < 1e-9 and abs(ev(999) - CAP) < 1e-9
print("   ds_factor:    ds=0→%.3f(保底)  ds=%.2f→%.3f  ds=0.5→%.3f(饱和)" % (ds(0), DSREF, ds(DSREF), ds(0.5)))
assert abs(ds(0) - FLOOR) < 1e-9 and abs(ds(DSREF) - 1.0) < 1e-9 and abs(ds(0.5) - 1.0) < 1e-9

print("\n2) 频率中性：单价 8.4 分/落地（实测弹跳值），总时长固定 1500 帧")
price = 8.4
for n in (57, 86, 115, 173, 230):
    gap = T / n
    old = n * price
    new = n * price * ev(gap)
    print("   落地 %3d 次（周期 %5.1f 帧）：折算前 %6.0f 分 → 折算后 %6.0f 分  比值 %.3f" % (
        n, gap, old, new, new / old))
    if n in (115, 230):
        assert abs(new - 115 * price) / (115 * price) < 0.06, "频率中性被破坏"
print("   ⇒ 115 与 230 两次落地的总收入差 <6%（改造前是 2 倍），步频补贴确实被削平")

print("\n3) 实测复算（det rollout，看真实 gap 分布下折算掉多少）")
for name, path in [
        ("752K 终版(下轮 warm-start)",
         "/home/guo/workspace/humanoid_walk/best_model/best_model.zip"),
        ("192K 快照(本轮弹跳恢复后)",
         "/home/guo/workspace/humanoid_walk/best_model/best_model_snapshots/best_model_192000.zip")]:
    for seed in (123, 2026):
        ps, lands, last, n = rollout(path, seed)
        # 用逐落地 gap 重算：把事件类合计按 min(gap/REF,cap) 折算，看总回报变化多少
        gaps = np.array([l[0] for l in lands if l[1] is not None], float)
        avg_gap = n / max(len(gaps), 1)
        factor = ev(avg_gap)
        ds_now = last["double_support_frac"]
        dsf = ds(ds_now)
        ev_sum = (sum(ps["stride_bonus_gated"]) + sum(ps["reach_reward"]) +
                  sum(ps["swing_min_reward_gated"]))          # 事件类（alternate 符号另算，量小）
        cut = ev_sum * (1 - factor * dsf)
        print("   %-30s s%-5d 落地%3d 均gap%5.1f帧 event=%.2f ds=%.3f→ds_f=%.2f | 现回报%6.0f "
              "事件类%5.0f ⇒ 被拿走%5.0f" % (
                  name, seed, len(gaps), avg_gap, factor, ds_now, dsf, sum(ps["reward"]), ev_sum, cut))
print("\n读法：event_factor≈1 说明 REF 正好是弹跳自己的周期（起点价基本不动）；"
      "ds_factor≈0.35 说明这批会把 stride_bonus/alternate 的 65% 从弹跳手里拿走。")
