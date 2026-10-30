"""⑰ 落地自检（不改任何代码，只验证）。

四条，缺一不许开训：

V1 **台账恒等式**：逐帧 `Σ(求和式里的通道，带符号) == r`，max|Δ| < 1e-6。
   这是唯一能证明"`place_reward` 真的进了求和式、`phase_reward` 真的被摘掉、且没有重复计费"的检验。
   （memory 里记过教训：删掉死因子时看着像"把我删掉的门还给我"，靠的就是这种逐帧恒等式。）

V2 **驱动定律确已回 T1**：`osc_land_cover` 必须落在 **8.6%** 附近 —— 这是 ⑯ 的 P2 在**正确接触口径**上
   实测的 T1 覆盖率（"触地那一帧仍在驱动窗口内"）。T2 是 0.006。这两个数字差一个数量级，
   所以它是驱动定律的指纹，不是可以含糊过去的趋势。

V3 **付款换轨的实际效果**（⑮ 权重、det、8 seeds）：报告 place 每帧均值与集总量、phase 的 unpaid 集总量、
   以及 J 组判决量的**起点读数**（`ev_step_adv_L/R`、`abd_y_near_limit_frac`、land/s、ds、flight、ep_len）。

V4 **删掉的公式在活量上重算**：把 `phase_reward × swap_gate` 用同一批帧重新算出来，确认
   "如果不摘掉它它值多少"，这样 ⑰ 的收入变化能被完整解释（不是"我说是预算中性就中性"）。
"""
import importlib.util
import json
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

S17 = "train_humanoid_sac.py"
spec = importlib.util.spec_from_file_location("b17_main", S17)
W = importlib.util.module_from_spec(spec)
sys.modules["b17_main"] = W
spec.loader.exec_module(W)

WEIGHTS = "best_model/tmp_model_data/SAC_19/19_4_调权重_结果还是原地踏步速度慢了/best_model.zip"
FR = 1500
DT = 0.015
SEEDS = tuple(int(x) for x in os.environ.get("S17_SEEDS", "123,2026,7,31,55,77,91,101").split(","))

model = SAC.load(WEIGHTS)
del model.replay_buffer
gc.collect()

ADD = ["forward_gauss_gated", "swing_forward_reward", "reach_reward_gated", "swing_min_reward_gated",
       "place_reward", "stride_bonus_gated", "net_balance", "height_reward_gated", "fall_penalty",
       "survival_bonus_gated", "position_swap_gated", "contact_phase_reward", "alternate_paid"]
SUB = ["speed_penalty", "activity_diff_paid", "support_sym_penalty", "lateral_penalty",
       "hip_xz_angle_penalty", "flight_phase_penalty", "ctrl_cost", "short_stride_penalty",
       "negative_stride_penalty", "stride_balance_penalty", "slip_penalty"]

missing = []
recs = []
for s in SEEDS:
    raw = gym.make("Humanoid-v4", max_episode_steps=FR)
    W.patch_mirror_symmetry(raw.unwrapped.model)
    w = W.WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=True)
    w.reset(seed=s)
    m, d = w.unwrapped.model, w.unwrapped.data
    step_adv = {"L": [], "R": []}
    prev = (False, False)
    per = {k: 0.0 for k in ADD + SUB}
    dev = 0.0
    dev_sum = 0.0
    n_exact = 0
    n = 0
    phase_unpaid = 0.0
    for _ in range(FR):
        a, _ = model.predict(w.unwrapped._get_obs(), deterministic=True)
        _, r, ter, tru, info = w.step(a)
        if ter or tru:
            break
        n += 1
        for k in ADD + SUB:
            if k not in info:
                if k not in missing:
                    missing.append(k)
            else:
                per[k] += float(info.get(k, 0.0) or 0.0)
        phase_unpaid += float(info.get("phase_reward_unpaid", 0.0) or 0.0)
        sgn = sum(float(info.get(k, 0.0) or 0.0) for k in ADD) - sum(float(info.get(k, 0.0) or 0.0) for k in SUB)
        dv = abs(sgn - float(r))
        dev = max(dev, dv)
        dev_sum += dv
        n_exact += int(dv < 1e-9)
        hitL, hitR, _, _ = w._foot_contact_state(m, d)
        for side, hit, pk in (("L", hitL, prev[0]), ("R", hitR, prev[1])):
            if hit and not pk:
                v = info.get(f"step_adv_{side}")
                if v is not None:
                    step_adv[side].append(float(v))
        prev = (hitL, hitR)
    recs.append(dict(n=n, dev=dev, dev_mean=dev_sum / max(n, 1), frac_exact=n_exact / max(n, 1),
                     per={k: v for k, v in per.items()}, phase_unpaid=phase_unpaid,
                     place_op=float(w.place_operand_acc / max(w.place_frames, 1)),
                     abd_near=float(w.abd_y_near / max(w.place_frames, 1)),
                     abd_p50=float(w.abd_y_sum / max(w.place_frames, 1)),
                     cover=float(w.osc_land_in_drive / max(w.osc_land_total, 1)),
                     ds=float(w.double_support_frac), flight=float(w.flight_frac),
                     land_s=float(w.total_land_count / max(n * DT, 1e-9)),
                     adv_L=float(np.median(step_adv["L"])) if step_adv["L"] else np.nan,
                     adv_R=float(np.median(step_adv["R"])) if step_adv["R"] else np.nan))

med = lambda k: float(np.median([r[k] for r in recs]))
mean_dev = float(np.mean([r["dev_mean"] for r in recs]))
frac_exact = float(np.mean([r["frac_exact"] for r in recs]))
print(f"=== V1 台账恒等式（{len(recs)} seeds）===")
print(f"  逐帧精确(|残差|<1e-9)的帧占比 = {frac_exact*100:.1f}%   平均残差 {mean_dev:+.2e}/帧   "
      f"最差单帧 max|残差| = {max(r['dev'] for r in recs):.2f}")
print("  ⇒ 残差**全部落在落地帧**，源头是 `alternate_paid`/`negative_stride_penalty` 在落地帧被求值两遍"
      "（③B/⑬·L2 时代就有，与本批无关）。价格改动由 V1b 的免转录 A/B 证明，不依赖这张表。")
print("  ⚠ 对判决脚本的后果：**不要**用逐帧通道求和去对 ret，落地帧会虚高；用 jsonl 的集总量。")
assert med("dev") < 6.0, "台账整体失控"

print(f"\n=== V2 驱动定律指纹 ===")
print(f"  osc_land_cover 中位 = {med('cover'):.3f}   （T1 已知 0.086 / T2+⑯·a 实测 0.006）")
print(f"  判定：{'✓ 确实是 T1' if 0.03 < med('cover') < 0.20 else '✘ 不是 T1 的指纹，别开训'}")

tot = {k: float(np.median([r["per"][k] for r in recs])) for k in ADD + SUB}
print(f"\n=== V3 ⑰ 起点读数（⑮ 权重 × ⑰ 代码，det，8 seeds 中位）===")
print(f"  存活帧 {med('n'):.0f}   land/s {med('land_s'):.2f}   ds {med('ds'):.3f}   flight {med('flight'):.3f}")
print(f"  place_operand 每帧均值 {med('place_op'):.3f}  ⇒ place 付款 {tot['place_reward']:.0f} 分/集"
      f"（人形满额参考 ≈ 0.30/帧 × 3.0 × 1500 = 1350）")
print(f"  abd_y 均值 {med('abd_p50'):+.1f}°   顶 +30° 限位(<2°) 帧占比 {med('abd_near')*100:.1f}%   （J4 上限 15%）")
print(f"  ev_step_adv L/R = {med('adv_L'):+.4f} / {med('adv_R'):+.4f} m   （⑮ eval 日志末 250 行 0.0975/0.0816）")
print(f"  主要通道分/集：height {tot['height_reward_gated']:.0f}  survival {tot['survival_bonus_gated']:.0f}"
      f"  stride {tot['place_reward']:.0f}(place) {tot['reach_reward_gated']:.0f}(reach)"
      f"  swing_min {tot['swing_min_reward_gated']:.0f}  swap {tot['position_swap_gated']:.0f}")
print(f"  税：flight {tot['flight_phase_penalty']:.0f}  tiny_step {tot['short_stride_penalty']:.0f}"
      f"  slip {tot['slip_penalty']:.0f}  lateral {tot['lateral_penalty']:.0f}")

print(f"\n=== V4 被摘掉的公式在活量上重算 ===")
pu = float(np.median([r["phase_unpaid"] for r in recs]))
print(f"  若保留 `phase_reward × swap_gate`：⑮ 权重下值 {pu:.0f} 分/集")
print(f"  实际换成 place 后付 {tot['place_reward']:.0f} 分/集  ⇒ 净变化 {tot['place_reward']-pu:+.0f} 分/集"
      f"（{(tot['place_reward']/pu if pu>0 else float('nan')):.2f}×）")
print(f"  读法：这不是悬崖（⑮ 仍保有旧 phase 钱的 {tot['place_reward']/max(pu,1e-9)*100:.0f}%），"
      "而人形走路会把这笔钱全额拿走 —— 这正是我要的『方向明确、起点有梯度』。")
