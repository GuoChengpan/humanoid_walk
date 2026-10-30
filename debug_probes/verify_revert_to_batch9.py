"""回滚验证（②→⑨ 价格）：把 ⑨ 的权重放进**回滚后**的奖励里跑，必须复现 ⑨ 自己当年记下的通道值。

这是唯一能证明"撤销干净"的检查：不是看代码 diff，而是看同一份权重在同一套价格下的
收入结构是否与它训练时记录的一致（`eval_logs/eval_gait.jsonl` 里 ⑨ 那 555 行的末段）。
对不上就说明还有残留（少删/多删了某条通道）。
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

ROOT = "/home/guo/workspace/humanoid_walk/"
P9 = ROOT + "best_model/tmp_model_data/SAC_18/18_8_从头训/best_model_snapshots/best_model_1696000.zip"
HORIZON, SEEDS = 1500, (123, 2026)
CH = ["height_reward_gated", "survival_bonus_gated", "position_swap_gated", "stride_bonus",
      "reach_reward", "swing_min_reward", "forward_gauss_gated", "flight_phase_penalty",
      "alternate_penalty", "tiny_step_penalty", "advance_bonus", "stride_upright"]


def run(seed):
    raw = gym.make("Humanoid-v4", max_episode_steps=HORIZON)
    patch_mirror_symmetry(raw.unwrapped.model)
    env = WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=False)
    model = SAC.load(P9, env=env, device="cpu")
    obs, _ = env.reset(seed=seed)
    acc = {k: 0.0 for k in CH}
    n, tot = 0, 0.0
    info = {}
    for _ in range(HORIZON):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        n += 1
        tot += float(r)
        for k in CH:
            v = info.get(k)
            if v is not None:
                acc[k] += float(v)
        if term or trunc:
            break
    del model, env, raw
    return n, tot, {k: v * HORIZON / max(n, 1) for k, v in acc.items()}, info


def main():
    rows = [json.loads(l) for l in open(ROOT + "eval_logs/eval_gait.jsonl")][:555]
    late = rows[-25:]
    def mean(rs, k):
        v = [float(x[k]) for x in rs if k in x and x[k] is not None]
        return float(np.mean(v)) if v else float("nan")
    print("对照基准 = ⑨ 那轮 jsonl 末 25 集（它训练时用的就是这套价格）")
    print("  当年：ret %.0f｜ep_len %.0f｜ds %.3f｜flight %.3f｜single %.3f｜speed %.2f｜rom_L %.2f°" % (
        mean(late, "ret"), mean(late, "ep_len"), mean(late, "double_support_frac"),
        mean(late, "flight_frac"), mean(late, "single_frac"), mean(late, "speed"), mean(late, "ev_rom_L_mean")))
    ns, tots, chans, infos = [], [], [], []
    for s in SEEDS:
        n, tot, ch, info = run(s)
        ns.append(n); tots.append(tot); chans.append(ch); infos.append(info)
    m = lambda k: float(np.mean([c[k] for c in chans]))
    print("  回滚后复跑：ret %.0f｜ep_len %.0f｜ds %.3f｜flight %.3f｜speed %.2f｜rom_L %.2f°" % (
        float(np.mean(tots)), float(np.mean(ns)),
        float(np.mean([float(i.get("double_support_frac", np.nan)) for i in infos])),
        float(np.mean([float(i.get("flight_frac", np.nan)) for i in infos])),
        float(np.mean([float(i.get("speed", np.nan)) for i in infos])),
        float(np.mean([float(i.get("ev_rom_L_mean", np.nan)) for i in infos]))))
    print("\n收入结构（分/集，按 1500 步折算）：")
    for k in CH:
        print("  %-24s %8.0f %s" % (k, m(k), "  ← 应为 0（⑩ 残留键，若>0 说明没撤干净）" if k in
                                    ("tiny_step_penalty", "advance_bonus", "stride_upright") and m(k) != 0 else ""))
    dense = m("height_reward_gated") + m("survival_bonus_gated") + m("position_swap_gated")
    print("\n  dense 三项合计 %.0f 分/集（⑨ 当年 2257 → 应接近）" % dense)
    print("  判定：① 三个 ⑩ 键全为 0；② ret/ds/flight 与当年同量级 ⇒ 回滚等价于 ⑨ 的价格。")


if __name__ == "__main__":
    main()
