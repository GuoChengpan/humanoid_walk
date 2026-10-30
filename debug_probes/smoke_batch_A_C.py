"""批次A+C' smoke：跑 1 集，验证新代码路径不炸、事件统计有值、reach 奖励按预期归零。

预期：944K 弹跳策略的左 reach 中位 ≈ −0.16 < floor −0.10 → reach_reward 应接近 0
（旧策略下屈髋驱动项实测 +17/集）。若 reach_reward 明显>0 说明结算量或符号弄错。
"""
import glob
import json
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
from probe_left_leg_chain import build_env, load_model, MODEL  # noqa: E402
from train_humanoid_sac import WalkRewardWrapper  # noqa: F401,E402

LOG = "/home/guo/workspace/humanoid_walk/debug_probes/smoke_eval_gait.jsonl"


def main():
    vec = build_env()
    env = vec.envs[0]
    env.episode_log_path = LOG
    for f in glob.glob(LOG):
        import os
        os.remove(f)
    model = load_model(vec)
    obs = vec.reset()
    total = 0.0
    reach_sum = 0.0
    keys_seen = set()
    steps = 0
    for _ in range(1500):
        act, _ = model.predict(obs, deterministic=True)
        obs, rew, done, info = vec.step(act)
        total += float(rew[0])
        reach_sum += float(info[0].get("reach_reward", 0.0))
        keys_seen |= set(k for k in info[0] if k.startswith("ev_"))
        steps += 1
        if done[0]:
            break
    print(f"步数={steps}  episode 回报={total:.0f}  reach_reward 合计={reach_sum:.1f}")
    print(f"last_reach_L={env.last_reach_L:+.3f} last_reach_R={env.last_reach_R:+.3f} "
          f"floor={env.reach_floor} top={env.reach_top}")
    print(f"phase: flight={env.flight_frac:.3f} dbl_sup={env.double_support_frac:.3f} "
          f"stance_L={env.stance_frac_L:.3f} stance_R={env.stance_frac_R:.3f}")
    print(f"\nevent_stats ({len(env.event_stats)} keys):")
    for k in sorted(env.event_stats):
        print(f"  {k:26s} {env.event_stats[k]:8.3f}")
    print(f"\ninfo 里出现过的 ev_ 键数: {len(keys_seen)}")
    try:
        with open(LOG) as f:
            lines = f.readlines()
        rec = json.loads(lines[-1])
        print(f"[episode_log] 写入 {len(lines)} 行，末行 ep_len={rec['ep_len']} ret={rec['ret']:.0f} "
              f"reach_L_mean={rec.get('ev_reach_L_mean')} reach_R_mean={rec.get('ev_reach_R_mean')} "
              f"lift_L={rec.get('ev_lift_L_mean')} step_adv_L={rec.get('ev_step_adv_L_mean')}")
    except FileNotFoundError:
        print("[episode_log] 没写到文件 ← 异常")
    vec.close()


if __name__ == "__main__":
    main()
