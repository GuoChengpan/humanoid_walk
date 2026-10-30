"""真交替步态的硬指标：落地脚的时序，不是 reward 占比。

reward 占比会说「交替信号很强」，但真正的交替是 L-R-L-R 落地序列。
本探针直接记录每次落地是哪只脚，算严格交替率与最长同侧连续数。
另外量「站着不动」在现行奖励下能拿多少分——判断站立静止局部最优是否可越。
"""
import os, sys, gc
sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from train_humanoid_sac import WalkRewardWrapper

BASE = "best_model/tmp_model_data/17_6/17_6_2.3+门控0.5/best_model_snapshots/best_model_576000.zip"
CURR = "best_model/best_model.zip"
CZ, ADV_MIN = 0.08, 0.05
SEEDS = [7, 11, 23, 31, 42, 55]


def run(path, tag, det=True):
    m = SAC.load(path, buffer_size=20000)
    rows = []
    for sd in SEEDS:
        env = WalkRewardWrapper(gym.make("Humanoid-v4", max_episode_steps=1500))
        obs, _ = env.reset(seed=sd)
        data = env.unwrapped.data
        rew = stride = lc = alt = nsp = 0.0
        seq = []                       # 落地脚序列 'L'/'R'
        prev_l = prev_r = True
        n_step = 0
        while True:
            a, _ = m.predict(obs, deterministic=det)
            obs, r, te, tr, info = env.step(np.asarray(a, dtype=np.float32))
            rew += float(r)
            stride += float(info.get("stride_bonus", 0.0))
            lc += float(info.get("lead_credit", 0.0))
            alt += float(info.get("alternate_penalty", 0.0))
            nsp += float(info.get("negative_stride_penalty", 0.0))
            n_step += 1
            lc_ = data.body("left_foot").xpos[2] < CZ
            rc_ = data.body("right_foot").xpos[2] < CZ
            if lc_ and not prev_l:
                seq.append("L")
            if rc_ and not prev_r:
                seq.append("R")
            prev_l, prev_r = lc_, rc_
            if te or tr:
                break
        env.close()
        rows.append(dict(rew=rew, stride=stride, lc=lc, alt=alt, nsp=nsp,
                         seq=seq, nstep=n_step))
    m.replay_buffer = None
    del m
    gc.collect()

    print("\n### %s  (%d 集, det=%s)" % (tag, len(SEEDS), det))
    alt_rate, runs, n_land, n_switch = [], [], 0, 0
    for sd, r in zip(SEEDS, rows):
        s = r["seq"]
        alt_rate.append(1.0 if len(s) < 2 else sum(1 for a, b in zip(s, s[1:]) if a != b) / (len(s) - 1))
        if s:
            cur, best = 1, 1
            for a, b in zip(s, s[1:]):
                cur = cur + 1 if a == b else 1
                best = max(best, cur)
            runs.append(best)
            n_switch += sum(1 for a, b in zip(s, s[1:]) if a != b)
            n_land += len(s)
        print("  seed=%-3d 步=%4d 落地=%3d  reward=%7.1f  stride_bonus=%6.1f  lead_credit=%6.1f  序列头=%s"
              % (sd, r["nstep"], len(s), r["rew"], r["stride"], r["lc"], "".join(s[:14])))
    S = np.array([r["rew"] for r in rows])
    print("  ---- 汇总")
    print("  reward      均值 %8.1f  (范围 %.1f ~ %.1f)" % (S.mean(), S.min(), S.max()))
    print("  落地总数    %d   左右切换 %d 次   →  严格交替率 %.1f%%" % (n_land, n_switch, 100 * n_switch / max(n_land, 1)))
    print("  逐集严格交替率 均值 %.1f%%   最长同侧连续 均值 %.2f 次" % (100 * np.mean(alt_rate), np.mean(runs)))
    alt_sig = float(np.mean([r["stride"] + r["lc"] - r["nsp"] + (-r["alt"]) for r in rows]))
    print("  交替相关奖励合计 %.1f/集   占总 reward %.1f%%"
          % (alt_sig, 100 * alt_sig / max(S.mean(), 1)))
    print("  交替相关项拆解: stride_bonus %.1f  lead_credit %.1f  alternate_penalty %+.1f  negative_stride %.1f"
          % (np.mean([r["stride"] for r in rows]), np.mean([r["lc"] for r in rows]),
             np.mean([r["alt"] for r in rows]), np.mean([r["nsp"] for r in rows])))
    return np.mean(alt_rate), S.mean(), alt_sig


def stand_still_probe():
    """零动作（或极小动作）站立，看现行奖励给多少分。"""
    print("\n### 站立静止局部最优是否可越（17 维动作为 0，6 集）")
    for sd in SEEDS[:4]:
        env = WalkRewardWrapper(gym.make("Humanoid-v4", max_episode_steps=1500))
        obs, _ = env.reset(seed=sd)
        rew = 0.0
        n = 0
        while True:
            obs, r, te, tr, info = env.step(np.zeros(17, dtype=np.float32))
            rew += float(r)
            n += 1
            if te or tr:
                break
        env.close()
        print("  seed=%-3d 步=%4d  reward=%8.1f" % (sd, n, rew))


print("动作空间 ±%.2f" % 0.4)
a576, r576, s576 = run(BASE, "576K 基线")
a560, r560, s560 = run(CURR, "当前 best 560K")
stand_still_probe()

print("\n=== 结论 ===")
print("严格交替率: 576K %.1f%%  vs  560K %.1f%%" % (100 * a576, 100 * a560))
print("交替相关奖励: 576K %.1f/集  vs  560K %.1f/集" % (s576, s560))
print("reward:       576K %.1f/集  vs  560K %.1f/集" % (r576, r560))
