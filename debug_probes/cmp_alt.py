"""比较 576K 基线 vs 当前 best(560K) 的交替指标，逐集口径。

关键点：按集累加 lead_credit，而不是读 tensorboard 的 per-step 均值，
用来判断 per-step 日志是否把事件型 reward 稀释掉了。
buffer_size=1000 —— zip 内不含 replay_buffer，避免默认 9GB 分配挤爆训练进程的内存。
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
CZ, SCALE, CAP, ADV_MIN = 0.08, 0.18, 1.5, 0.05
N_EP, SEEDS = 6, [7, 11, 23, 31, 42, 55]


def run(path, det, tag):
    m = SAC.load(path, buffer_size=20000)
    rows = []
    for ep, sd in enumerate(SEEDS, 1):
        env = WalkRewardWrapper(gym.make("Humanoid-v4", max_episode_steps=1500))
        obs, _ = env.reset(seed=sd)
        data = env.unwrapped.data
        rew = lc = 0.0
        n_step = n_land = n_neg = n_neg_gate = 0
        leads = []
        prev_l = prev_r = True
        swap = stall = 0.0
        while True:
            a, _ = m.predict(obs, deterministic=det)
            obs, r, te, tr, info = env.step(np.asarray(a, dtype=np.float32))
            rew += float(r)
            n_step += 1
            c = float(info.get("lead_credit", 0.0))
            lc += c
            lx, rx = data.body("left_foot").xpos[0], data.body("right_foot").xpos[0]
            lc_ = data.body("left_foot").xpos[2] < CZ
            rc_ = data.body("right_foot").xpos[2] < CZ
            if (lc_ and not prev_l) or (rc_ and not prev_r):
                n_land += 1
                leads.append(lx - rx)
                d = (rx - lx) / SCALE
                ba = float(info.get("body_advance", 0.0))
                if d < 0:
                    n_neg += 1
                    if ba > ADV_MIN:
                        n_neg_gate += 1
            prev_l, prev_r = lc_, rc_
            swap = float(info.get("position_swap_rate", 0.0))
            stall = float(info.get("stall_max", 0.0))
            if te or tr:
                break
        env.close()
        rows.append(dict(rew=rew, lc=lc, n=n_land, nn=n_neg, nng=n_neg_gate,
                         swap=swap, stall=stall, nstep=n_step, leads=leads))
    m.replay_buffer = None
    del m
    gc.collect()

    R = np.array([r["rew"] for r in rows])
    L = np.array([r["lc"] for r in rows])
    N = np.array([r["n"] for r in rows])
    NN = np.array([r["nn"] for r in rows])
    NG = np.array([r["nng"] for r in rows])
    NS = np.array([r["nstep"] for r in rows])
    leads = np.concatenate([r["leads"] for r in rows]) if any(r["leads"] for r in rows) else np.array([np.nan])

    print("\n### %s  det=%s  (%d 集)" % (tag, det, N_EP))
    for i, r in enumerate(rows):
        print("  集%d seed=%d reward=%8.1f  lc=%7.2f  落地=%3d  负diff=%3d(过门=%3d)  swap_rate=%.4f  stall_max=%.0f  步=%d"
              % (i + 1, SEEDS[i], r["rew"], r["lc"], r["n"], r["nn"], r["nng"], r["swap"], r["stall"], r["nstep"]))
    print("  ---- 汇总")
    print("  reward      均值 %8.1f  范围 %7.1f ~ %7.1f" % (R.mean(), R.min(), R.max()))
    print("  lead_credit 均值 %8.2f/集   per-step 均值 %.5f   (lc/步数 = lc/集 / %.0f 步)"
          % (L.mean(), L.sum() / NS.sum(), NS.mean()))
    print("  落地数      均值 %8.1f/集   每 %.1f 步一次" % (N.mean(), NS.mean() / max(N.mean(), 1)))
    print("  负diff落地  %d/%d (%.1f%%)  其中过 body_advance 门 %d" % (NN.sum(), N.sum(),
                                                     100 * NN.sum() / max(N.sum(), 1), NG.sum()))
    print("  落地时 left-right 间距: 均值 %+.3f m  最大 %.3f m  最小 %.3f m  (n=%d)"
          % (leads.mean(), leads.max(), leads.min(), len(leads)))
    print("  左脚领先 >5cm 落地 %3d 次 (%.1f%%),  >10cm %3d 次 (%.1f%%)"
          % ((leads > .05).sum(), 100 * (leads > .05).mean(),
             (leads > .10).sum(), 100 * (leads > .10).mean()))
    return L.mean(), R.mean()


b_lc, b_rw = run(BASE, True, "576K 基线")
c_lc, c_rw = run(CURR, True, "当前 best 560K")
run(CURR, False, "当前 best 560K（随机策略 = 训练日志口径）")
print("\n=== 对比（deterministic）===")
print("lead_credit/集: 576K %7.2f  →  560K %7.2f  (%s)" % (b_lc, c_lc, "涨" if c_lc > b_lc else "跌"))
print("reward/集:      576K %8.1f  →  560K %8.1f  (%s)" % (b_rw, c_rw, "涨" if c_rw > b_rw else "跌"))
