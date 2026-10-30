"""det（eval 口径）vs stochastic（rollout 口径）的**同一条每帧台账**差在哪。

动机：我诊断 rollout 每帧率为负时说"最大一项是碎步税"，那只量了采样侧，没证明确定性策略不缴这笔钱。
本脚本用 verify_batch14 的同一套"实付"键对最新快照做 det rollout，与 TB 的 stochastic 末段并排。
只读权重、只读 rollout；不训练、不写文件。
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
sys.path.insert(0, "/home/guo/workspace/humanoid_walk/debug_probes")
os.chdir("/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

import verify_batch14 as V14
from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

N_EP = int(os.environ.get("G_EP", "5"))


def newest_snapshot():
    fs = [f for f in glob.glob("best_model/best_model_snapshots/*.zip")] or ["best_model/best_model.zip"]
    return max(fs, key=os.path.getmtime)


def newest_tb():
    fs = [f for f in glob.glob("sac_humanoid_walk/**/events.out.tfevents*", recursive=True)
          if os.path.getsize(f) > 1e6 and b"gait/swap_gate" in open(f, "rb").read()]
    return max(fs, key=os.path.getmtime) if fs else None


def det_ledger(path):
    model = SAC.load(path)
    raw = gym.make("Humanoid-v4", max_episode_steps=1500)
    patch_mirror_symmetry(raw.unwrapped.model)
    w = WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=True)
    keys = list(V14.SIGN["+"]) + list(V14.SIGN["-"])
    rows = []
    for i in range(N_EP):
        obs, _ = w.reset(seed=4242 + i)
        t, tot = 0, 0.0
        sums = {k: 0.0 for k in keys}
        while True:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = w.step(np.asarray(act, dtype=np.float64))
            t += 1
            tot += float(r)
            for k in keys:
                sums[k] += float(info.get(k, 0.0))
            if term or trunc:
                rows.append(dict(len=t, ret=tot, fell=bool(term), **{k: sums[k] / t for k in keys}))
                break
    return rows


def stoch_ledger(f):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(f, size_guidance={"scalars": 0})
    ea.Reload()
    tg = ea.Tags()["scalars"]

    def m(t, n=200):
        s = [x.value for x in ea.Scalars(t)][-n:] if t in tg else []
        return float(np.mean(s)) if s else np.nan
    out = {k: m("gait/" + k) for k in V14.SIGN["+"]}
    out.update({k: -m("gait/" + k) for k in V14.SIGN["-"]})
    return out, m("rollout/ep_len_mean"), m("rollout/ep_rew_mean")


def main():
    snap = newest_snapshot()
    rows = det_ledger(snap)
    L = np.mean([r["len"] for r in rows])
    pf = np.mean([r["ret"] / r["len"] for r in rows])
    print(f"快照 {snap}  det {N_EP} 集：ep_len 均值 {L:.0f}、摔倒 {sum(r['fell'] for r in rows)}/{N_EP}"
          f"  ret {[round(r['ret']) for r in rows]}  每帧率均值 {pf:+.3f}")

    s, sL, sR = (None, np.nan, np.nan)
    f = newest_tb()
    if f:
        s, sL, sR = stoch_ledger(f)
        print(f"TB stochastic 末段：ep_len {sL:.0f}、ep_rew {sR:.0f} ⇒ 每帧率 {sR / sL:+.3f}")
    print(f"\n{'项':26s} {'det 分/帧':>10} {'stoch 分/帧':>11} {'差(det−stoch)':>13}")
    keys = list(V14.SIGN["+"]) + list(V14.SIGN["-"])
    tot_d = tot_s = 0.0
    for k in keys:
        # 统一成"对 reward 的贡献"符号：罚项两侧都取负
        sgn = 1 if k in V14.SIGN["+"] else -1
        d = sgn * float(np.mean([r[k] for r in rows]))
        sv = sgn * abs(float(s.get(k, np.nan))) if (s and np.isfinite(s.get(k, np.nan))) else np.nan
        tot_d += d
        tot_s += sv
        tag = "  ← 碎步税" if k == "short_stride_penalty" else ("  ← 摔倒" if k == "fall_penalty" else "")
        print(f"  {k:26s} {d:+10.3f} {sv:+11.3f} {d - sv:+13.3f}{tag}")
    print(f"  {'合计（不含缺失键）':24s} {tot_d:+10.3f} {tot_s:+11.3f} {tot_d - tot_s:+13.3f}")
    print("  注：TB 未导出 gait/short_stride_penalty 与 activity_diff_paid ⇒ stoch 侧这两项是 NaN，"
          "合计差里含它们的缺口（我已记为待补遥测）。")


if __name__ == "__main__":
    main()
