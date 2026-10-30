"""同一份权重、两把尺的对拍：证明 `gait/ev_rom_L_mean`（训练）比 det eval 高不是"策略变好了"。

背景（2026-10-11 用户问："日志里 ev_rom_L_mean 都在 5 多，你凭什么说 2-3？"）：
 · TB 的 `gait/ev_rom_L_mean` 来自**训练 vec env 的随机策略**（⑥ 之后 std≈0.24）
 · 判决用的是 `eval_logs/eval_gait.jsonl` 里**确定性策略**的同名量
这里加载**同一个 checkpoint**，只改 `predict(deterministic=)` 一个开关，量两个值。
如果 det 明显低、且随机值随 std 衰减而下降，差值就是探索噪声不是能力。

只读快照（best_model_snapshots/），不碰正被训练进程覆写的 best_model.zip。
"""
import glob
import re
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

HORIZON = 1500
SEEDS = (123, 2026)


def run2(path, det_flag):
    """加载 checkpoint，只切换 predict 的 deterministic 开关，量同一批观测量。"""
    raw = gym.make("Humanoid-v4", max_episode_steps=HORIZON)
    patch_mirror_symmetry(raw.unwrapped.model)
    env = WalkRewardWrapper(raw, swing_drive_weight=2.0)
    model = SAC.load(path, env=env, device="cpu")
    obs, _ = env.reset(seed=SEEDS[0])
    per_step, ev_last, rets, fl, sg = [], float("nan"), [], float("nan"), float("nan")
    for _ in range(HORIZON):
        act, _ = model.predict(obs, deterministic=det_flag)
        obs, rew, term, trunc, info = env.step(act)
        rets.append(float(rew))
        per_step.append(float(info.get("last_rom_L_deg", 0.0)))
        if "ev_rom_L_mean" in info:
            ev_last = float(info["ev_rom_L_mean"])
        fl, sg = float(info.get("flight_frac", fl)), float(info.get("single_frac", sg))
        if term or trunc:
            break
    return dict(ret=sum(rets), per_step=float(np.mean(per_step)), ev=ev_last,
                flight=fl, single=sg, steps=len(rets))


def episodes(path, det_flag, want=6, cap=8):
    """跑多条 episode（随机或 det），按条记录 (长度, 是否摔, 该集末的 ev_rom_L_mean)。
    目的：TB 那个量是**训练 vec env 跨整条 rollout 窗**的均值，而训练窗里大量集是摔掉的
    （ep_len_mean≈486/1500）。如果摔的集 ROM 明显高，那 TB 与 det 的差就**不只是动作噪声**，
    还混进了"倒地时腿被甩起来"这种根本不属于步态的行程。"""
    raw = gym.make("Humanoid-v4", max_episode_steps=HORIZON)
    patch_mirror_symmetry(raw.unwrapped.model)
    env = WalkRewardWrapper(raw, swing_drive_weight=2.0)
    model = SAC.load(path, env=env, device="cpu")
    obs, _ = env.reset(seed=SEEDS[0])
    eps, n, ev, ret = [], 0, float("nan"), 0.0
    for _ in range(cap * HORIZON):
        act, _ = model.predict(obs, deterministic=det_flag)
        obs, rew, term, trunc, info = env.step(act)
        n += 1
        ret += float(rew)
        if "ev_rom_L_mean" in info:
            ev = float(info["ev_rom_L_mean"])
        if term or trunc:
            eps.append(dict(len=n, fell=bool(term), ev_rom=ev, ret=ret))
            if len(eps) >= want:
                break
            obs, _ = env.reset()
            n, ev, ret = 0, float("nan"), 0.0
    del model, env, raw
    return eps


def report_split(eps, tag):
    f = [e for e in eps if e["fell"]]
    s = [e for e in eps if not e["fell"]]
    print("  %s：摔 %d 条(ROM %s) / 满 1500 %d 条(ROM %s)" % (
        tag, len(f), " ".join("%.1f" % e["ev_rom"] for e in f) or "—",
        len(s), " ".join("%.1f" % e["ev_rom"] for e in s) or "—"))
    if f and s:
        print("     ⇒ 摔的集比活着的集 ROM 高 %+.1f°；TB 窗均值里摔集占 %.0f%%" % (
            np.mean([e["ev_rom"] for e in f]) - np.mean([e["ev_rom"] for e in s]),
            100.0 * len(f) / len(eps)))


def main():
    cands = sorted(glob.glob("/home/guo/workspace/humanoid_walk/best_model/best_model_snapshots/"
                             "best_model_*.zip"),
                   key=lambda p: int(re.search(r"_(\d+)\.zip", p).group(1)))
    path = cands[-1]
    ts = int(re.search(r"_(\d+)\.zip", path).group(1))
    print("同一份权重：%s（num_timesteps=%d）\n" % (path.split("/")[-1], ts))
    print("%-14s %6s %10s %10s %8s %8s %8s" % (
        "predict", "步数", "rom每步均", "ev_rom末值", "ret", "flight", "single"))
    for flag, name in ((True, "det(判决用)"), (False, "随机(TB)")):
        r = run2(path, flag)
        print("%-14s %6d %10.2f %10.2f %8.0f %8.3f %8.3f" % (
            name, r["steps"], r["per_step"], r["ev"], r["ret"], r["flight"], r["single"]))
    print("\n按条 episode 拆开看（随机策略，6 条）：")
    report_split(episodes(path, False, want=6), "随机")
    print("\n读法：TB 的 ev_rom_L_mean 是随机策略跨整窗的均值，det 判决值来自跑满 1500 的确定性集。")
    print("     两者的差 = 动作噪声 + 摔倒在地的集被计入；两者都**不是**步态能力，判决只用 det。")


if __name__ == "__main__":
    main()
