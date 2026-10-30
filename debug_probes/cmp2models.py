"""对比两个模型：步态交替、支撑相、稳定性。串行 10 集，deterministic。"""
import os, sys, time, gc
sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from train_humanoid_sac import WalkRewardWrapper

MODELS = [
    ("576K 基线", "best_model/tmp_model_data/17_6/17_6_2.3+门控0.5/best_model_snapshots/best_model_576000.zip"),
    ("本轮 best 480K", "best_model/best_model.zip"),
]
N = 10
CONTACT_Z = 0.08
DIFF_SCALE = 0.18
TRACE_CAP = 40


def feet(data):
    l = data.body("left_foot").xpos
    r = data.body("right_foot").xpos
    return l[0], r[0], l[2], r[2]


def run_episode(model, seed):
    env = WalkRewardWrapper(gym.make("Humanoid-v4", max_episode_steps=1500))
    obs, _ = env.reset(seed=seed)
    data = env.unwrapped.data
    tot, info, steps = 0.0, {}, 0
    ds = ss = fl = 0
    prev_l = prev_r = True
    land_seps, trace_items = [], []
    n_landing = 0
    dp = de = 0.0
    t_all = time.perf_counter()
    while True:
        _t = time.perf_counter()
        act, _ = model.predict(obs, deterministic=True)
        dp += time.perf_counter() - _t
        _t = time.perf_counter()
        obs, r, term, trunc, info = env.step(np.asarray(act, dtype=np.float32))
        de += time.perf_counter() - _t
        tot += float(r)
        steps += 1
        lx, rx, lz, rz = feet(data)
        lc, rc = lz < CONTACT_Z, rz < CONTACT_Z
        if lc and rc:
            ds += 1
        elif lc ^ rc:
            ss += 1
        else:
            fl += 1
        if (lc and not prev_l) or (rc and not prev_r):
            n_landing += 1
            sep = abs(lx - rx)
            land_seps.append(sep)
            if len(trace_items) < TRACE_CAP:
                trace_items.append((sep, (rx - lx) / DIFF_SCALE,
                                    "L" if (lc and not prev_l) else "R"))
        prev_l, prev_r = lc, rc
        if term or trunc:
            break
    env.close()
    return {
        "reward": tot, "len": steps,
        "ds": ds / max(steps, 1), "ss": ss / max(steps, 1), "fl": fl / max(steps, 1),
        "n_land": n_landing,
        "sep_mean": float(np.mean(land_seps)) if land_seps else 0.0,
        "sep_std": float(np.std(land_seps)) if land_seps else 0.0,
        "sep_max": float(np.max(land_seps)) if land_seps else 0.0,
        "alt_r": info.get("alternate_reward_rate"), "alt_p": info.get("alternate_penalty_rate"),
        "neg": info.get("negative_stride_rate"), "short_r": info.get("short_stride_event_rate"),
        "strbr": info.get("stride_bonus_rate"), "strdl": info.get("stride_length"),
        "lsw": info.get("mean_left_swing"), "rsw": info.get("mean_right_swing"),
        "speed": info.get("speed"), "swap": info.get("position_swap_rate"),
        "dp": dp, "de": de, "all": time.perf_counter() - t_all,
    }, trace_items


def col(rows, k, dec=3):
    v = np.array([r[k] for r in rows if isinstance(r[k], (int, float))])
    return "mean=%.*f  std=%.*f" % (dec, v.mean(), dec, v.std()) if v.size else "n/a"


seed_env = WalkRewardWrapper(gym.make("Humanoid-v4", max_episode_steps=1500))
for label, path in MODELS:
    t0 = time.perf_counter()
    model = SAC.load(path, env=seed_env)
    del model.replay_buffer
    gc.collect()
    print("=" * 78)
    print("模型: %s   (num_timesteps=%d)" % (label, model.num_timesteps))
    print("=" * 78)

    rows, trace, trace_ep = [], [], None
    for ep in range(1, N + 1):
        r, ti = run_episode(model, seed=1000 + ep)
        rows.append(r)
        if trace_ep is None and len(ti) >= 30:
            trace, trace_ep = ti, ep

    print("\n--- 每集明细 ---")
    print("%-3s %9s %5s %6s %6s %6s %8s %8s %7s" % (
        "集", "reward", "len", "ds%", "ss%", "fl%", "alt_r", "间距均", "落地"))
    for i, rr in enumerate(rows, 1):
        print("%-3d %9.1f %5d %6.1f %6.1f %6.1f %8.3f %8.3f %7d" % (
            i, rr["reward"], rr["len"], rr["ds"] * 100, rr["ss"] * 100, rr["fl"] * 100,
            rr["alt_r"] or 0, rr["sep_mean"], rr["n_land"]))

    print("\n--- 10 集汇总 ---")
    for k, nm, dec in [("reward", "reward", 1), ("len", "ep_len", 0),
                       ("ds", "双支撑占比", 3), ("ss", "单支撑占比", 3), ("fl", "腾空占比", 3),
                       ("n_land", "落地次数", 0), ("sep_mean", "落地间距均值", 3),
                       ("sep_std", "落地间距std", 3), ("sep_max", "落地间距最大", 3),
                       ("alt_r", "alternate_reward_rate", 3), ("alt_p", "alternate_penalty_rate", 3),
                       ("neg", "negative_stride_rate", 3), ("short_r", "short_stride_rate", 3),
                       ("strbr", "stride_bonus_rate", 3), ("strdl", "stride_length", 3),
                       ("speed", "speed", 3), ("swap", "position_swap_rate", 4)]:
        print("  %-24s %s" % (nm, col(rows, k, dec)))
    L = np.array([r["lsw"] for r in rows if isinstance(r["lsw"], (int, float))])
    R = np.array([r["rsw"] for r in rows if isinstance(r["rsw"], (int, float))])
    if L.size:
        print("  %-24s %.4f / %.4f = 比值 %.4f  (1.0=完全对称)" % (
            "lsw / rsw", L.mean(), R.mean(), L.mean() / R.mean()))

    st = sum(r["len"] for r in rows)
    ta = sum(r["all"] for r in rows)
    print("\n--- 计时 ---")
    print("  %d 步 / %.2f s = %.4f ms/步   (env.step %.4f ms + predict %.4f ms)" % (
        st, ta, 1000 * ta / max(st, 1),
        1000 * sum(r["de"] for r in rows) / max(st, 1),
        1000 * sum(r["dp"] for r in rows) / max(st, 1)))

    print("\n--- 落地序列 (第 %s 集, 前 %d 次; diff=right-left /0.18) ---" % (
        trace_ep if trace_ep is not None else "无", len(trace)))
    print("  " + "  ".join("%+.2f%s" % (d, t) for _, d, t in trace))
    print("  符号: " + "".join("+-"[int(d >= 0)] for _, d, _ in trace))
    flips = sum(1 for i in range(1, len(trace)) if (trace[i][1] >= 0) != (trace[i - 1][1] >= 0))
    print("  符号翻转 %d / %d = %.0f%%" % (flips, max(len(trace) - 1, 1),
                                          100 * flips / max(len(trace) - 1, 1)))
    rl = [d for _, d, t in trace if t == "R"]
    ll = [d for _, d, t in trace if t == "L"]
    if rl:
        print("  R 脚落地的 diff: %.2f ~ %.2f (均值 %.2f)" % (min(rl), max(rl), np.mean(rl)))
    if ll:
        print("  L 脚落地的 diff: %.2f ~ %.2f (均值 %.2f)" % (min(ll), max(ll), np.mean(ll)))

    print("\n耗时 %.0f 秒\n" % (time.perf_counter() - t0))
    del model
    gc.collect()
