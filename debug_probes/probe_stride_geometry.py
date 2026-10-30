"""为什么步幅这么小？把"步幅"拆成三个能分别测量的量（只读，加载 ⑨ 的 1120K 快照）。

三种互斥的解释，各自的可测量预测：
  (H1) **几何限制**：脚能往前放多远 = 腿长 × 髋角行程。若实测步幅 ≈ 0.9-1.0×几何上限 ⇒ 步幅被摆幅卡住，
       要大步必须先有大 ROM（K2 那条判据就是这个意思）。
  (H2) **价格没指对**：大步能拿钱但没拿。判法：直接算 `stride_bonus` 在"0.3 m 步幅"下的定价 vs 现步幅。
       （我 20 分钟前在 §38.2 给的就是这个解释，但没算过账——这里算。）
  (H3) **仪表假象**：接触抖动把"落地次数"数多了，于是每次落地的位移看起来很小。判法：落地间隔直方图
       + 去抖（<5 帧算同一次）后的步幅/步频对比。
"""
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

PATH = "/home/guo/workspace/humanoid_walk/best_model/best_model_snapshots/best_model_1120000.zip"
HORIZON, SEEDS = 1500, (123, 2026)
DT = 0.02


def run(seed):
    raw = gym.make("Humanoid-v4", max_episode_steps=HORIZON)
    patch_mirror_symmetry(raw.unwrapped.model)
    env = WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=False)
    model = SAC.load(PATH, env=env, device="cpu")
    u = env.unwrapped      # 物理 data 在底层 env 上
    w = env                # 接触状态在 wrapper 上（env.unwrapped 会跳过它）
    obs, _ = env.reset(seed=seed)
    rec = {"tL": [], "tR": [], "bx": [], "fx": {"L": [], "R": []}, "stance_slip": {"L": [], "R": []}}
    n, reward_sum = 0, 0.0
    prev = {"L": (None, None), "R": (None, None)}
    for _ in range(HORIZON):
        act, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(act)
        n += 1
        reward_sum += float(r)
        # wrapper 在 step 末尾把本帧接触存进 *_contact_prev，所以这里读到的就是"这一帧站着没"
        cL, cR = bool(w.left_contact_prev), bool(w.right_contact_prev)
        bx = float(u.data.qpos[0])
        rec["tL"].append(cL); rec["tR"].append(cR); rec["bx"].append(bx)
        for side, bid in (("L", "left_foot"), ("R", "right_foot")):
            rec["fx"][side].append(float(u.data.body(bid).xpos[0]))
        for side, c in (("L", cL), ("R", cR)):
            fx = rec["fx"][side][-1]
            if c:
                if prev[side][0] is None:
                    prev[side] = (fx, bx)
                else:
                    pass
            else:
                if prev[side][0] is not None:
                    rec["stance_slip"][side].append(abs(fx - prev[side][0]))
                    prev[side] = (None, None)
        if term or trunc:
            break
    del model, env, raw
    return n, reward_sum, rec, info


def landings(contact_seq):
    c = np.asarray(contact_seq, dtype=bool)
    edges = np.where((~c[:-1]) & c[1:])[0] + 1
    return edges


def dedup(edges, min_gap=5):
    out = []
    for e in edges:
        if out and e - out[-1] < min_gap:
            continue
        out.append(e)
    return np.asarray(out, dtype=int)


def main():
    n, rs, rec, info = run(123)
    bx = np.asarray(rec["bx"], float)
    eL, eR = landings(rec["tL"]), landings(rec["tR"])
    dL, dR = dedup(eL), dedup(eR)
    both = np.sort(np.concatenate([dL, dR])) if len(dL) + len(dR) else np.array([], int)
    gaps = np.diff(np.sort(np.concatenate([eL, eR]))) if len(eL) + len(eR) else np.array([])
    print("权重 ⑨ 1120K｜det 种子 123｜本次存活 %d 步｜回报 %.0f" % (n, rs))
    print("\n=== H3：落地次数是不是被接触抖动灌水 ===")
    print("  原始落地 L %d + R %d = %d 次（%.2f 次/秒）" % (len(eL), len(eR), len(eL) + len(eR), (len(eL) + len(eR)) * DT / n))
    print("  去抖(≥5 帧=0.1 s)后 L %d + R %d = %d 次（%.2f 次/秒）" % (len(dL), len(dR), len(both), len(both) * DT / n))
    if len(gaps):
        print("  相邻落地间隔帧数：p10 %.0f p50 %.0f p90 %.0f｜间隔<5 帧的占比 %.0f%%  ⇒ %s" % (
            np.percentile(gaps, 10), np.percentile(gaps, 50), np.percentile(gaps, 90),
            100.0 * (gaps < 5).mean(), "**存在抖动**" if (gaps < 5).mean() > 0.2 else "无抖动，计数可信"))

    def strides(edges_side):
        e = edges_side[np.argsort(edges_side)]
        return np.diff(bx[e]) if len(e) > 1 else np.array([])
    sL_raw, sR_raw = strides(eL), strides(eR)
    sL_dd, sR_dd = strides(dL), strides(dR)
    print("\n=== 实测步幅（同脚相邻落地之间的躯干位移）===")
    if len(sL_raw):
        print("  左：原始 mean %.3f m｜去抖 mean %.3f m   右：原始 mean %.3f m｜去抖 mean %.3f m" % (
            sL_raw.mean(), sL_dd.mean() if len(sL_dd) else float("nan"),
            sR_raw.mean(), sR_dd.mean() if len(sR_dd) else float("nan")))
    net = bx[-1] - bx[0]
    print("  净位移 %.2f m / %.1f 秒 ⇒ v_net = %.2f m/s；TB 口径的速度 = %.2f m/s（两者差 = 来回滑移）" % (
        net, n * DT, net / (n * DT), float(info.get("speed", np.nan))))
    print("  脚掌在支撑期的打滑 |Δx|：左 %.3f m、右 %.3f m（p50）" % (
        np.median(rec["stance_slip"]["L"]) if rec["stance_slip"]["L"] else float("nan"),
        np.median(rec["stance_slip"]["R"]) if rec["stance_slip"]["R"] else float("nan")))

    romL = float(info.get("ev_rom_L_mean", np.nan)); romR = float(info.get("ev_rom_R_mean", np.nan))
    # 腿长：髋(左 thigh 起点)到脚的几何距离，直接用 xpos 差
    print("\n=== H1：几何上限（步幅 ≤ 腿长 × 髋角行程）===")
    leg_len = 0.90   # Humanoid: pelvis→foot ≈ 0.4(thigh)+0.4(shin)+0.1(ankle)
    for side, rom, st in (("左", romL, sL_dd), ("右", romR, sR_dd)):
        if not np.isfinite(rom):
            continue
        ceiling = leg_len * np.radians(rom)          # 小角近似：弧长 = R·Δθ
        used = st.mean() if len(st) else float("nan")
        print("  %s rom=%.1f° ⇒ 摆放上限 ≈ %.3f m（腿长按 %.2f m 计）｜实测步幅 %.3f m ⇒ 用掉上限的 %.0f%%" % (
            side, rom, ceiling, leg_len, used, 100.0 * used / ceiling))
    print("  反算：要 0.30 m 步幅需要髋角行程 ≈ %.0f°（=arcsin 意义下的弧长 0.30/%.2f）" % (
        np.degrees(0.30 / leg_len), leg_len))

    print("\n=== H2：大步到底给不给钱（用现奖励的公式直接算）===")
    W, FLOOR, TOP = 35.0, 0.10, 0.32
    for s in (0.046, 0.10, 0.15, 0.20, 0.30):
        clip = max(0.0, min((s - FLOOR) / (TOP - FLOOR), 1.0))
        v_walk = 0.71
        land_per_ep = (n * DT) * (v_walk / max(s, 1e-6))
        print("  步幅 %.2f m ⇒ stride_bonus %.1f 分/落地 × %.0f 次/集 = **%.0f 分/集**（现策略整集才 %.0f 分）" % (
            s, W * clip, land_per_ep, W * clip * land_per_ep, rs))


if __name__ == "__main__":
    main()
