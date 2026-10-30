"""批次④ 离线定价：把"两脚间距"通道的三处钱并轨到 limb-relative 落点之前，先量清代价。

只读：不改训练文件、不动模型、不碰训练进程。口径 = **分/满 1500 步集**，与 §13-§20 一致。

要定价的三个变体（§20.4 修正后的 ④，唯一靶子=间距通道）：
  V1 删 `stride_bonus`（绑在两脚间距上），额度并入 `reach`：swing_drive_weight 2.0→4.0
     ⇒ reach 是"每落地×权重"的线性项，翻倍 = 实收精确 ×2
  V2 取消 `forward_gate`（速度钱不再由 min(左右间距)/0.15 放行）
  V3 `alternate_penalty` 幅度 product → sign(product)×min(|prev|,|cur|)，带饱和 cap（扫格定标）

除了"从当前策略（弹跳者）手里拿走多少"（=新起步价，决定 E 回退档要不要重算），
还算"走路解 vs 弹跳解的方向差"——④ 的正当性完全在这个差上，不在绝对值上。
走路反事实用 §16 的实测几何：L=0.194 m/步、摆动 0.204 s ⇒ duty 0.50 配 v≈0.95。
"""
import gc
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

ROOT = "/home/guo/workspace/humanoid_walk/best_model/"
CANDS = [
    ("448K(当前best)", ROOT + "best_model.zip"),
    ("608K(归档)", ROOT + "tmp_model_data/SAC_18/18_3_min(左右步幅)/best_model.zip"),
]
SEEDS = (123, 2026)
HORIZON = 1500

# 与训练文件里的常数一一对应（改这里要同步核对源码）
FW, TS, SG = 0.4, 0.95, 0.40        # forward_weight / target_speed / sigma
APW = 2.3                            # alternate_penalty_weight
DS = 0.18                            # diff_scale
STEP_S = 0.015                       # Humanoid-v4 控制周期

PER_STEP = ["forward_gauss", "forward_gate", "walk_gate", "flat_gate", "stride_bonus_gated",
            "reach_reward", "alternate_penalty", "speed", "swing_min_reward_gated",
            "position_swap_gated", "height_reward_gated", "survival_bonus_gated",
            "forward_gauss_gated", "negative_stride_penalty", "stride_balance_penalty"]
LAST = ["flight_frac", "single_frac", "double_support_frac", "pitch_angle",
        "ev_reach_L_mean", "ev_reach_R_mean", "ev_rom_L_mean", "ev_rom_R_mean"]


def gauss_raw(v):
    return FW * np.exp(-((v - TS) ** 2) / (2 * SG ** 2))


def rollout(path, seed):
    raw = gym.make("Humanoid-v4", max_episode_steps=HORIZON)
    patch_mirror_symmetry(raw.unwrapped.model)
    env = WalkRewardWrapper(raw, swing_drive_weight=2.0)
    model = SAC.load(path, env=env, device="cpu")

    # 抓住每次落地的 (prev_diff, cur_diff, product)：_check_cycle 是唯一发钱入口
    lands = []
    orig = env._check_cycle

    def spy(current_diff):
        prev = env.prev_diff
        out = orig(current_diff)
        lands.append([len(ps["reward"]), prev, current_diff, out])
        return out

    env._check_cycle = spy
    ps = {"reward": []}
    for k in PER_STEP:
        ps[k] = []
    obs, _ = env.reset(seed=seed)
    for _ in range(HORIZON):
        act, _ = model.predict(obs, deterministic=True)
        obs, rew, term, trunc, info = env.step(act)
        ps["reward"].append(float(rew))
        for k in PER_STEP:
            ps[k].append(float(info.get(k, 0.0)))
        if term or trunc:
            break
    last = {k: float(info.get(k, float("nan"))) for k in LAST}
    last["total_land_count"] = float(env.total_land_count)
    del model, env, raw
    gc.collect()
    return ps, lands, last, HORIZON


def alt_income(lands, wg, cap=None):
    """落地级：一次遍历同时给出「当前 product 形式」与「sign×min 形式（可选饱和）」的每集收支。
    符号沿用 reward 求和里的 `- alternate_penalty * walk_gate`，所以返回值就是它 contributing 的钱。"""
    cur = new = 0.0
    for idx, prev, cur_d, prod in lands:
        if prev is None:
            continue
        g = wg[min(idx, len(wg) - 1)]
        m = np.sign(prod) * min(abs(prev), abs(cur_d))
        if cap is not None:
            m = np.sign(m) * min(abs(m), cap)
        cur += -prod * APW * g
        new += -m * APW * g
    return cur, new


def price(ps, lands, n):
    wg = np.array(ps["walk_gate"])
    cur = sum(ps["reward"])
    sb = sum(ps["stride_bonus_gated"])
    reach = sum(ps["reach_reward"])
    # V1: 删 stride_bonus，reach 翻倍
    v1 = cur - sb + reach
    # V2: forward_gate 归一 ⇒ 补回被 min(间距) 压掉的那部分速度钱
    raw = np.array([gauss_raw(v) for v in ps["speed"]])
    v2 = cur + float(np.sum((raw - np.array(ps["forward_gauss"])) * wg))
    # V3: alternate 幅度 product → sign×min（无饱和）
    alt_cur, alt_min = alt_income(lands, wg)
    v3 = cur - alt_cur + alt_min
    return dict(cur=cur, v1=v1, v2=v2, v3_no_cap=v3, sb=sb, reach=reach,
                alt=alt_cur, n=n)


def main():
    print("批次④ 离线定价（det %d 步 × 种子 %s）\n" % (HORIZON, "/".join(map(str, SEEDS))))
    for name, path in CANDS:
        rows = []
        for seed in SEEDS:
            ps, lands, last, n = rollout(path, seed)
            rows.append((ps, lands, last, n, seed))
        print("=== %s ===" % name)
        for ps, lands, last, n, seed in rows:
            pr = price(ps, lands, n)
            fg_back = float(np.sum(
                (np.array([gauss_raw(v) for v in ps["speed"]]) - np.array(ps["forward_gauss"]))
                * np.array(ps["walk_gate"])))
            print(" seed%-6d 步数%5d 总回报%7.0f | stride_bonus实发%6.0f reach%6.0f "
                  "alternate实收%+7.0f | 速度钱 现%6.0f(fg可回收%+.0f) 落地%4d "
                  "gate%.2f flat%.2f single%.2f 腾空%.2f v%.3f pitch%.3f" % (
                      seed, n, pr["cur"], pr["sb"], pr["reach"], pr["alt"],
                      sum(ps["forward_gauss_gated"]), fg_back,
                      len(lands), float(np.mean(ps["walk_gate"])), float(np.mean(ps["flat_gate"])),
                      last["single_frac"], last["flight_frac"], float(np.mean(ps["speed"])),
                      last["pitch_angle"]))
        # 变体单独 + 合计（两种子均值）
        cur = np.mean([price(r[0], r[1], r[3])["cur"] for r in rows])
        v1 = np.mean([price(r[0], r[1], r[3])["v1"] for r in rows])
        v2 = np.mean([price(r[0], r[1], r[3])["v2"] for r in rows])
        v3 = np.mean([price(r[0], r[1], r[3])["v3_no_cap"] for r in rows])
        # 合计：V1+V2+V3(cap 扫格)
        comb = {}
        for cap in (0.15, 0.30, 0.50, 0.75, 1.0, 1e9):
            tot = []
            for ps, lands, last, n, seed in rows:
                wg = np.array(ps["walk_gate"])
                raw = np.array([gauss_raw(v) for v in ps["speed"]])
                delta = (sum(ps["reward"]) - sum(ps["stride_bonus_gated"]) + sum(ps["reach_reward"])
                         + float(np.sum((raw - np.array(ps["forward_gauss"])) * wg)))
                c_alt, n_alt = alt_income(lands, wg, cap=cap)
                tot.append(delta - c_alt + n_alt)
            comb[cap] = np.mean(tot)
        print("  起步价：当前 %.0f | V1(删stride+reach×2) %+.0f | V2(关forward_gate) %+.0f | "
              "V3(min无饱和) %+.0f" % (cur, v1 - cur, v2 - cur, v3 - cur))
        print("  合计 V1+V2+V3 按 alternate 饱和 cap：%s" % ", ".join(
            "%s:%+.0f" % ("inf" if c > 9 else c, comb[c] - cur) for c in sorted(comb)))
        print("  合计绝对值：%s" % ", ".join("%s:%.0f" % ("inf" if c > 9 else c, comb[c]) for c in sorted(comb)))

        # 落地级分布：定 cap 用
        all_l = [(abs(pv), abs(cd), abs(pv * cd), np.sign(pv * cd))
                 for _, pv, cd, _ in rows[0][1] if pv is not None]
        mn = np.array([min(a, b) for a, b, _, _ in all_l])
        pd_ = np.array([p for _, _, p, _ in all_l])
        sgn = np.array([s for _, _, _, s in all_l])
        q = lambda a: "/".join("%.2f" % x for x in np.percentile(a, [10, 50, 90]))
        print("  落地样本 n=%d：|product| p10/50/90=%s  max=%.2f | min(|p|,|c|) 同=%s" % (
            len(all_l), q(pd_), pd_.max(), q(mn)))
        print("     交替(积<0，发钱)占比 %.0f%%，非交替(积>0，罚)占比 %.0f%%" % (
            100 * (sgn < 0).mean(), 100 * (sgn > 0).mean()))
        print()


if __name__ == "__main__":
    main()
