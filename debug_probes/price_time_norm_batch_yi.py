"""(乙) 的离线定价：**把"按落地发钱"改成"按时间发钱"**到底会改动谁的分数。

为什么是"时间"而不是"里程"：
 · §24.3 实测这条奖励里有 70%（1100/1571 分）是**按落地事件结算**的，
   d(钱)/dv = +854 分/集 per (m/s) —— 落得勤就多赚，这才是弹跳的护城河（不是它单价高）。
 · 若改成"每次落地按该腿两次落地之间的时长折算"（`gap_frames / REF`），
   那么 Σ(单价 × gap) ≈ 单价均值 × 总时长，**总时长是固定的 1500 步** ⇒ 步频补贴被结构性取消，
   而"单价"仍然是间距/交换/落点这些步态质量项 —— 慢而真的迈步不再吃亏。
 · 反过来若按"里程"折算（乘 L_step），会**奖励大步**，弹跳的 L=0.252 m 比走路相的 0.11 m 大得多，
   等于帮倒忙。所以选时长不选里程，这条是设计要点不是口味。

脚本做什么：对若干 checkpoint 跑 det rollout，用与 wrapper 相同的接触判据找出每次落地，
记录该次落地实际发了哪些钱（stride_bonus/alternate/reach/position_swap/swing_min 的结算值），
然后按不同的 REF 重算"时长折算后"的每集收入，并与总分、分/米一起列出。
关键读数：**走路相（16K/32K）相对弹跳（192K）的分数比是否被抬起来**。
"""
import glob
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

HORIZON = 1500
EVENT_KEYS = ["stride_bonus_gated", "alternate_penalty", "reach_reward",
              "swing_min_reward_gated", "position_swap_gated"]
REFS = [1.00, 0.75, 0.60, 0.50]        # 折算因子 = min(gap/REF, 2)；REF=中位步频时长时≈1.0


def contacts(u, gl, gr):
    cl = cr = False
    c = u.data.contact
    for i in range(u.data.ncon):        # 注意：ncon 在 Data 上，contact 结构体本身没有 ncon
        con = c[i]
        g1, g2 = con.geom
        if gl in (g1, g2):
            cl = True
        if gr in (g1, g2):
            cr = True
    return cl, cr


def rollout(path, seed=123):
    raw = gym.make("Humanoid-v4", max_episode_steps=HORIZON)
    patch_mirror_symmetry(raw.unwrapped.model)
    env = WalkRewardWrapper(raw, swing_drive_weight=2.0)
    model = SAC.load(path, env=env, device="cpu")
    obs, _ = env.reset(seed=seed)
    gl = gr = None
    prev_cl = prev_cr = False
    last_land = {"L": 0, "R": 0}
    rows, total, adv, n = [], 0.0, 0.0, 0
    info = {}
    for step in range(HORIZON):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        if gl is None:
            gl, gr = env._foot_geom_one
        cl, cr = contacts(env.unwrapped, gl, gr)
        total += float(r)
        adv += float(info.get("body_advance", 0.0))
        n += 1
        for side, now, was in (("L", cl, prev_cl), ("R", cr, prev_cr)):
            if now and not was:                     # 该腿刚刚落地
                gap = step - last_land[side]
                last_land[side] = step
                rows.append((side, gap, [float(info.get(k, 0.0)) for k in EVENT_KEYS]))
        prev_cl, prev_cr = cl, cr
        if term or trunc:
            break
    del model, env, raw
    return dict(rows=rows, total=total, adv=adv, steps=n,
                single=float(info.get("single_frac", np.nan)),
                flight=float(info.get("flight_frac", np.nan)),
                ds=float(info.get("double_support_frac", np.nan)),
                romL=float(info.get("ev_rom_L_mean", np.nan)),
                gate=float(info.get("walk_gate", np.nan)))


def price(res):
    """返回 (事件类原始合计, 非事件类合计, {REF: 折算后的每集总分})。非事件类按住不动。"""
    base = sum(sum(vs) for _, _, vs in res["rows"])
    other = res["total"] - base
    out = {}
    for ref in REFS:
        out[ref] = other + sum(sum(vs) * min(gap / ref, 2.0) for _, gap, vs in res["rows"])
    return base, other, out


def main():
    pats = sys.argv[1:] or [
        "/home/guo/workspace/humanoid_walk/best_model/best_model_snapshots/best_model_16000.zip",
        "/home/guo/workspace/humanoid_walk/best_model/best_model_snapshots/best_model_32000.zip",
        "/home/guo/workspace/humanoid_walk/best_model/best_model_snapshots/best_model_96000.zip",
        "/home/guo/workspace/humanoid_walk/best_model/best_model_snapshots/best_model_192000.zip",
    ]
    print("折算因子 = min(gap帧数/REF, 2)，1 帧 = 0.015 s。非事件类收入（高度/生存/速度/税）按住不动。\n")
    print("%-10s %5s %6s %8s %9s %8s | %8s |" % (
        "policy", "落地", "步数", "gap中位", "事件类分", "分/米", "原始总分")
        + "".join("  REF=%-5.2f" % r for r in REFS))
    res_all = []
    for p in pats:
        if not glob.glob(p):
            print("%-10s 文件不存在" % p.split("/")[-1])
            continue
        res = rollout(p)
        gaps = np.array([g for _, g, _ in res["rows"]], float)
        base, other, pr = price(res)
        name = p.split("/")[-1].replace("best_model_", "").replace(".zip", "")
        res_all.append((name, res, base, pr))
        print("%-10s %5d %6d %8.1f %9.0f %8.1f | %8.0f |" % (
            name, len(res["rows"]), res["steps"], float(np.median(gaps)), base,
            res["total"] / max(res["adv"], 1e-6), res["total"])
            + "".join(" %10.0f" % pr[r] for r in REFS))
        print("           det 机制：rom_L %.2f° single %.3f flight %.3f ds %.3f walk_gate %.2f" % (
            res["romL"], res["single"], res["flight"], res["ds"], res["gate"]))
    if not res_all:
        return
    ref = next((x for x in res_all if x[0] == "192000"), res_all[-1])
    ref_name, ref_res, _, ref_pr = ref
    print("\n以 %s（弹跳恢复后）为分母的**相对分数**——这才是『有没有动力离开』的量：" % ref_name)
    print("%-10s %10s |" % ("policy", "原始比") + "".join("  REF=%-5.2f" % r for r in REFS))
    for name, res, base, pr in res_all:
        print("%-10s %10.3f |" % (name, res["total"] / ref_res["total"])
              + "".join(" %10.3f" % (pr[r] / ref_pr[r]) for r in REFS))


if __name__ == "__main__":
    main()
