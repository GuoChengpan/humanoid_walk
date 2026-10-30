"""批次⑧ 候选（dense 收入 × ds 硬门控）的**离线定价 + 恶性反例检查**（只读，不改 train_humanoid_sac.py）。

要回答的三件事，按重要性排：
 1) **恶性反例**：把 dense 三项（height/survival/position_swap）乘上 `clip(ds/ds_ref,0,1)` 之后，
    「两脚都撑地、不前进」会不会变成比弹跳更赚的新局部最优？——这是这条改动的**否决项**，
    它不成立才谈得上后面两条。用两条实测 rollout 量：
      · `a=0`（零动作）——真实存活长度/接触结构/dense 收入，不靠估；
      · `a=λ·policy`（λ=0.35/0.6）——「尽量压腿站着但几乎不推进」的近似最优驻留。
 2) **牙有多大**：弹跳的 ds 实测(det 0.000-0.006 / 训练窗 0.0145-0.020)在 ds_ref=0.02/0.05/0.10
    三档下各自保留多少 dense 钱（=起步冲击，单位 分/集 与 Q）。
 3) **方向对不对**：门控后"走路相对弹跳"的差价还剩下多少（走路用 §29 实测的**访问过的**走路相 ds，
    不用理想值——理想走路本来就比弹跳多 77%，排序不是问题）。
"""
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

ROOT = "/home/guo/workspace/humanoid_walk/"
PATH = ROOT + "best_model/best_model_snapshots/best_model_272000.zip"   # 本轮 best（md5 ea87bc95，272K）
HORIZON, SEEDS = 1500, (123, 2026)
DENSE = ("height_reward_gated", "survival_bonus_gated", "position_swap_gated")
DS_REF = (0.02, 0.05, 0.10)


def run(mode="policy", lam=1.0):
    raw = gym.make("Humanoid-v4", max_episode_steps=HORIZON)
    patch_mirror_symmetry(raw.unwrapped.model)
    env = WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=False)
    model = SAC.load(PATH, env=env, device="cpu")
    rows = []
    for seed in SEEDS:
        obs, _ = env.reset(seed=seed)
        n, tot = 0, 0.0
        comp = {k: 0.0 for k in DENSE}
        comp.update({"forward_gauss_raw": 0.0, "stride_bonus": 0.0, "alternate_reward": 0.0,
                     "reach_reward": 0.0, "swing_min_reward": 0.0, "flight_phase_penalty": 0.0})
        prog = 0.0
        x0 = float(raw.unwrapped.data.qpos[0])
        info = {}
        for _ in range(HORIZON):
            if mode == "zero":
                act = np.zeros(env.action_space.shape)
            else:
                a, _ = model.predict(obs, deterministic=True)
                act = lam * np.asarray(a)
            obs, r, term, trunc, info = env.step(act)
            n += 1
            tot += float(r)
            for k in comp:
                comp[k] += float(info.get(k, 0.0))
            if term or trunc:
                break
        x1 = float(raw.unwrapped.data.qpos[0])
        rows.append(dict(n=n, tot=tot, v=(x1 - x0) / (n * 0.02),
                         ds=float(info.get("double_support_frac", np.nan)),
                         single=float(info.get("single_frac", np.nan)),
                         flight=float(info.get("flight_frac", np.nan)),
                         dense=sum(comp[k] for k in DENSE), comp=comp,
                         ev=float(info.get("ev_rom_L_mean", np.nan))))
    del model, env, raw
    return rows


def main():
    print("权重=本轮 best 272K｜det×两种子(123/2026)｜无振荡器｜奖励=现形式(⑤ 后)\n")
    cases = [("策略全量 a", run("policy", 1.0)), ("策略 a×0.6", run("policy", 0.6)),
             ("策略 a×0.35", run("policy", 0.35)), ("零动作 a=0", run("zero"))]
    print("%-14s %7s %8s %7s %8s %8s %8s %8s %9s" % (
        "行为", "ep_len", "总分", "v", "ds", "single", "flight", "dense三项", "ev_romL"))
    for name, rs in cases:
        m = lambda k: float(np.mean([x[k] for x in rs]))
        print("%-14s %7.0f %8.0f %7.2f %8.3f %8.3f %8.3f %9.0f %9.2f   存活=%s" % (
            name, m("n"), m("tot"), m("v"), m("ds"), m("single"), m("flight"), m("dense"),
            m("ev"), [x["n"] for x in rs]))

    hop = dict(ds=float(np.mean([x["ds"] for x in cases[0][1]])),
               tot=float(np.mean([x["tot"] for x in cases[0][1]])),
               dense=float(np.mean([x["dense"] for x in cases[0][1]])),
               v=float(np.mean([x["v"] for x in cases[0][1]])))
    print("\n=== ① 恶性反例检查（否决项）===")
    print("弹跳基线：ds=%.4f、dense=%.0f 分/集、v=%.2f、总分=%.0f" % (hop["ds"], hop["dense"], hop["v"], hop["tot"]))
    for name, rs in cases[1:]:
        d = float(np.mean([x["dense"] for x in rs]))
        ds = float(np.mean([x["ds"] for x in rs]))
        tot = float(np.mean([x["tot"] for x in rs]))
        for ref in DS_REF:
            g_h = np.clip(hop["ds"] / ref, 0, 1)
            g_x = np.clip(ds / ref, 0, 1)
            after_h = hop["dense"] * g_h
            after_x = d * g_x
            print("  %-12s ds=%.3f → 门控(%.2f)后 dense %.0f→%.0f 分；弹跳 dense %.0f→%.0f 分；%s" % (
                name, ds, ref, d, after_x, hop["dense"], after_h,
                ("**该行为在 dense 项上已超过弹跳**" if after_x > after_h else "仍低于弹跳")))
    print("  （注：总分还要加/减前进类通道；a=0 不前进 ⇒ forward_gauss≈0、stride/alternate≈0，"
          "所以真正的比较要把下表的事件类损失一并看）")

    print("\n=== ② 牙有多大（对弹跳的起步冲击）===")
    print("  弹跳实测 ds = %.4f（det；训练窗 0.0145-0.020）" % hop["ds"])
    for ref in DS_REF:
        g_det = np.clip(hop["ds"] / ref, 0, 1)
        g_tr = np.clip(0.017 / ref, 0, 1)
        print("  ds_ref=%.2f：det 保留 %.2f → 冲击 %+.0f 分/集；训练窗保留 %.2f → 冲击 %+.0f 分/集"
              "（≈ %.1f Q 单位，1Q≈15 分/集）" % (
                  ref, g_det, hop["dense"] * (g_det - 1), g_tr, hop["dense"] * (g_tr - 1),
                  hop["dense"] * (g_det - 1) / 15.0))

    print("\n=== ③ 门控后走路相对弹跳还剩多少差价（用**访问过的**走路相，不用理想值）===")
    print("  §29 实测：走路相每步收入相对弹跳 = ≈0/−14%（理想走路是 +77%，但策略看不见理想值）")
    print("  §25 实测：能造出 ds 0.12-0.14 的非对称初态只能活 70-140 步 ⇒ 它的 dense 收入按存活长度折算")
    for ref in DS_REF:
        for ds_w, life in ((0.05, 1500), (0.12, 105)):
            g_w = np.clip(ds_w / ref, 0, 1)
            g_h = np.clip(hop["ds"] / ref, 0, 1)
            dense_w = hop["dense"] * g_w * (life / HORIZON)
            print("  ds_ref=%.2f｜走路相 ds=%.2f 存活 %d 步：dense %.0f vs 弹跳 %.0f → %s" % (
                ref, ds_w, life, dense_w, hop["dense"] * g_h,
                "走路占优" if dense_w > hop["dense"] * g_h else "弹跳仍占优"))

    print("\n=== 判读（预先声明，不看完再改）===")
    print("  否决：只要 a=0 或 a×λ 的门控后 dense **超过**弹跳，且它不需要前进 ⇒ 这条改动造出新的更坏局部最优，")
    print("        必须先把 dense 里的 survival/height 与「前进」绑定，那就不是「只加一个门控」了。")
    print("  放行：所有驻留行为在全部三档下都**不**超过弹跳，且 ds_ref=0.02/0.05 中至少一档让")
    print("        「访问过的走路相」的 dense 严格高于弹跳 ⇒ 才值得写代码 + 钉判据。")


if __name__ == "__main__":
    main()
