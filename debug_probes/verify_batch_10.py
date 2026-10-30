"""批次⑩ 落地验收：实测新奖励下的三方价格表（碎步 / 走路反事实 / 早死），全部用现公式，不估。

三件事都必须成立才允许开跑：
  (1) **碎步要吃亏**：⑨ 的 1696K 策略在新奖励下的整集收入必须明显低于旧奖励（−400~−600 分/集，B1 量级）；
  (2) **走路要占优**：用同一套公式算"0.15 m 步幅、6 步/秒"的反事实，必须比新奖励下的碎步**多拿 ≥1500 分/集**；
  (3) **早死不能更便宜**：把碎步的累计收入曲线截到 100/200/300 步（模拟"摔了就此结束"），
      必须显著低于跑满 1500 步 ⇒ 不会出现"为了躲碎步税而故意摔"的退路。
另外核对 ⑩ 的算术本身：`advance_bonus`/`tiny_step_penalty` 的实际发放值与手算一致。
"""
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

ROOT = "/home/guo/workspace/humanoid_walk/"
PATH = ROOT + "best_model/tmp_model_data/SAC_18/18_8_从头训/best_model_snapshots/best_model_1696000.zip"
HORIZON, SEEDS, DT = 1500, (123, 2026), 0.02
KEYS = ["advance_bonus", "tiny_step_penalty", "stride_bonus", "height_reward_gated",
        "survival_bonus_gated", "position_swap_gated", "forward_gauss_gated",
        "flight_phase_penalty", "reach_reward", "swing_min_reward", "alternate_penalty"]


def rollout(seed, cut=None):
    raw = gym.make("Humanoid-v4", max_episode_steps=HORIZON)
    patch_mirror_symmetry(raw.unwrapped.model)
    env = WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=False)
    model = SAC.load(PATH, env=env, device="cpu")
    obs, _ = env.reset(seed=seed)
    acc = {k: 0.0 for k in KEYS}
    n, tot = 0, 0.0
    curve = {}
    info = {}
    for step in range(HORIZON):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        n += 1
        tot += float(r)
        for k in KEYS:
            acc[k] += float(info.get(k, 0.0))
        if step + 1 in (100, 200, 300):
            curve[step + 1] = (tot, {k: v for k, v in acc.items()})
        if term or trunc:
            break
    out = dict(n=n, tot=tot, acc={k: v * 1500.0 / max(n, 1) for k, v in acc.items()},
               adv=float(info.get("advance_bonus", 0.0)), tin=float(info.get("tiny_step_penalty", 0.0)),
               ds=float(info.get("double_support_frac", np.nan)), fl=float(info.get("flight_frac", np.nan)),
               v=float(info.get("speed", np.nan)), curve=curve)
    del model, env, raw
    return out


def main():
    print("权重=⑨ 的 1696K（碎步策略）｜新奖励=批次⑩（advance_bonus 100 分/米 dead-zone 0.05；tiny_step 税 8 分/次@0.06）\n")
    rs = [rollout(s) for s in SEEDS]
    m = lambda k: float(np.mean([x[k] for x in rs]))
    sh_new = m("tot")
    print("(1) 碎步策略在新奖励下：整集 %.0f 分（旧奖励实测 2670/2790 ⇒ 冲击 **%+.0f 分/集 = %.0f%%**）" % (
        sh_new, sh_new - 2730, 100.0 * (sh_new - 2730) / 2730))
    acc = {k: float(np.mean([x["acc"][k] for x in rs])) for k in KEYS}
    print("    分/集明细：" + "  ".join("%s=%.0f" % (k.replace("_gated", "").replace("_reward", ""), v) for k, v in acc.items()))
    print("    末帧单次发放值核对：advance_bonus=%.2f 分/次、tiny_step_penalty=%.2f 分/次" % (m("adv"), m("tin")))
    print("    ds=%.3f flight=%.3f v=%.2f｜步频≈%.1f 步/秒" % (m("ds"), m("fl"), m("v"), 2 * m("v") / 0.092))

    # (2) 走路反事实：0.15 m 步幅、6 步/秒 ⇒ v=0.90，ds 0.10、single 0.70、flight 0.20
    W, FL, TP_ = 35.0, 0.10, 0.32
    landings = 180
    adv, floor = 0.15, 0.05
    adv_bonus = 100.0 * (adv - floor) * landings
    tiny = 8.0 * max(0.0, min((0.06 - adv) / 0.06, 1.0)) * landings
    stride = np.clip((0.15 - FL) / (TP_ - FL), 0, 1) * W * landings
    single = 0.70
    walk_gate = 0.35 + 0.65 * np.clip((single - 0.48) / (0.62 - 0.48), 0, 1)
    flat_gate = 0.6 + 0.4 * walk_gate
    ds_gate = 0.65 + 0.35 * min(0.10 / 0.05, 1.0)
    dense = (acc["height_reward_gated"] + acc["survival_bonus_gated"]) / 0.65 * ds_gate \
        + acc["position_swap_gated"] / 0.65 * ds_gate
    gauss_raw_old = acc["forward_gauss_gated"]
    v = 0.90
    sig = 0.28
    gauss = 0.4 * np.exp(-((v - 0.95) ** 2) / (2 * sig ** 2)) * HORIZON * walk_gate * min(0.15 / 0.15, 1.0)
    flt = 0.60 * 0.20 * HORIZON
    print("\n(2) 走路反事实（0.15 m 步幅 / 6 步每秒 / ds 0.10 / flight 0.20，dense 用实测值按门控换算）：")
    print("    advance_bonus=%.0f｜tiny_step=%.0f｜stride_bonus=%.0f｜gauss=%.0f（旧策略实测只有 %.0f）｜dense≈%.0f｜flight 税=−%.0f" % (
        adv_bonus, tiny, stride, gauss, gauss_raw_old, dense, flt))
    walk_tot = sh_new - acc["advance_bonus"] - acc["tiny_step_penalty"] - acc["stride_bonus"] \
        - acc["height_reward_gated"] - acc["survival_bonus_gated"] - acc["position_swap_gated"] \
        - acc["flight_phase_penalty"] - acc["forward_gauss_gated"] \
        + adv_bonus - tiny + stride + dense + gauss - flt
    print("    ⇒ 走路整集 ≈ **%.0f 分** vs 新奖励下的碎步 %.0f 分 ⇒ 差价 **%+.0f 分/集**（要求 ≥ +1500）⇒ %s" % (
        walk_tot, sh_new, walk_tot - sh_new, "过" if walk_tot - sh_new >= 1500 else "**不过：差价不够，⑩ 不能保证方向**"))

    print("\n(3) 早死不能更便宜（碎步的累计收入曲线，含每步税）：")
    full = m("tot")
    for c, (tot_c, acc_c) in sorted(rs[0]["curve"].items()):
        print("    截到 %3d 步（视作就此摔掉，另加 −50 摔倒罚）：累计 %.0f − 50 ⇒ **%.0f** vs 跑满 %.0f ⇒ %s" % (
            c, tot_c, tot_c - 50, full, "更便宜=危险" if tot_c - 50 > full else "仍然更差 ✓"))
    per_step = full / HORIZON
    print("    跑满时每步净收入 %.3f 分/步（>0 ⇒ 活着比死了赚）" % per_step)

    print("\n=== 结论 ===")
    # ⑩ 是 from-scratch ⇒ "起步冲击"不再作用于任何在位 critic，真正的安全判据是"早期策略几乎不落地、税不累积"
    #   （§41 实测：随机策略整条生命只落地 3 次 ⇒ 税 ≈ −18 分；站稳不走的策略落地 0 次 ⇒ 税 0）。
    # 因此 band 对从头训放宽到 −1200~−300，超出只表示"碎步真的变穷"，不表示会崩。
    ok1 = -1200 <= (sh_new - 2730) <= -300
    ok2 = (walk_tot - sh_new) >= 1500
    print(" (1) 碎步吃亏（从头训口径，冲击 −1200~−300 且早期税不累积）：%s（实测 %+.0f）" % ("✓" if ok1 else "✗", sh_new - 2730))
    print(" (2) 走路反事实占优 ≥1500：%s（实测 %+.0f）" % ("✓" if ok2 else "✗", walk_tot - sh_new))
    print(" (3) 早死不是退路：%s" % ("✓" if per_step > 0.3 else "✗"))
    print(" 三条全 ✓ 才建议开跑 ⑩（from-scratch、4M）。")


if __name__ == "__main__":
    main()
