"""批次⑩.1-fix 的开跑前验收：**四种退化行为在同一把尺下重新定价**（只读，不改训练）。

⑩ 原版把 `adv` 定义成**躯干**位移 ⇒ 跌倒就是最快的躯干位移 ⇒ 策略学会"往前扑"（实测 620 集 0 集跑满）。
fix 之后：`adv = min(左,右) 两脚落点间距`（纯脚量）× `upright = clip((z−1.15)/0.15)`（必须站稳）。
这里在**新奖励**下重测四条最省事的赚钱路径，看还有没有哪条比"真走路"便宜：
  1) **碎步**：⑨ 的成熟碎步策略（1696K，8.8 步/秒、两脚几乎重叠）
  2) **扑倒**：⑩ 的跌倒策略（1920K，每次落地两脚张开但躯干已塌）
  3) **站立不动**：把动作置零（前进门会掐掉 dense，验证它是不是新避风港）
  4) **早死**：把 1)/2) 的累计收入截到 100/200/300 步 + 摔罚 −50，与跑满比较
另给 **走路反事实**（lo=0.15、z=1.35 ⇒ upright=1、6 步/秒）作为应当最高的那一档。
"""
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

ROOT = "/home/guo/workspace/humanoid_walk/"
P9 = ROOT + "best_model/tmp_model_data/SAC_18/18_8_从头训/best_model_snapshots/best_model_1696000.zip"
P10 = ROOT + "best_model/best_model_snapshots/best_model_1920000.zip"
HORIZON, SEEDS = 1500, (123, 2026)
KEYS = ["stride_upright", "stride_step_m", "tiny_step_penalty", "stride_bonus", "height_reward_gated",
        "survival_bonus_gated", "position_swap_gated", "forward_gauss_gated",
        "flight_phase_penalty", "reach_reward", "swing_min_reward", "slip_penalty",
        "negative_stride_penalty", "stride_balance_penalty", "lateral_penalty",
        "alternate_penalty", "ctrl_cost", "dense_ds_factor", "advance_step_m", "advance_upright"]


def run(path, mode="policy", lam=1.0, cut_at=None):
    raw = gym.make("Humanoid-v4", max_episode_steps=HORIZON)
    patch_mirror_symmetry(raw.unwrapped.model)
    env = WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=False)
    model = None if path is None else SAC.load(path, env=env, device="cpu")
    obs, _ = env.reset(seed=123 if model is None else 123)
    acc = {k: 0.0 for k in KEYS}
    n, tot = 0, 0.0
    info = {}
    for _ in range(HORIZON):
        if model is None:
            a = np.zeros(env.action_space.shape)
        else:
            a, _ = model.predict(obs, deterministic=True)
            a = lam * np.asarray(a, dtype=np.float64)
        obs, r, term, trunc, info = env.step(a)
        n += 1
        tot += float(r)
        for k in KEYS:
            v = info.get(k)
            if v is not None:
                acc[k] += float(v)
        if cut_at and n >= cut_at:
            break
        if term or trunc:
            break
    scale = HORIZON / max(n, 1)
    out = dict(n=n, tot=tot, adv=acc["stride_bonus"] * scale, tax=acc["tiny_step_penalty"] * scale,
               dense=(acc["height_reward_gated"] + acc["survival_bonus_gated"] + acc["position_swap_gated"]) * scale,
               stride_other=0.0, reach=acc["reach_reward"] * scale,
               ftax=acc["flight_phase_penalty"] * scale, slip=acc["slip_penalty"] * scale,
               up=(acc["stride_upright"] / max(n, 1)),
               per_step=tot / max(n, 1),
               step_m=(acc["stride_step_m"] / max(n, 1)))
    del model, env, raw
    return out


def counterfactual(lo, z, steps_per_sec):
    """按现在的公式手算一个反事实：两脚间距 lo、躯干高 z、步频 steps_per_sec。"""
    upright = float(np.clip((z - 1.15) / 0.15, 0.0, 1.0))
    landings = steps_per_sec * HORIZON * 0.02
    adv = np.clip((lo - 0.10) / (0.22 - 0.10), 0, 1) * 45.0 * upright * landings   # 合并后的 stride_bonus
    tax = 6.0 * float(np.clip((0.10 - lo) / 0.10, 0.0, 1.0)) * landings
    # single = 1 − flight − ds（定义式，不是不等式）。这里取反事实的 flight 0.30、ds 0.08
    single = 1.0 - 0.30 - 0.08
    wg = 0.35 + 0.65 * np.clip((single - 0.48) / 0.14, 0, 1)
    fg = 0.6 + 0.4 * wg
    dsf = 0.65 + 0.35 * min(0.08 / 0.05, 1.0)
    dense = (1339 + 690 + 183) / 0.65 * dsf * fg if lo > 0 else 0.0
    stride = 0.0
    ftax = 0.60 * 0.30 * HORIZON
    return dict(adv=adv, tax=tax, dense=dense, stride=stride, ftax=ftax,
                 tot=dense + adv - tax + stride - ftax, landings=landings)


def main():
    print("新奖励 = ⑩.1-fix（adv 用两脚落点间距 × upright 门）\n")
    rows = [("⑨ 碎步 1696K", run(P9)), ("⑩ 扑倒 1920K", run(P10)), ("零动作(站立/瘫)", run(None))]
    print("%-16s %6s %8s %9s %9s %8s %8s %8s %9s" % (
        "行为", "存活", "整集分", "分/步", "stride钱", "碎步税", "dense", "落点间距m", "upright均值"))
    for name, r in rows:
        print("%-16s %6.0f %8.0f %9.3f %8.0f %8.0f %8.0f %8.0f %9.3f" % (
            name, r["n"], r["tot"], r["per_step"], r["adv"], -r["tax"], r["dense"], r["step_m"],
            r["up"] / max(r["n"], 1)))
    print("\n走路反事实（应当是最高的那一档）：")
    for lo, z, cps, tag in ((0.119, 1.35, 3.83, "⑧ 弹跳 lo=0.119/115次"), (0.12, 1.35, 5.0, "保守 0.12m/5步秒"), (0.15, 1.35, 6.0, "目标 0.15m/6步秒"),
                           (0.20, 1.35, 5.0, "理想 0.20m/5步秒"), (0.05, 1.20, 8.0, "半碎步+z塌")):
        c = counterfactual(lo, z, cps)
        print("  %-18s advance=%+6.0f 税=%6.0f dense=%6.0f stride=%6.0f flight税=%5.0f ⇒ 合计 %6.0f 分/集" % (
            tag, c["adv"], -c["tax"], c["dense"], c["stride"], -c["ftax"], c["tot"]))

    print("\n早死是否更便宜（把 ⑨/⑩ 的整集分截到 100/200/300 步 + 摔罚 −50）：")
    for name, p in (("⑨", P9), ("⑩", P10)):
        full = run(p)["tot"]
        for cut in (100, 200, 300):
            r = run(p, cut_at=cut)
            v = r["tot"] - 50 if r["n"] < 1499 else r["tot"]
            print("  %s 截 %3d 步：实际累计 %.0f（+摔罚 −50 ⇒ %.0f）vs 跑满 %.0f ⇒ %s" % (
                name, cut, r["tot"], v, full, "仍更差 ✓" if v < full else "**早死更便宜 ✗**"))
    print("\n落地时 stride_upright（z 门）实测：",
          "  ".join("%s=%.2f" % (nm, r["up"]) for nm, r in rows))
    print("判读（预先声明，看完不改）：")
    print("  放行条件：① ⑨ 碎步的整集分必须**低于** ⑩ 扑倒与走路反事实中最好的一档，且碎步不再靠躯干位移拿钱；")
    print("            ② 任何退化档（站立/扑倒/早死）的合计都**不得高于**走路反事实；")
    print("            ③ ⑨ 碎步每步净收入必须 > 0（否则早期会学会故意摔）。")


if __name__ == "__main__":
    main()
