"""批次①+② 的验收探针（只读，不写模型/日志）。

要证的四件事：
 1) **恒等式**：single_frac = 1 − flight − ds、duty_sum = 1 − flight + ds（逐帧核对，防止
    我把三个占比算错——它们全来自同一个 200 步滚动窗）
 2) **门控公式**：逐帧核对 walk_gate = floor + (1−floor)·clip((single−lo)/(ref−lo))，并把
    斜坡在若干工作点上列出来（确认当前工作点的**局部斜率非零**，这是批次②的全部动机）
 3) **新旧门控的即时代价**：同一策略、同一动作序列下，旧 ds 门控 vs 新 single 门控各自
    发放多少 → 重启当天会掉多少钱（预期 ~46 分/集），以及"走路能多拿多少"
 4) **批次① 的记账**：新键是否真的有值、三个 *_rate 是否回到 [0,1]、spacing 三兄弟的
    分布与负值占比（这才是"负步幅罚改任意落地会误伤多少"的实测答案）
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

MODEL = os.environ.get("PROBE_MODEL", "/home/guo/workspace/humanoid_walk/best_model/best_model.zip")
N_STEP = 2000
SEEDS = (123, 2026)


def build(seed):
    def _f():
        raw = gym.make("Humanoid-v4", max_episode_steps=1500)
        patch_mirror_symmetry(raw.unwrapped.model)
        return WalkRewardWrapper(raw, swing_drive_weight=2.0)
    v = DummyVecEnv([_f])
    v.seed(seed)
    return v


def main():
    vec = build(SEEDS[0])
    model = SAC.load(MODEL, env=vec, device="cpu")
    w = vec.envs[0]          # WalkRewardWrapper 本体（.unwrapped 会剥到 HumanoidEnv）

    rows = []
    obs = vec.reset()
    for t in range(N_STEP):
        act, _ = model.predict(obs, deterministic=True)
        obs, rew, done, info = vec.step(act)
        rows.append(info[0])
        if done[0]:
            obs = vec.reset()                  # 与训练一致：撞 horizon 就重开，保持事件窗连续
    vec.close()

    g = lambda k: np.array([float(r.get(k, np.nan)) for r in rows])
    single, flight, ds = g("single_frac"), g("flight_frac"), g("double_support_frac")
    duty, gate = g("duty_sum"), g("walk_gate")

    print(f"模型={MODEL}\n采样={len(rows)} 步 det（事件窗取收敛后的后 70%）")
    m = int(len(rows) * 0.7)

    # --- 1 恒等式 ---
    e1 = np.abs(single - (1.0 - flight - ds))[m:].max()
    e2 = np.abs(duty - (1.0 - flight + ds))[m:].max()
    print(f"\n[1] 恒等式 |single−(1−flight−ds)|max={e1:.2e}  |duty_sum−(1−flight+ds)|max={e2:.2e}"
          f"  → {'✓' if max(e1, e2) < 1e-9 else '✗ 占比算错'}")

    # --- 2 门控公式 ---
    lo, ref, fl = w.walk_gate_lo, w.walk_gate_ref, w.walk_gate_floor
    pred = fl + (1.0 - fl) * np.clip((single - lo) / (ref - lo), 0.0, 1.0)
    e3 = np.abs(gate - pred)[m:].max()
    print(f"[2] walk_gate 与公式一致: 最大偏差={e3:.2e} → {'✓' if e3 < 1e-9 else '✗'}")
    print(f"    参数 floor={fl} lo={lo} ref={ref}")
    print("    single → gate : " + "  ".join(
        f"{s:.2f}→{fl + (1-fl)*min(1, max(0,(s-lo)/(ref-lo))):.3f}"
        for s in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)))
    slope = (1 - fl) / (ref - lo)
    print(f"    工作点局部斜率 d(gate)/d(single) = {slope:.2f}（旧 ds 门控在 ds≤0.02 段的斜率"
          f"同样存在，但 single 从 0.48 起步就有分，旧门控要 ds>0 才开始）")

    # --- 3 新旧门控的即时代价 ---
    old_gate = fl + (1.0 - fl) * np.minimum(1.0, ds / 0.15)     # R''#1 的 ds 式（上一轮口径）
    gauss, strat, swap = g("forward_gauss"), g("stride_bonus"), g("position_swap_reward")
    gated = gauss + strat + swap
    flat_pool = g("height_reward") + g("survival_bonus")
    ff = w.flat_gate_floor
    for name, gg in ("旧 ds 门控", old_gate), ("新 single 门控", gate):
        paid = float(np.mean(gated[m:] * gg[m:])) * 1500
        lost = float(np.mean(gated[m:] * (1.0 - gg[m:]))) * 1500
        print(f"[3] {name:14s} gate均值={np.mean(gg[m:]):.3f} → 发放 {paid:+7.0f}/集  被门掉 {-lost:+7.0f}/集"
              f"   (未门控毛收入 {float(np.mean(gated[m:]))*1500:+.0f}/集)")
    # 上一轮口径 = 实际回报 + (旧ds门−新single门)×毛池 + (1−flat_gate)×平地池（换算到 1500 步满集）
    prev_rate = float(np.mean((gated * old_gate + flat_pool)[m:])) * 1500
    now_rate = float(np.mean((gated * gate + flat_pool * g("flat_gate"))[m:])) * 1500
    print(f"    平地池(height+survival) = {float(np.mean(flat_pool[m:]))*1500:+.0f} 分/集"
          f"  flat_gate 均值={float(np.mean(g('flat_gate')[m:])):.3f} (floor {ff})")
    print(f"    **即时冲击（上一轮口径 {prev_rate:+.0f} → 本轮口径 {now_rate:+.0f}）= "
          f"{now_rate-prev_rate:+.0f} 分/满集**")
    slope_pool = (1.0 - fl) / (ref - lo)
    slope_prev_round = 2.6          # 上一轮(批次②)的 (1−0.35)/(0.75−0.50)
    pool1500 = float(np.mean(gated[m:])) * 1500
    flat1500 = float(np.mean(flat_pool[m:])) * 1500
    print(f"    反飞行斜率 d(分/集)/d(1pp single)：本轮 = ({pool1500:.0f} + {flat1500:.0f}×{1-ff:.1f})"
          f" × {slope_pool:.2f} = **{(pool1500 + flat1500*(1-ff))*slope_pool/100:.1f}**；"
          f"上一轮 = {pool1500:.0f} × {slope_prev_round:.2f} = {pool1500*slope_prev_round/100:.1f}"
          f"（无平地门控）→ **×{((pool1500+flat1500*(1-ff))*slope_pool)/(pool1500*slope_prev_round):.1f}**")
    print(f"    → 重启当天相对上一轮的即时冲击 = {float(np.mean((gate[m:]-old_gate[m:])*gated[m:]))*1500:+.0f} 分/集")

    # --- 4 记账与分布 ---
    print("\n[4] 批次① 新键实测（后 70% 均值 ×1500 = 每满集的钱）")
    for k in ("survival_bonus", "height_reward", "reach_reward", "forward_gauss", "stride_bonus",
              "position_swap", "stride_bonus_gated", "position_swap_gated", "forward_gauss_gated",
              "hip_xz_angle_penalty", "support_sym_penalty", "slip_penalty", "flight_phase_penalty",
              "single_frac", "duty_sum", "walk_gate"):
        v = g(k)[m:]
        if np.isnan(v).all():
            print(f"    {k:24s} 缺键 ✗")
            continue
        unit = "" if k.endswith("frac") or k.endswith("gate") or k.endswith("sum") else "×1500"
        val = v.mean() * (1500 if unit == "×1500" else 1)
        print(f"    {k:24s} {val:9.3f} {'分/集' if unit=='×1500' else '占比'}")
    print("\n    三个 *_rate 是否回到 [0,1]：")
    for k in ("stride_bonus_rate", "short_stride_event_rate", "negative_stride_rate", "ds_landing_rate"):
        v = g(k)
        print(f"      {k:26s} 末值={v[-1]:.3f} 范围=[{np.nanmin(v):.3f},{np.nanmax(v):.3f}]"
              f" {'✓' if np.nanmax(v) <= 1.0001 else '✗ 仍>1'}")
    print("\n    spacing 三兄弟（落地帧两脚前后间距；这就是负步幅罚的观测量）：")
    print("      %-12s %8s %8s %8s %8s %8s %8s %8s" % ("leg", "n_ds", "n_air", "mean", "p10", "p50", "p90", "negfrac"))
    for s in "LR":
        for st in ("spacing", "spacing_ds", "spacing_air"):
            key = f"ev_{st}_{s}"
            n = g(f"{key}_n")[-1]
            mean = g(f"{key}_mean")[-1]
            p10, p50, p90 = (g(f"{key}_p10")[-1], g(f"{key}_p50")[-1], g(f"{key}_p90")[-1])
            nf = g(f"{key}_negfrac")[-1]
            if st == "spacing":
                nds, nair = g(f"ev_spacing_ds_{s}_n")[-1], g(f"ev_spacing_air_{s}_n")[-1]
                print("      %-12s %8.0f %8.0f %8.3f %8.3f %8.3f %8.3f %8.2f" %
                      (f"{s} 全部", nds, nair, mean, p10, p50, p90, nf))
            else:
                print("      %-12s %8s %8s %8.3f %8.3f %8.3f %8.3f %8.2f" %
                      (f"{s} {'双撑' if st.endswith('ds') else '腾空'}", "", "", mean, p10, p50, p90, nf))
    # 若把负步幅罚改成"任意落地"，按现有 NEGATIVE 参数会扣多少
    wgt, scale = 10.0, 0.08
    for s in "LR":
        air = g(f"ev_spacing_air_{s}_mean")
        nf = g(f"ev_spacing_air_{s}_negfrac")[-1]
        print(f"      若『任意落地』启用: {s} 腾空期落地占 {100*nf:.0f}% 为负 → "
              f"粗算罚 = {float(nf)*wgt*0.5:.1f} 分/次落地")
    print(f"\n[5] reach 对照：ev_reach_L={g('ev_reach_L_mean')[-1]:+.3f} R={g('ev_reach_R_mean')[-1]:+.3f}"
          f"  pitch={g('pitch_angle')[-1]:+.4f}（③ 要治的腰前倾套利）")


if __name__ == "__main__":
    main()
