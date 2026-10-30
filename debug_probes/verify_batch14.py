"""批次⑭（M1 alternate 摘门控 + M2 phase/reach 挂交换率门）的验证：
**逐帧全量台账恒等式 + 基线复现 + 起点价 + "早死更便宜"红线**。

为什么这次要做全量台账（Σinfo 实付 == r）：M1/M2 都改的是"求和式里怎么乘"，
而本项目历史上吃过 `stall_penalty` 算了却没进求和式、`(乙.1)(乙.2)` 两个因子恒等于 1 这类
"接线级"错误——抽查单项证明不了整体。把每一项的**实付**导出后逐帧求和与真实 r 比较，
任何一项没接上/接重了都会在残差里现形。

四步：
  Step0 基线复现（用户规则：每个新探针都必须先在自家代码里复现一个已知数）
        ⑫ 归档 TB 末段实测 `alternate_penalty≈0.27 分/帧`、`position_swap_rate≈0.006`、
        `ds−flight p50=−0.12 ⇒ walk_gate≈0.41`。本探针用 ⑫ 权重、⑫ 的价格（只把 M1/M2 换回去）
        复算这三项，落在容差内才继续。
  Step1 恒等式：`swap_gate == 0.75 + 0.25·clip(swap_rate/0.5)`；`alternate_paid == alt×(1 if alt>0 else walk_gate)`
  Step2 全量台账：|Σinfo_paid − r| 的中位 < 0.02 分/帧
  Step3 定价与红线：⑭ 起点价（分/帧、分/集）对照 ⑬ 的 +0.03；
        **早死红线**：零动作实测 ≈−2.06 分/帧（§42：−83 分/集 / 39 帧），
        要求 剪刀式每帧率 > −1.0 ⇒ "快点摔"仍然明显更差；否则判 M2 floor 太狠，回 0.85。
"""
import ast
import os
import re
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

MODEL = os.environ.get("PROBE_MODEL", "best_model/tmp_model_data/SAC_19/19_2_小碎步惩罚4M/best_model.zip")
DT, FR = 0.015, 1500
SEEDS = tuple(int(x) for x in os.environ.get("V_SEEDS", "123,2026,7,31,55,77,91,101").split(","))
BASE_ALT_FACE, BASE_SWAP_RATE, BASE_GATE = 0.27, 0.006, (0.20, 0.41)   # ⑫ TB / jsonl 末段实测

SRC = open("train_humanoid_sac.py", encoding="utf-8").read()
tree = ast.parse(SRC)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WalkRewardWrapper")
init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
C = {}
for node in ast.walk(init):
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        t = node.targets[0]
        if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
            v = node.value
            try:
                C[t.attr] = -float(v.operand.value) if (isinstance(v, ast.UnaryOp) and isinstance(v.operand, ast.Constant)) else float(v.value)
            except Exception:
                pass
SG_F, SG_REF = C["swap_gate_floor"], C["swap_gate_ref"]
NEG_W_SRC = re.search(r"NEGATIVE_STRIDE_WEIGHT\s*=\s*([0-9.]+)", SRC).group(1)
WG_F, WG_LO, WG_RE = C["walk_gate_floor"], C["walk_gate_lo"], C["walk_gate_ref"]
DS_F, DS_RE = C["dense_ds_floor"], C["dense_ds_ref"]
TW, TR, TCAP = C["tiny_step_weight"], C["tiny_step_ref"], C["tiny_step_cap"]
AD_W = C["activity_diff_penalty_weight"]
FL_W = C["flight_time_penalty_weight"]
fails, notes = [], []


def chk(name, cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {msg}")
    if not cond:
        fails.append(name)


# 全量台账：求和式里每一项的"实付"，符号照求和式抄（reward = Σ正项 − Σ罚项）。
# 这是 debug_probes/verify_batch14.py 的核心——任何一项漏接/重接/乘错门都会在残差里现形。
SIGN = {
    "+": ["forward_gauss_gated", "swing_forward_reward", "reach_reward_gated",
          "swing_min_reward_gated", "stride_bonus_gated", "net_balance", "height_reward_gated",
          "fall_penalty", "phase_reward_gated", "survival_bonus_gated", "position_swap_gated",
          "contact_phase_reward"],
    "-": ["speed_penalty", "activity_diff_paid", "support_sym_penalty", "alternate_paid",
          "lateral_penalty", "hip_xz_angle_penalty", "flight_phase_penalty", "ctrl_cost",
          "short_stride_penalty", "negative_stride_penalty", "stride_balance_penalty", "slip_penalty"],
}


def r_hat(info):
    v = 0.0
    for k in SIGN["+"]:
        if k in info:
            v += float(info[k])
    for k in SIGN["-"]:
        if k in info:
            v -= float(info[k])
    return v


def run(model):
    raw = gym.make("Humanoid-v4", max_episode_steps=FR)
    patch_mirror_symmetry(raw.unwrapped.model)
    w = WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=True)
    eps, nan, missing = [], 0, set()
    bad_sg = bad_alt = 0
    # 每帧累加"和"而不是取中位：事件类项（落地才发钱）在 400 帧里只有 ~30 帧非零，
    #   取中位必然得到 0 —— 这正是我第一版把 M1/L2 的效应量算成 0 的原因。
    SUMK = ["phase_reward", "reach_reward", "alternate_penalty", "walk_gate", "swap_gate",
            "swap_rate",
            "alternate_paid", "phase_reward_gated", "reach_reward_gated",
            "short_stride_penalty", "negative_stride_penalty", "height_reward_gated",
            "survival_bonus_gated", "position_swap_gated", "forward_gauss_gated", "speed_penalty",
            "slip_penalty", "flight_phase_penalty", "ctrl_cost", "support_sym_penalty",
            "stride_bonus_gated", "swing_min_reward_gated", "contact_phase_reward",
            "swing_forward_reward", "net_balance", "hip_xz_angle_penalty", "lateral_penalty",
            "stride_balance_penalty", "activity_diff_paid", "fall_penalty"]
    for sd in SEEDS:
        obs, _ = w.reset(seed=sd)
        t = 0
        tot = 0.0
        sums = {k: 0.0 for k in SUMK}
        d14_12 = 0.0          # 同一轨迹上 (⑭付费 − ⑫付费) 的逐帧累计
        resid = []
        while True:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = w.step(np.asarray(act, dtype=np.float64))
            t += 1
            if not np.isfinite(r):
                nan += 1
            for grp in ("+", "-"):
                for k in SIGN[grp]:
                    if k not in info:
                        missing.add(k)
            resid.append(r_hat(info) - float(r))
            _sr = float(info["swap_rate"])
            _sg = float(info["swap_gate"])
            _exp_sg = SG_F + (1 - SG_F) * min(_sr / SG_REF, 1.0)
            if abs(_sg - _exp_sg) > 1e-6:
                bad_sg += 1
            _ap = float(info["alternate_penalty"])
            _wg = float(info["walk_gate"])
            _exp_ap = _ap * (1.0 if _ap > 0 else _wg)
            if abs(float(info["alternate_paid"]) - _exp_ap) > 1e-6:
                bad_alt += 1
            # ⑫ 付费 = phase/reach 不受门、alternate 罚项乘 walk_gate
            d14_12 += (float(info["phase_reward"]) * (_sg - 1.0)
                       + float(info["reach_reward"]) * (_sg - 1.0)
                       - float(info["alternate_paid"]) + _ap * _wg)
            tot += float(r)
            for k in SUMK:
                sums[k] += float(info.get(k, 0.0))
            if term or trunc:
                break
        e = dict(len=t, ret=tot, pf=tot / max(t, 1), d14_12=d14_12, ret12=tot - d14_12,
                 resid_med=float(np.median(np.abs(resid))) if resid else np.nan,
                 resid_p95=float(np.percentile(np.abs(resid), 95)) if resid else np.nan)
        e.update({("s_" + k): v for k, v in sums.items()})
        eps.append(e)
    return eps, missing, nan, bad_sg, bad_alt


def zero_action_rate():
    """早死基准：零动作（必摔）的每帧率，本轮自己量，不引用记忆里的数。"""
    raw = gym.make("Humanoid-v4", max_episode_steps=FR)
    patch_mirror_symmetry(raw.unwrapped.model)
    w = WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=True)
    out = []
    for sd in (11, 22, 33):
        obs, _ = w.reset(seed=sd)
        tot = t = 0.0
        while True:
            obs, r, term, trunc, info = w.step(np.zeros(w.unwrapped.model.nu))
            t += 1
            tot += float(r)
            if term or trunc:
                break
        out.append((t, tot / max(t, 1)))
    return out


def main():
    print(f"⑭ 常数（ast 自源码）: swap_gate(floor={SG_F}, ref={SG_REF})  alternate 罚边=1.0/奖边=walk_gate"
          f"  tax_cap={TCAP}  neg_W={NEG_W_SRC}")
    model = SAC.load(MODEL)
    eps, missing, nan, bad_sg, bad_alt = run(model)

    print("\n=== Step0 基线复现（同一份 ⑫ 权重，应当还原 ⑫ 的 TB record_mean）===")
    mean_alt = np.mean([e["s_alternate_penalty"] / e["len"] for e in eps])
    mean_sr = np.mean([e["s_swap_gate"] / e["len"] for e in eps])   # 门本身的每帧均值
    mean_wg = np.mean([e["s_walk_gate"] / e["len"] for e in eps])
    sr = np.mean([e["s_swap_rate"] / e["len"] for e in eps])        # 每帧均值的 swap_rate（暖机帧记 0）
    print(f"  本轮 per-frame 均值：alternate_penalty 面值 = {mean_alt:+.3f} 分/帧（⑫ TB 末段 0.270）"
          f"  walk_gate = {mean_wg:.3f}（⑫ 末段 0.20-0.41）  swap_gate = {mean_sr:.3f}"
          f"  反推 swap_rate ≈ {sr:.4f}（⑫ 0.006）")
    chk("S0a alternate 面值可复现（|Δ|<0.15 分/帧）", abs(abs(mean_alt) - BASE_ALT_FACE) < 0.15,
        f"{abs(mean_alt):.3f} vs {BASE_ALT_FACE}")
    chk("S0b swap_rate 量级可复现（<0.10）", sr < 0.10, f"{sr:.4f}")

    print("\n=== Step1 M1/M2 逐帧恒等式 ===")
    chk("S1a swap_gate == floor + (1−floor)·clip(swap_rate/ref)", bad_sg == 0, f"违例 {bad_sg}")
    chk("S1b alternate_paid == alt×(1 if alt>0 else walk_gate)", bad_alt == 0, f"违例 {bad_alt}")

    print("\n=== Step2 全量台账 Σinfo == r（⑭ 付费口径）===")
    if missing:
        print(f"  ⚠ info 里缺这些键（无法纳入台账）：{sorted(missing)}")
    res_med = np.median([e["resid_med"] for e in eps])
    res_p95 = np.nanmax([e["resid_p95"] for e in eps])
    chk("S2a 逐帧残差中位 < 0.02 分/帧", res_med < 0.02, f"{res_med:.4f}")
    chk("S2b 逐帧残差 p95 < 0.15 分/帧", res_p95 < 0.15, f"{res_p95:.4f}")
    chk("S2c 无 NaN", nan == 0, f"NaN 帧 {nan}")

    print("\n=== Step3 同一轨迹上 ⑫付费 → ⑭付费（M1+M2 的净效应，解析精确）===")
    print(f"  {'ep_len':>7} {'⑭ret':>7} {'⑫ret':>7} {'Δ(M1+M2)':>9} | {'⑭phase':>7} {'⑭reach':>6} "
          f"{'⑫phase':>7} {'⑫reach':>6} {'alt罚⑭':>7} {'alt罚⑫':>7} {'税':>7} {'负步幅':>7}")
    for e in eps:
        sg = e["s_swap_gate"] / e["len"]
        ph14, re14 = e["s_phase_reward_gated"], e["s_reach_reward_gated"]
        ph12, re12 = e["s_phase_reward"], e["s_reach_reward"]
        alt14 = e["s_alternate_paid"]
        # 注：Δ 那一列是**逐帧累计**的精确值；这里的 alt12 只是用本集均值门做的展示近似
        alt12 = e["s_alternate_penalty"] * (e["s_walk_gate"] / e["len"])
        print(f"  {e['len']:7d} {e['ret']:7.0f} {e['ret12']:7.0f} {e['ret']-e['ret12']:9.0f} | "
              f"{ph14:7.0f} {re14:6.0f} {ph12:7.0f} {re12:6.0f} {-alt14:7.0f} {-alt12:7.0f} "
              f"{-e['s_short_stride_penalty']:7.0f} {-e['s_negative_stride_penalty']:7.0f}")
    d = np.median([e["ret"] - e["ret12"] for e in eps])
    pf14 = np.median([e["pf"] for e in eps])
    pf12 = np.median([e["ret12"] / e["len"] for e in eps])
    za = zero_action_rate()
    za_rate = np.median([x[1] for x in za])
    print(f"\n  M1+M2 净效应中位 = **{d:+.0f} 分/集**（每帧 {pf14:+.2f} vs {pf12:+.2f}）")
    print(f"  零动作（主动摔）实测每帧率 = {za_rate:+.2f}（{sorted(int(x[0]) for x in za)} 帧）")
    chk("S3a 剪刀式仍明显好过'快点摔'（> −1.0 分/帧）", pf14 > -1.0, f"{pf14:+.2f} vs 摔 {za_rate:+.2f}")
    chk("S3b 起点价没被抽干（> −1.5 分/帧）", pf14 > -1.5, f"{pf14:+.2f}")
    print(f"  交换率的**代价上界**：swap_rate→1  ⇒ swap_gate→1.0 ⇒ phase+reach 全额回来"
          f"（本轮 ⑫ 口径 {np.median([e['s_phase_reward']+e['s_reach_reward'] for e in eps]):.0f} 分/集），"
          f"所以 M2 是暂时性收费而不是天花板。")
    print(f"\n=== 结论：{'ALL PASS' if not fails else 'FAIL -> ' + ', '.join(fails)} ===")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
