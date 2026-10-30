#!/usr/bin/env python3
"""只读汇总：把各阶段归档的 det eval 日志按同一口径压成一张表（不改任何盘上文件）。

两种日志口径，分开标注，绝不混用：
  · det  = eval_logs/eval_gait.jsonl 系（确定性评估，5 集/次，一行一次评估）
  · best = best_extra_logs.jsonl 系（只在刷新最佳模型时写，随机训练口径）

用法: python3 debug_probes/summarize_gait_stages.py [--rows 250]
"""
import argparse
import json
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DT = 0.015  # Humanoid-v4 默认 frame dt；land_per_s 缺失时用它从 ep_landings 反算

DET_STAGES = [
    ("⑧ 腾空税 2M",       "best_model_saved/round2/SAC_18/18_加重腾空惩罚/eval_gait.jsonl"),
    ("⑪·T1 模板 1.55M",   "best_model_saved/round2/SAC_19/19_1/eval_gait.jsonl"),
    ("⑫ 碎步税 4M",       "best_model_saved/round2/SAC_19/19_2_小碎步惩罚4M/eval_gait.jsonl"),
    ("⑬⑭ 续训 1.86M",     "best_model_saved/round2/SAC_19/19_3_实现原地踏步交替/eval_gait.jsonl"),
    ("⑮ 续训 1.19M",      "best_model_saved/round2/SAC_19/19_4_调权重_结果还是原地踏步速度慢了/eval_gait.jsonl"),
    ("⑯ T2 4M",           "best_model_saved/round2/SAC_19/19_5_厥着腚高速原地踏步/eval_gait.jsonl"),
    ("⑰ place 换轨 1.86M", "eval_logs/eval_gait.jsonl"),
]

BEST_STAGES = [
    ("V9 交替慢跑",  "best_model_saved/round1/V9/V9-实现交替慢跑/best_extra_logs.jsonl"),
    ("V10 15_3",     "best_model_saved/round1/V10/15_3/best_extra_logs.jsonl"),
    ("⑨ 从头训",     "best_model_saved/round2/SAC_18/18_8_从头训/best_extra_logs.jsonl"),
]

# det 口径要报的列：(表头, 取值函数)
DET_COLS = [
    ("ep_len",    lambda r: r.get("ep_len")),
    ("ret",       lambda r: r.get("ret")),
    ("speed",     lambda r: r.get("speed")),
    ("land/s",    lambda r: r.get("land_per_s") if r.get("land_per_s") is not None
                  else (r["ep_landings"] / (r["ep_len"] * DT) if r.get("ep_landings") and r.get("ep_len") else None)),
    ("swap",      lambda r: r.get("swap_rate")),
    ("ds",        lambda r: r.get("double_support_frac")),
    ("flight",    lambda r: r.get("flight_frac")),
    ("adv_L",     lambda r: r.get("ev_step_adv_L_mean")),
    ("adv_R",     lambda r: r.get("ev_step_adv_R_mean")),
    ("sp_L",      lambda r: r.get("ev_spacing_L_p50")),
    ("sp_R",      lambda r: r.get("ev_spacing_R_p50")),
    ("rom_L",     lambda r: r.get("ev_rom_L_mean")),
    ("rom_R",     lambda r: r.get("ev_rom_R_mean")),
    ("st_L",      lambda r: r.get("stance_frac_L")),
    ("st_R",      lambda r: r.get("stance_frac_R")),
    ("slip_L",    lambda r: r.get("ev_slip_L_mean")),
    ("slip_R",    lambda r: r.get("ev_slip_R_mean")),
    ("lift_L",    lambda r: r.get("ev_lift_L_mean")),
    ("reach_L",   lambda r: r.get("ev_reach_L_mean")),
    ("reach_R",   lambda r: r.get("ev_reach_R_mean")),
    ("abd贴限",   lambda r: r.get("abd_y_near_limit_frac")),
    ("place",     lambda r: r.get("place_operand_mean")),
]

BEST_COLS = [
    ("step",      lambda r: r.get("timestep")),
    ("ret",       lambda r: r.get("eval/mean_reward")),
    ("ep_len",    lambda r: r.get("eval/mean_ep_length")),
    ("speed",     lambda r: r.get("gait/speed")),
    ("swap",      lambda r: r.get("gait/position_swap_rate")),
    ("stride",    lambda r: r.get("gait/stride_length")),
    ("L_stride",  lambda r: r.get("gait/last_left_stride")),
    ("R_stride",  lambda r: r.get("gait/last_right_stride")),
    ("L_swing",   lambda r: r.get("gait/mean_left_swing")),
    ("R_swing",   lambda r: r.get("gait/mean_right_swing")),
    ("fwd_gate",  lambda r: r.get("gait/forward_gate")),
]


def load(path):
    p = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if not os.path.exists(p):
        return None
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def med(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return st.median(vals) if vals else None


def fmt(v, w=7):
    if v is None:
        return " " * (w - 1) + "-"
    if abs(v) >= 1000:
        return f"{v:>{w}.0f}"
    return f"{v:>{w}.3f}"


def table(stages, cols, tail):
    out = []
    for name, rel in stages:
        rows = load(rel)
        if rows is None:
            out.append((name, len("缺失"), {c[0]: None for c in cols}))
            continue
        win = rows[-tail:] if tail else rows
        out.append((name, len(rows), {c[0]: med([c[1](r) for r in win]) for c in cols}))
    heads = [c[0] for c in cols]
    width = max(len(n) for n, _, _ in out) + 2
    line = "阶段".ljust(width) + f"{'行数':>5}  " + "".join(f"{h:>7}  " for h in heads)
    print(line)
    print("-" * len(line))
    for name, n, d in out:
        print(name.ljust(width) + f"{n:>5}  " + "".join(fmt(d[h]) + "  " for h in heads))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=250, help="取末 N 行（一行 = 一次 5 集确定性评估）")
    a = ap.parse_args()
    print(f"=== det 口径（eval_gait.jsonl，末 {a.rows} 行中位数）===")
    table(DET_STAGES, DET_COLS, a.rows)
    print()
    print(f"=== best 口径（best_extra_logs.jsonl，末 {a.rows} 行中位数；仅刷新最佳时写，随机口径）===")
    table(BEST_STAGES, BEST_COLS, a.rows)
