#!/usr/bin/env python3
"""把各阶段 TensorBoard 事件拼成 README 用的两张总览曲线。

  /home/guo/tools_envs/mujoco_base/bin/python3 debug_probes/export_curves.py

设计约束（都是这个项目踩过的坑，写死在代码里而不是写在文档里提醒人）：
  · **缺 tag 就空着并标"无此指标"，绝不画成 0** —— record_mean 缺失与"值真的是 0"是两回事。
  · **确定性评估（eval/*）与随机训练 rollout（rollout/*、gait/*）不同尺**，分面板画、不叠同一条轴。
  · **V6/V7 时代 max_episode_steps=4000、V8 起 1500** ⇒ 第一轮的单集长度面板在 1500 处画一条参考线，
    否则读者会把"上限改了"读成"退步了"。
  · `sac_humanoid_walk/SAC_19` 目录里同时有 ⑪ 与 ⑰ 两个 run 的事件文件 ⇒ 按文件名里的 PID 选，不按目录名。
  · 每个 run 的 x 轴是它自己的 env 步数（续训会重置 num_timesteps），所以颜色区分的是一条条**片段**，
    不是一条连续曲线；片段顺序 = 项目时间顺序。
"""

import glob
import os
import re
import sys
import statistics as st

from tensorboard.backend.event_processing import event_file_loader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 图里的中文要有 CJK 字体，否则 matplotlib 会静默用 DejaVu（方框）
for _fp in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc"):
    if os.path.exists(_fp):
        try:
            font_manager.fontManager.addfont(_fp)
            matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=_fp).get_name()
            matplotlib.rcParams["axes.unicode_minus"] = False
            break
        except Exception as _e:
            print(f"  [字体] {_fp}: {_e}")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "docs")

# ---------------- run 清单（label, 相对目录, 可选 PID 过滤）----------------
ROUND1 = [
    ("V6",        "sac_humanoid_walk/SAC_10", None),
    ("V7",        "best_model_saved/round1/V7/SAC_11", None),
    ("V8",        "best_model_saved/round1/V8/SAC_13", None),
    ("V9/14_2",   "best_model_saved/round1/V9/14_2/SAC_14_2", None),
    ("V9/14_3",   "best_model_saved/round1/V9/14_3/SAC_14_3", None),
    ("V9/14_4",   "best_model_saved/round1/V9/14_4/SAC_14_4", None),
    ("V9/14_5",   "best_model_saved/round1/V9/14_5/SAC_14_5", None),
    ("V9/14_6",   "best_model_saved/round1/V9/14_解决上半身朝右严重倾斜问题/SAC_14_6", None),
    ("V9 慢跑",    "best_model_saved/round1/V9/V9-实现交替慢跑/SAC_14", None),
    ("V10 软饱和", "best_model_saved/round1/V10/15_1/软饱和/SAC_15_1_软包和", None),
    ("V10 15_2",  "best_model_saved/round1/V10/15_2/SAC_1_5_2", None),
    ("V10 15_3",  "best_model_saved/round1/V10/15_3/SAC_15", None),
]

ROUND2 = [
    ("17_1",       "sac_humanoid_walk/TMP/SAC_17_1", None),
    ("17_2",       "sac_humanoid_walk/TMP/SAC_17_2", None),
    ("17_3",       "sac_humanoid_walk/TMP/SAC_17_3", None),
    ("17_4",       "sac_humanoid_walk/TMP/SAC_17_4", None),
    ("17_5",       "sac_humanoid_walk/TMP/SAC_17_5", None),
    ("17_6+swap",  "sac_humanoid_walk/SAC_17_6_的最终版_加positions_swap", None),
    ("17_7 门控0.08", "sac_humanoid_walk/SAC_17_7_将奖励权重+forward门控0.08", None),
    ("17_7 膝深折2.0", "sac_humanoid_walk/SAC_17_7_门控0.15_膝深折惩罚2.0", None),
    ("18_1 镜像",   "sac_humanoid_walk/SAC_18_1_镜像增广", None),
    ("18_4 降速",   "sac_humanoid_walk/SAC_18_4_降速+改接触判定+日志逻辑等", None),
    ("18_6",       "sac_humanoid_walk/SAC_18_6", None),
    ("18_8 从头训", "sac_humanoid_walk/SAC_18", None),
    ("19_2 碎步税",  "sac_humanoid_walk/SAC_19_2_小碎步惩罚4M", None),
    ("19_3 原地交替", "sac_humanoid_walk/SAC_19_3_实现原地踏步交替", None),
    ("19_4 调权重",  "sac_humanoid_walk/SAC_19_4_调权重_结果还是原地踏步速度慢了", None),
    ("19_5 撅腚",    "sac_humanoid_walk/SAC_19_5_厥着腚高速原地踏步", None),
    ("19 最新一轮",  "sac_humanoid_walk/SAC_19", "11744"),   # 目录里混了两个 run，按 PID 取
]

PANELS = {
    "eval/mean_reward":        ("评估分数（确定性策略 5 集）", None),
    "eval/mean_ep_length":     ("评估单集长度", 1500),
    "gait/speed":              ("速度 m/s（训练 rollout）", None),
    "gait/ev_step_adv_L_mean": ("左腿单步前进 m（功能量）", None),
    "gait/ev_step_adv_R_mean": ("右腿单步前进 m（功能量）", None),
    "gait/position_swap_rate": ("落地交换率（过程量）", None),
    "gait/double_support_frac": ("双脚同时着地占比", None),
    "gait/flight_frac":        ("腾空占比", None),
    "train/exploration_std":   ("探索噪声 std（持续上涨=崩溃前兆）", None),
}

# 第一轮没记录功能量/双支撑/腾空/探索噪声 ⇒ 只画它有数据的面板，别留一堆空格
ROUND_PANELS = {
    1: ["eval/mean_reward", "eval/mean_ep_length", "gait/speed", "gait/position_swap_rate"],
    2: list(PANELS),
}

NOTE = {
    1: ("V6/V7 时代 max_episode_steps=4000（V8 起改 1500）⇒ 左上分数与右上长度**不可与其它版本横比**；"
        "第一轮的 run 没有记录单步前进/双支撑/腾空。"),
    2: ("gait/* 来自**随机训练 rollout**：摔倒集里的正单步前进不可信（确定性口径的数值见 README 的形态表）。"
        "17_1-18_1 尚未记录单步前进/双支撑/腾空 ⇒ 那些面板只有后面的段。"),
}

SMOOTH = 15   # 滚动中位数窗口（点）；原始标量是逐步 record_mean，毛刺会盖住趋势


def pick_files(d, pid=None):
    """一个目录里可能有多个事件文件（同一条线的重启、或两个不同 run 混在同一目录）。"""
    cands = sorted(glob.glob(os.path.join(ROOT, d, "events.out.tfevents*")))
    found = sorted({re.search(r"\.(\d+)\.0$", c).group(1) for c in cands if re.search(r"\.(\d+)\.0$", c)})
    if pid:
        cands = [c for c in cands if f".{pid}.0" in c]
    if len(found) > 1 and not pid:
        print(f"  [注意] {d} 有 {len(found)} 个进程写的事件文件（续训接续），按 step 合并成一条线：PID={found}")
    return cands


def scalar_of(v):
    """Summary.Value 的标量藏在 simple_value（老写法）或 tensor（新写法）里。"""
    if v.simple_value != 0.0:
        return float(v.simple_value)
    try:
        if v.HasField("tensor"):
            import numpy as np
            t = v.tensor
            if t.tensor_content:
                dt = {1: np.float32, 2: np.float64}.get(t.dtype, np.float32)
                return float(np.frombuffer(t.tensor_content, dtype=dt).flat[0])
            if t.float_val:
                return float(t.float_val[0])
    except (ValueError, AttributeError):
        pass
    return 0.0


def read_tags(paths, wanted):
    """合并一个目录里的多个事件文件。

    实测这些文件是**同一血统的续训接续**（如 SAC_10：16K→2.99M，然后 2.69M→5.4M），所以合并成一条线是对的；
    少数目录里有小段重叠（同一目录被两个进程先后写过同一段 step）⇒ 按文件名顺序后者覆盖前者，保证可复现。
    """
    acc = {t: {} for t in wanted}
    for p in sorted(paths):
        for e in event_file_loader.EventFileLoader(p).Load():
            for v in (e.summary.value if e.summary else []):
                if v.tag in acc:
                    acc[v.tag][int(e.step)] = scalar_of(v)
    return {t: sorted(d.items()) for t, d in acc.items()}


def smooth(ys):
    """滚动中位数：原始标量是逐步 record_mean，毛刺会盖住趋势。"""
    import numpy as np
    a = np.asarray(ys, dtype=float)
    if len(a) < SMOOTH:
        return a
    w = np.lib.stride_tricks.sliding_window_view(a, SMOOTH)
    mid = np.median(w, axis=1)
    pad = SMOOTH // 2
    out = np.empty_like(a)
    out[pad:pad + len(mid)] = mid
    out[:pad] = mid[0]
    out[pad + len(mid):] = mid[-1]
    return out


def draw(round_no, round_name, runs, fname):
    tags = ROUND_PANELS[round_no]
    data = []
    for label, d, pid in runs:
        fs = pick_files(d, pid)
        if not fs:
            print(f"  [缺文件] {label}: {d}")
            continue
        dd = read_tags(fs, list(PANELS))
        data.append((label, dd))
        have = " ".join(t.split("/")[-1][:7] for t in tags if dd[t])
        print(f"  [读] {label}（{len(fs)} 个文件）: {have}")

    cols = 2 if len(tags) <= 4 else 3
    rows = (len(tags) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.6 * cols, 3.0 * rows))
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
    cmap = plt.get_cmap("tab20")

    for i, tag in enumerate(tags):
        ax = axes[i]
        title, ref = PANELS[tag]
        any_line = False
        for j, (label, dd) in enumerate(data):
            pts = sorted(dd.get(tag) or [])
            if len(pts) < 3:
                continue
            xs = [p[0] / 1e6 for p in pts]
            ys = smooth([p[1] for p in pts])
            ax.plot(xs, ys, lw=1.2, color=cmap(j % 20))
            any_line = True
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("该轮内部 env 步数（百万）", fontsize=8)
        ax.grid(alpha=.3)
        if ref:
            ax.axhline(ref, ls="--", lw=.9, c="k")
        if not any_line:
            ax.text(.5, .5, "这些 run 没有记录该指标", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="0.5")
    for k in range(len(tags), len(axes)):
        axes[k].axis("off")

    handles = [plt.Line2D([], [], color=cmap(j % 20), lw=2, label=lab) for j, (lab, _) in enumerate(data)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.004),
               ncol=min(6, len(handles)), fontsize=8, frameon=False)
    fig.suptitle(f"{round_name} —— 一条颜色 = 一次训练（x 为该轮内部步数，续训会重新计数；曲线为 {SMOOTH} 点滚动中位数）",
                 fontsize=12)
    fig.text(.012, .072, "注：" + NOTE[round_no], fontsize=7.5, color=".35", ha="left", va="bottom")
    fig.tight_layout(rect=(0, .125, 1, .95))
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, fname)
    fig.savefig(p, dpi=130)
    print(f"  → {p}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "1"):
        print("第一轮：")
        draw(1, "第一轮（V6→V10，奖励重构期）", ROUND1, "curve_round1.png")
    if which in ("all", "2"):
        print("第二轮：")
        draw(2, "第二轮（SAC_17→SAC_19，降速与鼓励迈步）", ROUND2, "curve_round2.png")
