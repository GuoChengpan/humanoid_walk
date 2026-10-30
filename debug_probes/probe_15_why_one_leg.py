"""为什么模板把左右 hip_y 都写死了反相，策略却只让一条腿在走？

只用 TB 末段（当前 rollout）的**分腿**仪表，逐项检查"对称的输入 → 不对称的输出"发生在哪一级：
  ① 输入级：模板给两腿的控制量是否真的对称（osc_active_L/R、rom、swing_steps）
  ② 关节级：髋动了，膝有没有跟着伸（last_rom_* vs knee 相关诊断）
  ③ 结果级：落点间距/前移是否一正一负
  ④ 价格级：专门管弱腿的那几项（swing_min / stride_balance / support_sym / activity_diff）实付多少钱，
     以及它们各自被哪个门乘掉了 —— 若弱腿通道的钱 ≈ 0，"另一条腿不摆"就是**没有代价**，不是没被驱动。

不做物理、不加载权重 ⇒ 结论只到"钱挂在哪个量上"这一层，不预测训练结果。
"""
import collections
import glob
import os
import sys

import numpy as np

ROOT = "/home/guo/workspace/humanoid_walk"
KEEP = 3000

TAGS = [
    # ① 输入：模板对两腿的接管程度
    "gait/osc_active_L", "gait/osc_active_R", "gait/osc_k_now", "gait/osc_phase",
    "gait/swing_steps_L", "gait/swing_steps_R",
    # ② 关节：髋摆幅（模板能决定的）与腿的整体摆动
    "gait/last_rom_L_deg", "gait/last_rom_R_deg",
    "gait/last_drive_L", "gait/last_drive_R",
    # ③ 结果：落点前移/间距
    "gait/last_reach_L", "gait/last_reach_R",
    "gait/last_left_stride", "gait/last_right_stride",
    "gait/last_stride_bonus",
    # ④ 价格：弱腿/对称/停滞相关通道，以及乘掉它们的门
    "gait/swing_min_reward", "gait/swing_min_reward_gated",
    "gait/stride_balance_penalty", "gait/support_sym_penalty",
    "gait/activity_difference", "gait/activity_diff_paid",
    "gait/stall_penalty", "gait/stall_max",
    "gait/walk_gate", "gait/swap_gate", "gait/forward_gate", "gait/dense_ds_factor",
]


def find_latest():
    c = []
    for f in glob.glob(os.path.join(ROOT, "sac_humanoid_walk", "**", "events.out.tfevents*"), recursive=True):
        if os.path.getsize(f) < 5000:
            continue
        with open(f, "rb") as fh:
            if b"gait/last_reach_L" not in fh.read(4_000_000):
                continue
        c.append((os.path.getmtime(f), f))
    if not c:
        return None
    return max(c)[1]


def read(path, tags):
    from tensorboard.backend.event_processing import event_file_loader
    out = collections.defaultdict(lambda: collections.deque(maxlen=KEEP))
    step = 0
    for ev in event_file_loader.EventFileLoader(path).Load():
        if not ev.summary.value:
            continue
        step = max(step, ev.step)
        for v in ev.summary.value:
            if v.tag in tags:
                if v.tensor.float_val:
                    out[v.tag].append(float(v.tensor.float_val[0]))
                elif v.tensor.double_val:
                    out[v.tag].append(float(v.tensor.double_val[0]))
                elif v.HasField("value"):
                    out[v.tag].append(float(v.value))
    return {k: np.array(v) for k, v in out.items()}, step


def main():
    p = find_latest()
    if p is None:
        print("没有含分腿仪表的事件文件")
        return 1
    M, step = read(p, TAGS)
    print(f"TB={os.path.relpath(p, ROOT)}  末步={step/1e6:.2f}M env 步  每 tag 取末 {KEEP} 点")
    miss = [t for t in TAGS if t not in M]
    if miss:
        print("  缺 tag（这轮没写进 TB）：", ", ".join(t.replace('gait/', '') for t in miss))

    def g(t):
        return float(np.mean(M[t])) if t in M else float("nan")

    print("\n=== ① 输入级：模板对两腿是否对称 ===")
    for t in ("osc_active_L", "osc_active_R", "osc_k_now", "swing_steps_L", "swing_steps_R"):
        print(f"  {t:16s} {g('gait/' + t):8.4f}")
    print(f"  ⇒ 设计值 osc_active≈0.367（22/60 摆相窗）；两腿若相等，说明**命令层面没有偏心**")

    print("\n=== ② 关节级：髋摆动了，腿有没有真的伸出去（度）===")
    for side in ("L", "R"):
        print(f"  {side}: last_rom={g('gait/last_rom_' + side + '_deg'):6.1f}°  "
              f"last_drive={g('gait/last_drive_' + side):+7.4f}  "
              f"reach(脚相对大腿)= {g('gait/last_reach_' + side) * 100:+6.1f} cm")
    print(f"  ⇒ 若一侧 rom 大而 reach 小 = 大腿在空摆、膝没伸 ⇒ 模板管不到膝，而**膝无价**（全文件 grep 只在诊断里）")

    print("\n=== ③ 结果级：两脚的落地前后关系（cm，带符号）===")
    sl, sr = g("gait/last_left_stride") * 100, g("gait/last_right_stride") * 100
    print(f"  left_x−right_x @左落地 = {sl:+6.1f}   right_x−left_x @右落地 = {sr:+6.1f}"
          f"   人体两值都 ≈ +44.8")
    print(f"  forward_gate={g('gait/forward_gate'):.3f} ⇒ 门看的是 min(上两值)，任一为负整个门=0")

    print("\n=== ④ 价格级：管弱腿的那几项实付多少（分/帧）===")
    sm_raw, sm_g = g("gait/swing_min_reward"), g("gait/swing_min_reward_gated")
    rows = [("swing_min_reward(弱腿摆幅)", sm_raw, sm_g, "walk_gate"),
            ("stride_balance(步幅对称)", g("gait/stride_balance_penalty"), g("gait/stride_balance_penalty"), None),
            ("support_sym(支撑不对称)", g("gait/support_sym_penalty"), g("gait/support_sym_penalty"), None),
            ("activity_diff(两腿活动差)", g("gait/activity_difference"), g("gait/activity_diff_paid"), None),
            ("stall(原地停滞)", g("gait/stall_penalty"), g("gait/stall_penalty"), None)]
    for nm, raw, paid, gate in rows:
        extra = "" if gate is None else f"   ×{gate}={g('gait/' + gate):.3f} ⇒ 付 {paid:+.4f}"
        print(f"  {nm:26s} 面值={raw:+8.4f}{extra}")
    print(f"  参照：原地可得的大头 phase={g('gait/phase_reward'):+.4f}、"
          f"height+survival={g('gait/height_reward_gated') + g('gait/survival_bonus_gated'):+.4f}"
          f"、reach={g('gait/reach_reward_gated'):+.4f}")
    print(f"  门实况 walk_gate={g('gait/walk_gate'):.3f} swap_gate={g('gait/swap_gate'):.3f} "
          f"dense_ds={g('gait/dense_ds_factor'):.3f} forward_gate={g('gait/forward_gate'):.3f}")
    print("\n判读线（事前写的）：①对称 且 ③一正一负 且 ④弱腿通道实付 < 0.05/帧 ⇒ "
          "『一条腿不摆』不是驱动失败，而是它**既不被驱动也不需要付代价**。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
