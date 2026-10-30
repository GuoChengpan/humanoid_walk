# 注意（2026-10-22）：日志精简后 duty_sum / osc_k_now / *_gated / asym_* / event_factor /
#   ds_factor 已不再记录；本脚本的判据与面板固定于批次⑦/⑤，缺 key 的段落会显示"数据不足"。
"""隔夜跑完后的晨读：一条命令给出趋势 + 预先钉死的判定规则，不需要人再解读。

用法：/home/guo/tools_envs/mujoco_base/bin/python3 debug_probes/morning_readout.py
判定规则是 2026-10-10 睡前公开写死的（不是事后找理由），见 VERDICT_RULES。
"""
import glob
import json
import os
import subprocess

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = "/home/guo/workspace/humanoid_walk"
# 必须与 train_humanoid_sac.py 的 stride_ds_floor 一致（(乙.2) 的保底），C4 用它判断钱有没有换边
ENV_DS_FLOOR = 0.35
# 【批次⑦·(丁)】三条阈值，全部由 `verify_batch_7_osc.py` 在**未训练 752K + 部署路径**上量出来后写死：
#   C1 = 被动摆幅 5.80° + 3.0°（绝对阈值会被振荡器自己刷过去 ⇒ 假阳性）
#   C2 = 被动 ds 实测恒 0.000 ⇒ 双支撑只能靠学，门槛取 0.05
#   C3 = 新基线 k=0.30 det 1905 的 80%（接口改了 ⇒ 与历史 1852/1868 不可比）
C1_DET_ROM_MIN = 8.8
C2_DS_MIN = 0.05
C3_DET_EVAL_MIN = 1524.0

KEY_GAUGES = [
    ("gait/single_frac", "单支撑占比", "★计价点 走路≈0.76 / 弹跳0.47 / 同相跳0 / 蹭地0"),
    ("gait/walk_gate", "收入门控", "★B1梯级 lo0.48→ref0.62 floor0.35"),
    ("gait/flat_gate", "平地二层门控", "★=0.6+0.4·walk_gate 管 height+survival"),
    ("gait/duty_sum", "两腿占空比和", "=1−flight+ds；B1 要它涨"),
    ("gait/double_support_frac", "双支撑", "旁证，不再计费"),
    ("gait/ev_reach_L_mean", "左腿落点超前", "★floor=−0.20 目标≥−0.05"),
    ("gait/osc_k_now", "⑦振荡器混合权重", "★开关：应=0.30（200K 前锁定），课程终点 0.35"),
    ("gait/osc_active", "⑦左脚空中占比", "★≈0.4-0.5 才对；≈0 说明注入没在动"),
    ("gait/asym_inject_rate", "(丙)集级注入率", "★⑦ 已回滚 ⇒ 应为 0；非 0 说明代码没生效"),
    ("gait/ev_rom_L_mean", "左腿摆动行程°", "★⑦ 用**相对**判据：被动基线 5.80°+3°=8.8°（det 尺）"),
    ("gait/asym_inject_rate", "集级注入率", "★(丙) 开关：应≈0.10；过低=注入没在工作"),
    ("gait/asym_injected", "注入步占比", "被短集稀释，~0.02 是正常的，别拿它当 p"),
    ("gait/event_factor", "⑥时长折算因子", "★≈1=REF 正好是弹跳周期；>1 说明这集落地偏少"),
    ("gait/ds_factor", "⑥ds加权", "★贴 0.35=钱还在弹跳侧；离开 0.35 才是 (乙.2) 生效"),
    ("gait/swing_min_reward_gated", "⑤发的钱/步", "×1500=分/集；开跑前实测 8.8（旧 floor 下是 2.1）"),
    ("gait/reach_reward", "reach 项/步", "×1500=每集的钱"),
    ("gait/flight_frac", "双腾空", "≤0.35 为佳"),
    ("gait/ev_spacing_L_negfrac", "左脚落地间距为负占比", "B2 定标用"),
    ("gait/ds_landing_rate", "双支撑期落地占比", "stride 族的门"),
    ("gait/height_reward_gated", "高度实发/步", "×1500；B1 之后会掉，涨回来=门控在放行"),
    ("rollout/ep_len_mean", "存活步数", ">800 且上行；<200 触发 E 近崩处置"),
    ("gait/ev_slip_L_mean", "左腿真滑移", "单脚拖行盆地先在这露头"),
    ("gait/lateral_penalty", "侧移罚", "横蹭会抬升"),
    ("gait/speed", "速度", "<0.5 = 爬行盆地"),
    ("gait/support_sym_penalty", "对称罚/步", "200步窗"),
    ("gait/hip_xz_angle_penalty", "髋税/步", "最大单税"),
    ("gait/survival_bonus", "生存收入/步", "①新补：与 height 之和=换步态的赌注"),
    ("gait/pitch_angle", "躯干前倾", "旁证：弹跳动力学的副产品，与 reach 无套利关系（§20.3 已证伪）"),
    ("train/exploration_std", "探索噪声", "必须留在 0.15-0.25"),
    ("train/critic_loss", "critic 损失", "还在追=没收敛"),
]

VERDICT_RULES = """
批次⑦·(丁) 左髋振荡器混合控制 判定规则 —— 2026-10-11 23:40 观察前钉死，事后不挪门槛：
  内容：`a[left_hip_y] ← clip((1−k)·a_net + k·(−A·sin(πφ)))`，φ 由接触时钟推进、落地归零；
        A=0.55、k 由 OscCurriculum 按全局步数推：**前 200K 锁 0.30，之后 400K 步升到 0.35**。
        训练与评测吃同一个 k（评的是部署控制器）。**镜像整集重放本轮关闭**（with_mirror=False，
        否则镜像集会在同一侧再注入一次）；`patch_mirror_symmetry` 保留。奖励一个常数没动。
  立项依据（§30，未训练 752K 上实测）：加性残差被平衡解原地抵消（A≤0.25 ⇒ rom p90 2.9→3.7° 而分/步反升）；
        覆盖式混合 k=0.30-0.35 未训练就跑满 1500、rom_L p90 10.0-11.2°、reach_L −0.154→−0.114、
        分/步 1.24→1.27-1.36 ⇒ **接口本身不花钱**；k≥0.40 两条路径分歧（离线扫描 318 步摔、
        wrapper 1452-1500 存活）⇒ 上限取两条路的交集 0.35。
  **起点值（752K，det × 两种子，wrapper 部署路径实测，A=0.55）**：
        k=0.00（被动对照）：总分 1859、ev_rom_L 2.32°、ds 0.000、reach_L −0.154
        k=0.30（开跑档位）：总分 **1905**、**ev_rom_L 5.80°** ← 这是"策略啥也没学、光靠脉冲"的摆幅
                            ds **0.000**、reach_L −0.124、分/步 1.27
  ⇒ **C1 必须是相对量**（绝对阈值会被脉冲自己刷过去 = 假阳性，这是我原本写 8° 的漏洞）：
  首判 **200K**（此时 k 仍锁 0.30，被动摆幅就是 5.80°），四条**全过**才续到 600K：
     C1 det `ev_rom_L_mean` ≥ **8.8°**（= 被动 5.80 + 3.0；多出来的 3° 必须是策略自己挣的）
     C2 det `ds`            ≥ **0.05**（被动实测 0.000，振荡器造不出双支撑 ⇒ 这条只能靠学，是本批真问题）
     C3 det eval            ≥ **1524**（= 新基线 1905 的 80%；接口改了，与历史 1852/1868 不可比）
     C4 开关：`gait/osc_k_now` = 0.30±0.01 且 `gait/osc_active` ≈0.4-0.5 且 `gait/asym_*` 全 0（确认⑥已回滚）
  600K 复判：C1/C2 必须**仍在**（"留住"才算，出现过又消失 = 不算过，这是 §27/§29 两轮的教训）。
  红线 E：ep_len_mean<200 且 eval 连续 3 次<0 → 立即停；`exploration_std` 看方向不看绝对值。
  判决一律用 det eval + 跨窗均值；TB 的 ev_rom_L 含探索噪声，不得当判决用。
  任一不过：**停 (丁)**（osc_enable=False 一行回滚），且**不再往上叠机制**——那说明在这个
  17 维 MLP + 本任务定义上，交替行走不是价格、探索、左腿接口能喂出来的，需要换问题
  （更小模型 / 分段技能 / 参考轨迹模仿），那是重新定目标，要用户决定。
  上一批（批次⑥ (乙)+(丙)）于 502K 判不过：C1 det rom_L 2.31°、C2 ds 0.001、C3 eval 1567 过、
  C4 ds_factor 0.451 过；已按顺序回滚（验收：det 回到 1852/1868，与改造前逐位相同）。见 §29/§30。
"""


def newest_event_file():
    """挑当前这轮的 TB。两个坑：
     1) 新 run 刚开时文件还没写够字节（旧代码 size>1000 的过滤会把它跳过，于是读到**归档 run**的 TB）；
     2) SB3 的文件名里就带 PID（events.out.tfevents.<ts>.host.<PID>.0），所以能直接和活着的进程对齐。
    """
    pid = subprocess.run(["pgrep", "-f", "train_humanoid_sac.py"],
                         capture_output=True, text=True).stdout.split()
    cands = glob.glob(os.path.join(ROOT, "sac_humanoid_walk/*/events.out.tfevents*"))
    for p in sorted(cands, key=os.path.getmtime, reverse=True):
        fields = os.path.basename(p).split(".")
        if len(fields) >= 3 and fields[-2] in pid:     # 末两段是 <PID>.<序号>
            return p
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)            # 没在跑：退回"最近写过的那个"（归档 run 的末态）


def det_evals(n=3, keys=("ev_rom_L_mean", "ev_rom_R_mean", "single_frac", "flight_frac",
                         "ev_reach_L_mean", "ret", "ep_len", "speed")):
    """确定性 eval 的最近 n 次均值——批次⑤/⑥ 的主判据只能用这个，不能用训练 TB。
    原因：⑥ 把 ent_coef 提到 0.10 之后，训练窗里的 ev_rom_L 被探索噪声抬高（实测 det 2.5° vs TB 7.3°）。
    坑：jsonl 无 step/run 字段、路径写死、新 run 会**追加**在旧 run 后面 → 这里取末尾，
    跨 run 用之前要先数行数对不对得上 TB 的 eval 点数。"""
    log = os.path.join(ROOT, "eval_logs/eval_gait.jsonl")
    if not os.path.exists(log):
        return None
    rows = [json.loads(l) for l in open(log) if l.strip()]
    if len(rows) < 5:
        return None
    tail = rows[-5 * n:]
    out = {}
    for k in keys:
        v = [r[k] for r in tail if k in r]
        out[k] = float(np.mean(v)) if v else float("nan")
    return out


def batch2_effect_split(ea, tags):
    """生效性分叉（原本是批次② 的，2026-10-11 改挂到 ④.3+⑤+⑥；只用跨窗均值，不用窗内斜率）。

    ⚠ 原来第②支写的是"加飞行税探针 w 0.25→0.6"——**那条已经被证伪两次**（w=0.25 时门控每 1pp
    只值 3.75 分/集，是 gate 的 1/6；w=0.5 跑满 2M 步 flight 仍 0.46→0.48）。这里不再推荐它。
    现在看的是：这批的钱有没有进梯度（flight 降 / single 升），以及 ⑤ 的通道有没有被拿到。
      ① flight 在降且 single 在升 → 生效，**什么都不加**，等 600K 按 det rom_L 判
      ② flight 不降、⑤ 的钱也没被拿到（swing_min_reward_gated 掉回 0）→ 说明 ⑤ 又被绕开，
         按预定走 D：**不再改奖励**，改接口
      ③ flight 不降但 swing_min 的钱在涨 → 它学会了"原地甩腿拿钱"＝C 情形，收斜坡顶不是加税
    """
    def win(k, lo, hi):
        if k not in tags:
            return None
        s = ea.Scalars(k)
        st = np.array([x.step for x in s], float)
        v = np.array([x.value for x in s], float)
        m = (st >= lo) & (st < hi)
        return float(np.nanmean(v[m])) if m.sum() > 3 else None

    top = win("rollout/ep_len_mean", 0, 10 ** 9)
    last = max(x.step for x in ea.Scalars("gait/speed")) if "gait/speed" in tags else 0
    early, late = win("gait/speed", 0, 80000), win("gait/speed", max(0, last - 80000), last + 1)
    d0, d1 = win("gait/duty_sum", 0, 80000), win("gait/duty_sum", max(0, last - 80000), last + 1)
    f0, f1 = win("gait/flight_frac", 0, 80000), win("gait/flight_frac", max(0, last - 80000), last + 1)
    s0, s1 = win("gait/single_frac", 0, 80000), win("gait/single_frac", max(0, last - 80000), last + 1)
    if None in (early, late, d0, d1):
        print("\n>>> 生效性分叉：数据不足（前/后 80K 窗没填满）")
        return
    print(f"\n>>> 生效性分叉（前/后 80K 窗均值；当前 {last/1000:.0f}K）："
          f"speed {early:.3f}→{late:.3f} ({late-early:+.3f})，flight "
          f"{f0:.3f}→{f1:.3f} ({f1-f0:+.3f})，single {s0:.3f}→{s1:.3f} ({s1-s0:+.3f})，"
          f"duty_sum {d0:.3f}→{d1:.3f} ({d1-d0:+.3f})")
    up_single = s1 is not None and s0 is not None and s1 > s0 + 0.02
    down_flight = f1 is not None and f0 is not None and f1 < f0 - 0.02
    paid = win("gait/swing_min_reward_gated", max(0, last - 80000), last + 1)
    print(f"    ⑤ 这条通道最近 80K 实发 {0.0 if paid is None else paid * 1500:.0f} 分/集"
          f"（warm-start 离线 det 是 8.8 分/集）")
    if down_flight or up_single:
        print("    → ① 方向进了梯度：**什么都不加**，等 600K 用 det rom_L 正式判")
    elif paid is not None and paid * 1500 > 25:
        print("    → ③ 贴地没改善但 ⑤ 的钱拿得很多＝原地甩腿拿钱（C 情形）。"
              "⚠此分支是批次⑤ 时期的处置（收斜坡顶 14°→10°），⑤ 已落地并判完，**不是当前建议**；"
              "当前判决以本文件上方批次⑦ 的 C1-C4 为准")
    else:
        print("    → ② flight 不降、⑤ 也没被拿到 ⇒ 这批对步态无效。按预定**不加价**，")
        print("       等 600K 用 det rom_L 判 D（<4° 就停止改奖励、改接口）。")
        print("       旧文档里的『飞行税 w 0.25→0.6』不启用：它在本项目已被证伪两次（§12/§21）。")


def main():
    p = newest_event_file()
    if p is None:
        print("找不到 TB 事件文件")
        return
    running = subprocess.run(["pgrep", "-f", "train_humanoid_sac.py"],
                             capture_output=True, text=True).stdout.strip()
    print(f"TB 文件: {os.path.relpath(p, ROOT)}")
    print(f"训练进程: {'存活 PID=' + running.replace(chr(10), ',') if running else '已退出（可能跑满 2M 自行结束）'}")
    ea = EventAccumulator(p, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]

    def series(t):
        s = ea.Scalars(t)
        return np.array([x.step for x in s]), np.array([x.value for x in s])

    st, ev = series("eval/mean_reward") if "eval/mean_reward" in tags else (np.array([]), np.array([]))
    last_step = int(st[-1]) if len(st) else 0
    if len(st):
        s2, el = series("eval/mean_ep_length")
        print(f"\neval 轨迹（每 16K）: " + " ".join(f"{int(x/1000)}K:{y:.0f}" for x, y in zip(st, ev)))
        print(f"eval 最高 {ev.max():.0f} @ {int(st[ev.argmax()]/1000)}K | 末点 {ev[-1]:.0f} | "
              f"ep_len(评估) 末点 {el[-1]:.0f}")
    s, v = series("rollout/ep_len_mean")
    if len(v):
        last_step = max(last_step, int(s[-1]))
        print(f"\n训练 rollout 末 12 点 ep_len: {[int(x) for x in v[-12:]]}")
    print(f"当前步数 ≈ {last_step/1000:.0f}K / 2000K  (total_timesteps=2M，跑满会自行保存并退出)\n")

    print("%-27s %9s %9s   %s" % ("gauge", "首1/8均", "末1/8均", "备注"))
    vals = {}
    for t, name, note in KEY_GAUGES:
        if t not in tags:
            print("%-27s %9s %9s   %s" % (t, "-", "-", name + " — 键不存在"))
            continue
        s, v = series(t)
        k = max(1, len(v) // 8)
        e, l = v[:k].mean(), v[-k:].mean()
        vals[t] = l
        scale = 1500 if t in ("gait/reach_reward", "gait/support_sym_penalty",
                              "gait/hip_xz_angle_penalty", "gait/slip_penalty",
                              "gait/lateral_penalty", "gait/survival_bonus") else 1
        unit = "/集" if scale == 1500 else ""
        print("%-27s %9.4f %9.4f   %s  末值%.4f%s(%s)" % (
            t, e, l, name, l * scale, unit, note))

    ds, fl = vals.get("gait/double_support_frac", float("nan")), vals.get("gait/flight_frac", float("nan"))
    det = det_evals(3)
    print("\n>>> 判定（批次⑦·(丁) 左髋振荡器混合控制，2026-10-11 23:40 观察前钉死；**主判据=det eval**）：")
    gate = vals.get("gait/walk_gate", float("nan"))
    single = vals.get("gait/single_frac", float("nan"))
    reach = vals.get("gait/ev_reach_L_mean", float("nan"))
    slip = vals.get("gait/ev_slip_L_mean", float("nan"))
    spd = vals.get("gait/speed", float("nan"))
    duty_now = vals.get("gait/duty_sum", float("nan"))
    rom_tr = vals.get("gait/ev_rom_L_mean", float("nan"))
    rom_det = det["ev_rom_L_mean"] if det else float("nan")
    ds_det = (1 - det["flight_frac"] - det["single_frac"]) if det else float("nan")
    inj = vals.get("gait/asym_inject_rate", float("nan"))          # 集级率，应≈p
    inj_step = vals.get("gait/asym_injected", float("nan"))        # 步级率，被集长稀释（预期 ~2%）
    conv = vals.get("gait/asym_convert_rate", float("nan"))
    print(f"    左腿摆动行程 rom_L：det eval 最近3次均 {rom_det:.2f}° ｜ 训练 TB（含探索噪声+注入集）{rom_tr:.2f}°")
    print(f"    注入开关：集级率 {inj:.3f}（应≈0.10）｜ 步级占比 {inj_step:.3f}（被短集稀释，~0.02 正常）"
          f"｜ 注入集摔倒被转成截断的比例 {conv:.2f}")
    if det:
        print(f"    det 同尺旁证：flight {det['flight_frac']:.3f} single {det['single_frac']:.3f} "
              f"ds {ds_det:.3f} reach_L {det['ev_reach_L_mean']:+.3f} ret {det['ret']:.0f} "
              f"ep_len {det['ep_len']:.0f}")
    if last_step < 200000:
        print(f"    未到首判点（**(丙) 首判 200K**，因为它是 200K 步有界试点）：当前 {last_step/1000:.0f}K")
    elif not np.isfinite(rom_det):
        print("    数据不足：读不到 eval_logs/eval_gait.jsonl，不能按主判据判决（别拿训练值当代用）")
    elif not np.isfinite(inj) or (inj > 0.03 and inj < 0.2):
        # 注意 ⑦ 里注入是**故意关掉**的（asym_*=0 属于 C4 的正常状态），
        # 所以这里只在"注入率异常高/异常乱"时截胡；0 必须放行到四条件判决，否则⑦的判据永远印不出来。
        # 早先写的是 `inj < 0.03`，那是 ⑤ 时代的语义，直接把 ⑦ 的判决块吞掉了。
        print(f"    ★注入通道数值异常（集级率={inj:.3f}，既不是⑦要求的≈0，也不是⑤的≈0.10）"
              f"⇒ 先确认这轮代码/TB 归属，别拿它判 ⑦")
    else:
        dsf = vals.get("gait/ds_factor", float("nan"))
        # 批次⑦：C1 用**相对**阈值（被动摆幅 5.80° + 3.0°），绝对阈值会被脉冲自己刷过去
        osc_k = vals.get("gait/osc_k_now", float("nan"))
        osc_act = vals.get("gait/osc_active", float("nan"))
        inj_left = vals.get("gait/asym_inject_rate", float("nan"))
        c1, c2 = rom_det >= C1_DET_ROM_MIN, (det and ds_det >= C2_DS_MIN)
        c3 = det and det["ret"] >= C3_DET_EVAL_MIN
        # 期望的 k 用与 OscCurriculum 同一条排程算出来（别写死 0.30，200K 之后本来就在爬）
        prog = 0.0 if last_step <= 200_000 else min((last_step - 200_000) / 400_000, 1.0)
        EXPECTED_OSC_K = 0.30 + (0.35 - 0.30) * prog
        c4 = (np.isfinite(osc_k) and abs(osc_k - EXPECTED_OSC_K) <= 0.02
              and np.isfinite(osc_act) and 0.5 <= osc_act <= 0.9      # 左腿"非支撑"占比≈flight+右单撑≈0.75，不是 0.4-0.5
              and (not np.isfinite(inj_left) or inj_left < 0.01))
        print(f"    四条件（200K 观察前钉死）：C1 det rom_L≥{C1_DET_ROM_MIN}°(被动5.80+3) {c1}({rom_det:.2f}) ｜ "
              f"C2 det ds≥{C2_DS_MIN} {c2}({ds_det:.3f}) ｜ C3 det eval≥{C3_DET_EVAL_MIN:.0f} {c3}"
              f"({det['ret'] if det else float('nan'):.0f}) ｜ C4 开关 osc_k={osc_k:.2f} "
              f"空中占比={osc_act:.2f} 残留注入率={inj_left:.3f} {c4}")
        if c1 and c2 and c3 and c4:
            print("    → 通过：策略在脉冲之上**自己**多挣了摆幅并出现了双支撑，续跑到 600K；复判看还在不在")
        else:
            print("    → **不通过就停 (丁)**（`OSC_ENABLE = False` 一行回滚），且**不再叠机制**，")
            print("       转\"换问题\"（更小模型 / 分段技能 / 参考轨迹模仿）——那要用户重新定目标。")
    # 【E】注入试点专用红线：注入集灌进来的若是纯崩盘数据，ep_len/eval 会先垮
    duty = vals.get("gait/duty_sum", float("nan"))
    eln_g = vals.get("rollout/ep_len_mean", float("nan"))
    if np.isfinite(eln_g) and len(ev) >= 3:
        if eln_g < 200 and (ev[-3:] < 0).all():
            print(f"    ★情形 E（近崩回退，触发即停）：ep_len={eln_g:.0f}<200 且 eval 末3次="
                  f"{[int(x) for x in ev[-3:]]} 全<0 → asym_inject_p 回 0（或直接回滚 (丙)）重启")
        else:
            print(f"    E 未触发：ep_len={eln_g:.0f}、eval 末3次={[int(x) for x in ev[-3:]]}"
                  f"（起点=真 752K warm-start 的 det A/B：1852/1868 → (乙)后 1835/1839，见 §28.3-28.4）")
    # single = 1 − flight − ds = duty_sum − 2·ds（duty_sum = single + 2ds）
    if np.isfinite(duty) and np.isfinite(ds) and np.isfinite(fl):
        print(f"    single 的来源核对：duty_sum={duty:.4f}、ds={ds:.4f}、flight={fl:.4f} → "
              f"single={duty - 2 * ds:.4f}（=1−flight−ds={1 - fl - ds:.4f}）；"
              f"**duty_sum 在跌而 single 在涨 = 靠减小两脚重叠（ds）抬的，不等于贴地变多**")
    es = vals.get("train/exploration_std", float("nan"))
    sp = vals.get("gait/speed", float("nan"))
    eln = vals.get("rollout/ep_len_mean", float("nan"))
    trip = []
    if np.isfinite(es) and not (0.15 <= es <= 0.25):
        # ⑥ 之后 std 的**初值本来就在带外**（0.10 抬上去的），所以方向比绝对值重要：
        # 往下收回带内=正常退火，往上发散才是熵崩溃
        s_es = ea.Scalars("train/exploration_std")
        first = np.mean([x.value for x in s_es[:max(1, len(s_es) // 8)]])
        arrow = "↓正收回带内" if es < first else "↑仍在发散"
        trip.append(f"exploration_std={es:.3f} 出带（首1/8={first:.3f}，{arrow}）")
    if np.isfinite(sp) and sp < 0.5:
        trip.append(f"speed={sp:.2f} 掉进爬行区")
    if np.isfinite(eln) and eln < 150:
        trip.append(f"ep_len={eln:.0f} 崩回低位")
    print("    危险信号：" + ("；".join(trip) if trip else "无"))

    log = os.path.join(ROOT, "eval_logs/eval_gait.jsonl")
    if os.path.exists(log):
        rows = [json.loads(l) for l in open(log) if l.strip()]
        tail = rows[-5:]
        print(f"\n最近 5 个 eval 集（每 16K 有 5 集）: 步数/回报/腾空/双撑/单撑/reachL")
        for r in tail:
            print("  %5d步 %7.0f  腾空%.2f 双撑%.2f 单撑%.2f reachL%+.3f" % (
                r["ep_len"], r["ret"], r["flight_frac"], r["double_support_frac"],
                r.get("single_frac", float("nan")),
                r.get("ev_reach_L_mean", float("nan"))))
        print(f"（共 {len(rows)} 集 = {len(rows)//5} 次 eval）")
    batch2_effect_split(ea, tags)
    print(VERDICT_RULES)


if __name__ == "__main__":
    main()
