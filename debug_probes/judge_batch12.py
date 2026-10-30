"""批次⑫ 的判决脚本。阈值全部来自 CONTEXT_HANDOFF §46（K1'-K6'）+ §50.2（⑫ 新增两条），
**写在观察之前**，事后不挪门柱。

设计要点（都是历史踩坑换来的）：
1. **不靠 --since**。⑫ 的 eval 行带了三个旧轮次没有的键（`land_per_s / tiny_step_frame_frac /
   dense_ds_factor`）⇒ 用"第一个含这些键的行"自动定位本轮起点；jsonl 被归档过也照样对。
   （⑪ 那次 --since 给错，把上一轮的盆地当成本轮读数。）
2. **窗口只用跨窗 100K 的中位数**，不看窗内斜率（B1 那轮实测窗内斜率 R²=0.01-0.05，是噪声）。
3. **合格 episode 必须同集满足 K1'-K4'**（沿用 K1 传统，≥2 集），并且**同时把"这些集活满 600 帧没有"
   单列报告**——不改判据，只加证据强度（摔倒瞬间 20 帧窗会把 ds 抬高，那是本项目抓过两次的假阳性）。
4. ⑫ 自己的三个新仪表单独成节（它们回答的是"这批改价格到底有没有上场"，不是"学没学会"）：
   `walk_gate` 是否落在斜坡中段、`tiny_step_frame_frac` 是否离开 1.0、`land_per_s` 是否下降。

`--ab` 跑 §50.2 钉死的模板归因：同一份权重 k=1 vs k=0 各量若干集（只在终判时用，会吃约 1 GB 内存）。
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = "/home/guo/workspace/humanoid_walk"
DT = 0.015
STEPS_PER_ROW = 3206.0          # 只在 TB 不可用时的回退值（⑨ 年代的标定；⑫ 实测 4 集/eval、16000 步/eval）
N_EVAL_EP = 5                    # 源码 EvalCallback(n_eval_episodes=5)，用来交叉校验实测比值
MARKERS = ("land_per_s", "tiny_step_frame_frac", "dense_ds_factor")

# §46 K1'-K6'（不变）+ §50.2 新规则
BARS = dict(k1_lo=1.5, k1_hi=3.0, k2=0.20, k3=0.10, k4=0.20, k5_len=800, k5_ret=1525.0,
            early_len=300, early_ds=0.12,      # §50.2 的 600K 早停
            l3_land=4.5)                        # ⑫ 新仪表：落地频率该从实测 6.68/s 往下走
REF = {"⑫起点(⑪权重实测)": dict(land=6.68, spacing=0.009, ds=0.05, flight=0.167, v=1.0, gate=0.77,
                             tax_frac=0.4, per_frame=0.20),
       "⑨最优碎步": dict(land=12.07, spacing=0.073, ds=0.090, flight=0.270, v=0.49, gate=1.00,
                     tax_frac=np.nan, per_frame=1.69),
       "人体p50": dict(land=2.13, spacing=0.393, ds=0.270, flight=0.000, v=0.66, gate=1.00,
                   tax_frac=0.0, per_frame=3.82)}


def sp_min_of(r):
    """该集里较弱一侧的 spacing p50；两腿都没读数 ⇒ NaN（不能当 0，那是本项目抓过的第二种假阳性）。"""
    vals = [x for x in (num(r, "ev_spacing_L_p50"), num(r, "ev_spacing_R_p50")) if np.isfinite(x)]
    return min(vals) if vals else np.nan


def num(r, k, default=np.nan):
    v = r.get(k, default)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def detect_start(rows):
    """第一个带 ⑫ 标记键的行 = 本轮第一行；返回 (index, 说明)。"""
    for i, r in enumerate(rows):
        if all(k in r for k in MARKERS):
            rest = [j for j in range(i, len(rows)) if not all(k in rows[j] for k in MARKERS)]
            note = f"第 {i} 行起" + (f"（注意：后面有 {len(rest)} 行缺标记键 ⇒ 可能混入别轮数据）"
                                     if rest else "，之后全部带标记")
            return i, note
    return 0, "没找到任何带 ⑫ 标记键的行 ⇒ 要么本轮还没产出 eval，要么这些键没写进去（这就是 FAIL）"


def load(path, since, esteps=None):
    """行号→env 步数：优先用 TB `eval/mean_reward` 的真实 step 序列（本轮实测 eval 间隔 16K，
    每次 5 集 ⇒ 旧硬编码 3206 步/行只适用于 ⑨；硬编码会随 eval 配置漂移，⑪ 就已经差 9%）。"""
    out = []
    rows_all = [(i, r) for i, r in enumerate(rows_iter(path)) if i >= since]
    n = len(rows_all)
    per = max(1, round(n / len(esteps))) if (esteps and n) else 0
    if per:
        note_per = ""
        if per != N_EVAL_EP:
            note_per = (f"  ⚠ 源码里 n_eval_episodes={N_EVAL_EP}，实测比值 {n/max(len(esteps),1):.2f} "
                        f"⇒ 有 eval 行缺失或被 ep_len 过滤掉 ⇒ 用实测值，步数会略偏早")
        print(f"  步数标定：{n} 行 / {len(esteps)} 次 eval ⇒ **{per} 集/eval**{note_per}，"
              f"eval 间隔 {(esteps[1]-esteps[0]) if len(esteps)>1 else 0} env 步 "
              f"⇒ {((esteps[1]-esteps[0])/per) if (per and len(esteps)>1) else 0:.0f} 步/行"
              f"（硬编码回退值 {STEPS_PER_ROW:.0f}）")
    for j, (i, r) in enumerate(rows_all):
        ep = num(r, "ep_len")
        if not np.isfinite(ep) or ep <= 0:
            continue
        step = int(esteps[min(j // per, len(esteps) - 1)]) if per else int(j * STEPS_PER_ROW)
        # 最后一次 eval 之后写进来的行还没被 TB 记录 ⇒ 用"步数=最后一次 eval 步"保守处理（窗口只会偏早，不会偏晚）
        out.append(dict(row=i, step=step, ep_len=ep,
                        ret=num(r, "ret"), v=num(r, "speed"),
                        land_s=num(r, "land_per_s"),                     # total_land_count 口径
                        ev_land_s=((num(r, "ev_spacing_L_n", 0) + num(r, "ev_spacing_R_n", 0)) / (ep * DT)),
                        spL=num(r, "ev_spacing_L_p50"), spR=num(r, "ev_spacing_R_p50"),
                        sp_min=sp_min_of(r),
                        ds=num(r, "double_support_frac"), flight=num(r, "flight_frac"),
                        single=num(r, "single_frac"),
                        rom_L=num(r, "ev_rom_L_mean"), rom_R=num(r, "ev_rom_R_mean"),
                        lift=max(num(r, "ev_lift_L_mean", 0), num(r, "ev_lift_R_mean", 0)),
                        slip=max(num(r, "ev_slip_L_mean", 0), num(r, "ev_slip_R_mean", 0)),
                        walk_gate=num(r, "walk_gate"), flat_gate=num(r, "flat_gate"),
                        ddf=num(r, "dense_ds_factor"), tax_frac=num(r, "tiny_step_frame_frac")))
    return out


def rows_iter(path):
    with open(path) as f:
        for l in f:
            l = l.strip()
            if l:
                yield json.loads(l)


def med(sub, k):
    vals = [r[k] for r in sub if np.isfinite(r.get(k, np.nan))]
    return float(np.median(vals)) if vals else np.nan


def report(rows, upto, label):
    sub = [r for r in rows if r["step"] <= upto]
    B = BARS
    print(f"\n### {label}（≤{upto/1e6:.2f}M env 步，{len(sub)} 集）")
    if not sub:
        print("  这个区间还没有数据")
        return None
    m = {k: med(sub, k) for k in ("land_s", "ev_land_s", "spL", "spR", "ds", "flight", "single",
                                  "ep_len", "ret", "v", "lift", "slip", "rom_L", "rom_R",
                                  "walk_gate", "flat_gate", "ddf", "tax_frac")}
    spacing_min = med(sub, "sp_min")     # 注意：这是"每集较弱侧 p50"再取中位，不是中位的较弱侧
    qual = [r for r in sub
            if B["k1_lo"] <= r["land_s"] <= B["k1_hi"]
            and np.isfinite(r["sp_min"]) and r["sp_min"] >= B["k2"]
            and np.isfinite(r["flight"]) and r["flight"] <= B["k3"]
            and np.isfinite(r["ds"]) and r["ds"] >= B["k4"]]
    qual_long = [r for r in qual if r["ep_len"] >= 600]
    print(f"  中位：land/s={m['land_s']:5.2f}（落点事件口径 {m['ev_land_s']:5.2f}） "
          f"spacing_min={spacing_min:6.3f} ds={m['ds']:5.3f} flight={m['flight']:5.3f} "
          f"single={m['single']:5.3f}")
    print(f"        ep_len={m['ep_len']:6.0f} ret={m['ret']:7.1f} v={m['v']:5.2f} "
          f"lift={m['lift']:5.3f} slip={m['slip']:5.3f} rom={m['rom_L']:5.1f}/{m['rom_R']:5.1f}")
    tax_ppf = (m["tax_frac"] * 0.30 * max(0.0, min(1.0, (0.34 - spacing_min) / 0.34))
               if (np.isfinite(m["tax_frac"]) and np.isfinite(spacing_min)) else np.nan)
    print(f"  ⑫ 仪表：walk_gate={m['walk_gate']:.3f}（斜坡 0.20-1.00；⑪ 权重在 ⑫ 价下实测 0.77） "
          f"flat_gate={m['flat_gate']:.3f} dense_ds_factor={m['ddf']:.3f} "
          f"tiny_step_frame_frac={m['tax_frac']:.3f}（碎步盆≈1，走路≈0）"
          f" ⇒ 反推税幅值≈{tax_ppf:.3f} 分/帧")
    if m["ep_len"] < 120:
        print(f"    ⚠ 税的 duty 现在**必然**偏低：活动门要求每集落地>5，而 ep_len 中位只有 "
              f"{m['ep_len']:.0f} 帧（≈{m['ep_len']*m['land_s']*DT:.1f} 次落地）⇒ 这不构成'税失效'的证据")
    v = {}
    v["K1' 落地/s∈[1.5,3.0]"] = (B["k1_lo"] <= m["land_s"] <= B["k1_hi"], m["land_s"])
    v["K2' min spacing p50≥0.20"] = (np.isfinite(spacing_min) and spacing_min >= B["k2"], spacing_min)
    v["K3' flight≤0.10"] = (m["flight"] <= B["k3"], m["flight"])
    v["K4' ds≥0.20"] = (m["ds"] >= B["k4"], m["ds"])
    v["K5' ep_len≥800"] = (m["ep_len"] >= B["k5_len"], m["ep_len"])
    v["K5' ret≥1525"] = (m["ret"] >= B["k5_ret"], m["ret"])
    v["合格episode≥2（同集 K1'-K4'）"] = (len(qual) >= 2, float(len(qual)))
    for k, (ok, val) in v.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}  实测={val:.3f}" if isinstance(val, float)
              else f"  {'PASS' if ok else 'FAIL'}  {k}  实测={val}")
    print(f"  证据强度（不改判据）：合格集 {len(qual)} 个，其中活满 600 帧的 **{len(qual_long)}** 个 "
          f"⇒ 若为 0，合格读数可能来自摔倒时的 20 帧窗")
    covered = max(r["step"] for r in sub)
    if upto >= 0.6e6 and covered >= 0.6e6:
        hit = m["ep_len"] < B["early_len"] and m["ds"] < B["early_ds"]
        print(f"  {'⚠ 触发' if hit else '未触发'} 600K 早停（§50.2：ep_len<300 且 ds<0.12）"
              f" ⇒ 实测 ep_len={m['ep_len']:.0f} ds={m['ds']:.3f}"
              + ("  ⇒ 按预登记规则**停训**，不要再加机制" if hit else ""))
    elif upto >= 0.6e6:
        print(f"  早停规则暂不判：窗口要到 0.60M，但实际数据只到 {covered/1e6:.2f}M")
    if np.isfinite(m["land_s"]) and m["land_s"] > B["l3_land"]:
        print(f"  提示：land/s={m['land_s']:.2f} 仍 >{B['l3_land']} ⇒ 步频折扣（真实折扣 "
              f"{min(1.0, 2.5/m['land_s']):.2f}）还没把频率压下来")
    return v


def find_tb(root, tag="gait/tiny_step_frame_frac"):
    """只认含 ⑫ 新键的事件文件（⑪ 的没有 tiny_step_frame_frac）。
    先按**原始字节里有没有这个 tag 名**粗筛（TB 把 tag 写成明文），再只对候选做 Reload —
    否则整个 sac_humanoid_walk/ 树要扫几分钟（⑪ 那次就是这么卡的）。"""
    cands = []
    for f in glob.glob(os.path.join(root, "**", "events.out.tfevents*"), recursive=True):
        if os.path.getsize(f) < 1000:
            continue
        try:
            with open(f, "rb") as fh:
                if tag.encode() not in fh.read():
                    continue
        except OSError:
            continue
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            ea = EventAccumulator(f, size_guidance={"scalars": 0})   # 0=全留：eval step 序列要用来标定步数/行
            ea.Reload()
            if tag not in ea.Tags()["scalars"]:
                continue
        except Exception as e:
            print(f"  [tb] 跳过 {os.path.basename(f)}: {e}")
            continue
        cands.append((os.path.getmtime(f), f, ea))
    if not cands:
        return None, None
    cands.sort()
    return cands[-1][1], cands[-1][2]


def do_ab(model_path, n_ep=5):
    """§50.2 钉死的模板归因：同一份权重，k=1（部署控制器）vs k=0（纯网络）各跑 n_ep 集 det。"""
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    import gymnasium as gym
    from stable_baselines3 import SAC
    from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

    model = SAC.load(model_path)
    res = {}
    for k in (1.0, 0.0):
        raw = gym.make("Humanoid-v4", max_episode_steps=1500)
        patch_mirror_symmetry(raw.unwrapped.model)
        env = WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=True)
        env.set_osc_k(k)
        acc = []
        for ei in range(n_ep):
            obs, _ = env.reset(seed=3001 + ei)
            tot = steps = 0.0
            lands = tax_frames = 0
            g = []
            while True:
                act, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(np.asarray(act, dtype=np.float64))
                steps += 1
                tot += float(r)
                g.append(info["walk_gate"])
                lands = env.total_land_count
                tax_frames = env.tiny_step_frame_count
                if term or trunc:
                    break
            acc.append(dict(len=steps, ret=tot, land_s=lands / max(steps * DT, 1e-6),
                            tax_frac=tax_frames / max(steps, 1), gate=float(np.median(g)),
                            ds=float(env.double_support_frac),
                            sp=float(min(env.last_left_stride if env.last_left_stride is not None else -9,
                                         env.last_right_stride if env.last_right_stride is not None else -9))))
        res[k] = acc
        L = [a["len"] for a in acc]
        print(f"  k={k}: ep_len 中位={np.median(L):6.0f}（{min(L)}-{max(L)}） "
              f"ret/帧={np.median([a['ret'] / a['len'] for a in acc]):6.2f} "
              f"land/s={np.median([a['land_s'] for a in acc]):5.2f} "
              f"ds={np.median([a['ds'] for a in acc]):5.3f} "
              f"walk_gate={np.median([a['gate'] for a in acc]):5.3f} "
              f"tax_frac={np.median([a['tax_frac'] for a in acc]):5.3f}")
    d = np.median([a["ret"] / a["len"] for a in res[1.0]]) - np.median([a["ret"] / a["len"] for a in res[0.0]])
    print(f"  ⇒ 模板净贡献（k=1 减 k=0，分/帧）= {d:+.2f}"
          f"；ds 差 = {np.median([a['ds'] for a in res[1.0]]) - np.median([a['ds'] for a in res[0.0]]):+.3f}"
          f"；ep_len 差 = {np.median([a['len'] for a in res[1.0]]) - np.median([a['len'] for a in res[0.0]]):+.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=os.path.join(ROOT, "eval_logs", "eval_gait.jsonl"))
    ap.add_argument("--tb-root", default=os.path.join(ROOT, "sac_humanoid_walk"))
    ap.add_argument("--since", type=int, default=-1, help="-1 = 用 ⑫ 标记键自动定位本轮起点")
    ap.add_argument("--windows", default="0.2e6,0.6e6,1.2e6,2.0e6,4.0e6")
    ap.add_argument("--ab", default="", help="给一份 ⑫ 自己的权重路径就跑 k=1/k=0 归因（终判用）")
    ap.add_argument("--ab-ep", type=int, default=5)
    ap.add_argument("--no-tb", action="store_true", help="跳过 TB 自检（只想看 jsonl 时用）")
    a = ap.parse_args()

    all_rows = list(rows_iter(a.jsonl))
    path, ea = (None, None) if a.no_tb else find_tb(a.tb_root)
    esteps = None
    if ea is not None and "eval/mean_reward" in ea.Tags()["scalars"]:
        esteps = [x.step for x in ea.Scalars("eval/mean_reward")]
    if a.since < 0:
        since, note = detect_start(all_rows)
        auto_fail = not any(all(k in r for k in MARKERS) for r in all_rows)
        if auto_fail:
            print(f"jsonl={a.jsonl} 总行数={len(all_rows)}  ({note})")
            print("  ⇒ **按 0 集处理**：自动定位失败时不退回『分析全部行』，否则读的是别轮次的盆地。"
                  "确认是本轮的话，显式加 --since N。")
            rows = []
        else:
            print(f"jsonl={a.jsonl} 总行数={len(all_rows)}  本轮起点={since}  ({note})")
            rows = load(a.jsonl, since, esteps)
    else:
        since = a.since
        print(f"jsonl={a.jsonl} 总行数={len(all_rows)}  本轮起点={since}  (手工 --since)")
        rows = load(a.jsonl, since, esteps)
    if not rows:
        print("本轮 0 集：⑫ 还没产出 eval，或标记键没写进 jsonl（后者是实现问题，先查 `_log_episode`）。")
    else:
        print(f"最新一集 ≈ {rows[-1]['step']/1e6:.2f}M env 步（{len(rows)} 集）")
        for w in a.windows.split(","):
            report(rows, float(w), f"{float(w)/1e6:.1f}M 窗口")
    print("\n=== ⑫ 实现自检（TB）===")
    if a.no_tb:
        print("  按 --no-tb 跳过")
        return 0
    if ea is None:
        print("  没找到含 gait/tiny_step_frame_frac 的事件文件 ⇒ ⑫ 还没跑，或新键没写进 TB"
              "（**这本身就是 FAIL：碎步税可能在空转**）")
    else:
        for t in ("gait/tiny_step_frame_frac", "gait/walk_gate", "gait/dense_ds_factor",
                  "gait/osc_active_L", "gait/osc_k_now", "rollout/ep_len_mean",
                  "rollout/ep_rew_mean", "eval/mean_reward"):
            if t in ea.Tags()["scalars"]:
                sc = ea.Scalars(t)
                vv = np.array([x.value for x in sc])
                print(f"  {t}: 首={vv[0]:.3f} 中位={np.median(vv):.3f} "
                      f"末段={vv[-max(3, len(vv)//20):].mean():.3f} @step={sc[-1].step}"
                      f" 采样={len(vv)}")
            else:
                print(f"  {t}: **缺键**")
        print(f"  取用的事件文件：{path}")
    print("\n=== 参照（实测，非判据）===")
    for k, d in REF.items():
        print(f"  {k:22s} land/s={d['land']:6.2f} spacing={d['spacing']:6.3f} ds={d['ds']:5.3f} "
              f"flight={d['flight']:5.3f} gate={d['gate']:5.2f} tax_frac={d['tax_frac']:5.2f} "
              f"分/帧={d['per_frame']:5.2f}")
    if a.ab:
        print("\n=== §50.2 模板归因 A/B（k=1 部署 / k=0 纯网络，同一份权重）===")
        do_ab(a.ab, n_ep=a.ab_ep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
