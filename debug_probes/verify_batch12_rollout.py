"""批次⑫ 的 **rollout 级**验证（需要能加载 SAC，⑪ 停训后才有内存 —— 之前 RAM 只剩 1.4 GB）。

它要回答的不是"能不能学会"（那只有训练能回答），而是三件可以当场证明的事：

1. **接线恒等式**（每一步都检查，不是抽查）：info 里的量必须满足 ⑫ 的公式
     walk_gate        == floor + (1-floor)·clip(((ds−flight)−lo)/(ref−lo))
     dense_ds_factor  == ds_floor + (1−ds_floor)·clip(ds/ds_ref)
     short_stride_pen == w·clip((ref−min(左,右落点间距))/ref,0,1)   仅当 集均速>0.3 且 落地>5
     stride_bonus     == 斜坡(lo) × 35 × min(1, 2.5/total_land_count 口径的落地每秒)
                        （只在该值>0 的落地帧检查；0 帧跳过并如实报告样本数）
   —— 这一条是"验证必须能证明它自己的断言"：公式在探针里独立重算一遍，代码接错就 FAIL。
2. **⑫ 下 ⑪ 自己盆地的起点价**（实测，不是外推）：受影响通道的整集分、每帧率、门取值范围。
3. **无 NaN**、站立/低速不交税（活动门真的生效）。

不训练、不保存任何文件。
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

DT = 0.015
FRAMES = 1500
MODEL = os.environ.get("PROBE_MODEL",
                        "best_model/tmp_model_data/SAC_19/19_1/best_model.zip")

# 从源码 ast 抓 ⑫ 常数（和 verify_batch12.py 同一把尺）
sys.path.insert(0, os.path.join(os.getcwd(), "debug_probes"))
import verify_batch12 as V  # noqa: E402

C = V.read_constants()
FL, TP, WT = V.read_stride_ramp()
WG_LO, WG_REF, WG_F = C["walk_gate_lo"], C["walk_gate_ref"], C["walk_gate_floor"]
DS_REF, DS_F = C["dense_ds_ref"], C["dense_ds_floor"]
TW, TR = C["tiny_step_weight"], C["tiny_step_ref"]
CAD = C["STRIDE_CAD_REF"]

fails = []


def chk(name, cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {msg}")
    if not cond:
        fails.append(name)


def build():
    raw = gym.make("Humanoid-v4", max_episode_steps=FRAMES)
    patch_mirror_symmetry(raw.unwrapped.model)
    return WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=True)


def expected_gate(ds, fl):
    return WG_F + (1.0 - WG_F) * np.clip(((ds - fl) - WG_LO) / (WG_REF - WG_LO), 0.0, 1.0)


def expected_ddf(ds):
    return DS_F + (1.0 - DS_F) * np.clip(ds / DS_REF, 0.0, 1.0)


def run(label, policy, n_ep, seeds):
    """policy=None ⇒ 零动作。返回每集统计 + 逐帧恒等式的违例计数。"""
    env = build()
    viol = {"gate": 0, "ddf": 0, "tax": 0, "disc": 0, "warmup": 0, "ddf_now": 0}
    prev_ds = 0.0
    prev_sb_count = 0
    n_settle = 0
    n_tax_frames = n_disc_checked = 0
    tot_frames = nan = 0
    eps = []
    for ei in range(n_ep):
        obs, _ = env.reset(seed=seeds[ei])
        acc = dict(ret=0.0, stride=0.0, tax=0.0, height=0.0, surv=0.0, swap=0.0,
                   fwd=0.0, gates=[], ddf=[], alt_pen=0.0, slip=0.0)
        steps = 0
        while True:
            if policy is None:
                act = np.zeros(env.unwrapped.model.nu)
            else:
                act, _ = policy.predict(obs, deterministic=True)
                act = np.asarray(act, dtype=np.float64)
            obs, r, term, trunc, info = env.step(act)
            steps += 1
            tot_frames += 1
            if not np.isfinite(r):
                nan += 1
            ds, fl = info["double_support_frac"], info["flight_frac"]
            # 门与因子只在接触窗满 20 帧时更新（暖机帧沿用上一集的值），且 dense 因子用的是
            # **上一帧**的 ds（代码注释里明写的滞后）⇒ 这两件事单独计成 warmup/lag，不混进违例。
            warmed = len(env.left_contact_hist) >= 20
            if not warmed:
                viol["warmup"] += 1
            else:
                if abs(info["walk_gate"] - expected_gate(ds, fl)) > 1e-6:
                    viol["gate"] += 1
                    if viol["gate"] <= 3:
                        print(f"    gate 违例: ds={ds:.3f} fl={fl:.3f} got={info['walk_gate']:.3f} "
                              f"exp={expected_gate(ds, fl):.3f}")
                if abs(info["dense_ds_factor"] - expected_ddf(prev_ds)) > 1e-6:
                    viol["ddf"] += 1
                    if viol["ddf"] <= 3:
                        print(f"    ddf 违例: ds_prev={prev_ds:.3f} ds_now={ds:.3f} "
                              f"got={info['dense_ds_factor']:.3f} exp={expected_ddf(prev_ds):.3f}")
                viol["ddf_now"] += int(abs(info["dense_ds_factor"] - expected_ddf(ds)) > 1e-6)
            prev_ds = ds
            # 税的恒等式（用 info 导出的 lo_defined / 两腿落点间距重算）
            ll, lr = info["last_left_stride"], info["last_right_stride"]
            w = env          # WalkRewardWrapper 本体（total_land_count 等计数器在它身上）
            defined = bool(info["lo_defined"])
            if defined:
                lo_now = min(ll, lr) if (ll != 0.0 or lr != 0.0) else \
                    min(w.last_left_stride, w.last_right_stride)
                v_mean = ((w.unwrapped.data.qpos[0] - w._ep_start_x) / max(w.total_steps * DT, 1e-6)
                          if w._ep_start_x is not None else 0.0)
                elig = w.total_land_count > 5 and v_mean > 0.3
                exp_tax = TW * np.clip((TR - lo_now) / TR, 0.0, 1.0) if elig else 0.0
                if abs(info["short_stride_penalty"] - exp_tax) > 1e-6:
                    viol["tax"] += 1
                    if viol["tax"] <= 3:
                        print(f"    tax 违例样本: lo={lo_now:+.3f} v_mean={v_mean:.2f} "
                              f"lands={w.total_land_count} steps={w.total_steps} "
                              f"got={info['short_stride_penalty']:.4f} exp={exp_tax:.4f}")
            else:
                if info["short_stride_penalty"] != 0.0:
                    viol["tax"] += 1
                lo_now = None
            n_tax_frames += int(info["short_stride_penalty"] > 0)
            # 步频折扣：只在结算帧（last_stride_bonus>0）检查
            sb = info["last_stride_bonus"]
            settled = info["stride_bonus_event_count"] > prev_sb_count
            prev_sb_count = info["stride_bonus_event_count"]
            if sb > 0 and settled:
                cad_now = w.total_land_count / max(w.total_steps * DT, 1e-6)
                exp_sb = np.clip((lo_now - FL) / (TP - FL), 0, 1) * WT * min(1.0, CAD / max(cad_now, 1e-9))
                if abs(sb - exp_sb) > 1e-5:
                    viol["disc"] += 1
                    if viol["disc"] <= 3:
                        print(f"    折扣违例样本: lo={lo_now:+.3f} lands={w.total_land_count} "
                              f"steps={w.total_steps} got={sb:.3f} exp={exp_sb:.3f}")
                n_disc_checked += 1
                if viol["disc"] == 0:
                    n_settle += 1
            acc["ret"] += float(r)
            acc["stride"] += info.get("stride_bonus_gated", 0.0)
            acc["tax"] += info["short_stride_penalty"]
            acc["height"] += info.get("height_reward_gated", 0.0)
            acc["surv"] += info.get("survival_bonus_gated", 0.0)
            acc["swap"] += info.get("position_swap_gated", 0.0)
            acc["fwd"] += info.get("forward_gauss_gated", 0.0)
            acc["slip"] += info.get("slip_penalty", 0.0)
            acc["gates"].append(info["walk_gate"])
            acc["ddf"].append(info["dense_ds_factor"])
            if term or trunc:
                break
        acc["len"] = steps
        acc["cadence_total"] = env.total_land_count / max(steps * DT, 1e-6)
        acc["tiny_frac"] = env.tiny_step_frame_count / max(steps, 1)
        eps.append(acc)
    g = np.concatenate([e["gates"] for e in eps])
    print(f"\n--- {label}: {n_ep} 集 / {tot_frames} 帧, ep_len={min(e['len'] for e in eps)}"
          f"-{max(e['len'] for e in eps)}, NaN 帧={nan} ---")
    print(f"  walk_gate  min/p50/max = {g.min():.3f}/{np.median(g):.3f}/{g.max():.3f}"
          f"   （地板 {WG_F} / 满门判定 x≥{WG_REF}）")
    print(f"  恒等式违例: gate={viol['gate']} ddf={viol['ddf']} tax={viol['tax']} "
          f"stride折扣={viol['disc']}(检查 {n_disc_checked} 个结算帧) 计税帧={n_tax_frames}")
    print(f"  暖机帧(接触窗<20，门按设计不更新)={viol['warmup']}；"
          f"若把 ddf 按**当帧** ds 比而不是前一帧，会有 {viol['ddf_now']} 帧差一帧 ⇒ 代码文档里写明的滞后")
    print(f"  {'ep_len':>7} {'ret':>8} {'落地/s(total)':>13} {'tiny_frac':>9} | "
          f"{'height+surv':>11} {'swap':>7} {'stride':>7} {'forward':>7} {'碎步税':>7} {'slip':>7}")
    for e in eps:
        print(f"  {e['len']:7d} {e['ret']:8.0f} {e['cadence_total']:13.2f} {e['tiny_frac']:9.3f} | "
              f"{e['height'] + e['surv']:11.0f} {e['swap']:7.0f} {e['stride']:7.0f} "
              f"{e['fwd']:7.0f} {-e['tax']:7.0f} {-e['slip']:7.0f}")
    return eps, viol, nan, tot_frames, n_disc_checked


def main():
    print(f"⑫ 常数（ast 自源码）: gate({WG_F},{WG_LO},{WG_REF}) ds({DS_F},{DS_REF}) "
          f"tax({TW},{TR}) cad={CAD} stride_ramp({FL},{TP},{WT}) osc_A={C['osc_A']}")

    eps0, v0, nan0, tf0, n0 = run("零动作（模板开，必摔）", None, 4, [11, 12, 13, 14])
    chk("R1 零动作下无 NaN", nan0 == 0, f"{tf0} 帧")
    REAL = ("gate", "ddf", "tax", "disc")
    chk("R2 零动作恒等式全部成立", all(v0[k] == 0 for k in REAL), f"真违例{{{', '.join(f'{k}:{v0[k]}' for k in REAL)}}}  暖机/滞后单列 {v0['warmup']}/{v0['ddf_now']}")
    lo_max = max(e["tax"] / max(e["len"], 1) for e in eps0)
    print(f"  （零动作每帧平均税 = {lo_max:.3f}，摔在 ~40 帧 ⇒ 落地数<5 时活动门应压住）")

    if os.path.exists(MODEL):
        m = SAC.load(MODEL)
        # 加载后的权重不带训练步数 ⇒ 用同目录 best_extra_logs.jsonl 里的 timesteps 标一下，只是标签
        nt = os.environ.get("PROBE_STEPS", "?")
        print(f"\n加载 {MODEL}")
        eps1, v1, nan1, tf1, n1 = run(f"⑪ 权重({nt} 步)", m, 6, list(range(21, 27)))
        chk("R3 ⑪ rollout 无 NaN", nan1 == 0, f"{tf1} 帧")
        chk("R4 ⑪ 恒等式全部成立（gate/ddf/tax/stride 折扣都按 ⑫ 公式发放）",
            all(v1[k] == 0 for k in REAL), f"真违例{{{', '.join(f'{k}:{v1[k]}' for k in REAL)}}}"
            f"  结算帧折扣检查 {n1} 次 / 暖机 {v1['warmup']}")
        rates = [e["ret"] / e["len"] for e in eps1]
        print(f"\n  ⇒ ⑫ 价格下 ⑪ 盆地的实测每帧分 = {np.mean(rates):.2f}"
              f"（区间 {min(rates):.2f}-{max(rates):.2f}）；"
              f"实测落地/s(total_land_count 口径)={np.mean([e['cadence_total'] for e in eps1]):.2f}"
              f" ⇒ 真实折扣={min(1.0, CAD / np.mean([e['cadence_total'] for e in eps1])):.2f}"
              f"（jsonl 落点事件口径给的是 0.60，那个数是乐观值）")
    else:
        print(f"\n跳过权重档：{MODEL} 不存在")

    print(f"\n=== 结论：{'ALL PASS' if not fails else 'FAIL -> ' + ', '.join(fails)} ===")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
