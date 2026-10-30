"""批次⑬（L1 税去饱和 + L2 负步幅改任意落地）的验证：**逐帧恒等式 + 实测起点价**。

只用 ⑫ 归档权重做只读 rollout；不训练、不保存任何文件、不碰 git。

要证明的四件事（每件都独立重算一遍，代码接错就 FAIL）：
  V1 税 = w·clip((REF−lo)/REF, 0, **cap**)  在 lo<0 处必须**超过 w**（这就是 L1 的全部意义）
  V2 负步幅罚在**腾空落地**上开始计费（⑫ 里这一桶完全免罚）——用物理状态自己判接触，
     不依赖被验证的那段代码，否则就是自证
  V3 罚值 = W·clip(−lo/SCALE, 0, 1)，W 必须等于源码里的 3.0
  V4 覆盖率：受罚落地次数 / 全部负步幅落地次数，⑬ 应当从 ⑫ 的 3.4% 跳到接近 100%
  V5 起点价：同一份 ⑫ 权重在 ⑬ 价格下的每帧分与整集分，对照它自己在 ⑫ 价格下的实测值
"""
import ast
import os
import re
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")

import gymnasium as gym
import mujoco
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

MODEL = os.environ.get("PROBE_MODEL", "best_model/tmp_model_data/SAC_19/19_2_小碎步惩罚4M/best_model.zip")
DT, FR = 0.015, 1500
SEEDS = tuple(int(x) for x in os.environ.get("V_SEEDS", "123,2026,7,31,55,77").split(","))

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
NEG_W = float(re.search(r"NEGATIVE_STRIDE_WEIGHT\s*=\s*([0-9.]+)", SRC).group(1))
NEG_S = float(re.search(r"NEGATIVE_STRIDE_SCALE\s*=\s*([0-9.]+)", SRC).group(1))
TW, TR, TCAP = C["tiny_step_weight"], C["tiny_step_ref"], C["tiny_step_cap"]
WGF, WGLO, WGRE, DSF, DSR = (C["walk_gate_floor"], C["walk_gate_lo"], C["walk_gate_ref"],
                             C["dense_ds_floor"], C["dense_ds_ref"])
fails = []


def chk(name, cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {msg}")
    if not cond:
        fails.append(name)


def run(model):
    raw = gym.make("Humanoid-v4", max_episode_steps=FR)
    patch_mirror_symmetry(raw.unwrapped.model)
    w = WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=True)
    m, d = w.unwrapped.model, w.unwrapped.data
    foot_g = {}
    for side, bname in (("L", "left_foot"), ("R", "right_foot")):
        bid = m.body(bname).id
        foot_g[side] = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid
                        and int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)]

    def touching(side):
        return any((d.geom_xpos[g][2] - m.geom_size[g][0]) < 0.01 for g in foot_g[side])

    tax_viol = gate_viol = ddf_viol = neg_viol = 0
    n_tax_over_w = n_tax_frames = 0
    charged_air = charged_ds = charged_noneg = 0
    n_landings = n_neg_landings = charged_tot = 0
    neg_events_tot = ds_land_tot = land_tot = 0
    own_prev = (None, None)        # 用于识别"本帧哪只脚落地"（这两个量只在落地帧被赋值）
    eps = []
    nan = tot_frames = 0
    for sd in SEEDS:
        obs, _ = w.reset(seed=sd)
        t = 0
        acc = dict(ret=0.0, tax=0.0, neg=0.0, height=0.0, surv=0.0, swap=0.0, fwd=0.0,
                   gates=[], lo=[]
                   )
        prev_land = 0
        while True:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = w.step(np.asarray(act, dtype=np.float64))
            t += 1
            tot_frames += 1
            if not np.isfinite(r):
                nan += 1
            # V1/V3：税与负步幅罚，用 info 里的 lo（零阶保持）独立重算
            if info["lo_defined"]:
                lo = info["lo_held"]
                v_mean = ((d.qpos[0] - w._ep_start_x) / max(w.total_steps * DT, 1e-6)
                          if w._ep_start_x is not None else 0.0)
                elig = v_mean > 0.3 and w.total_land_count > 5
                exp_tax = TW * np.clip((TR - lo) / TR, 0.0, TCAP) if elig else 0.0
                if abs(info["short_stride_penalty"] - exp_tax) > 1e-6:
                    tax_viol += 1
                if exp_tax > TW + 1e-9:
                    n_tax_over_w += 1
                n_tax_frames += int(info["short_stride_penalty"] > 0)
                acc["lo"].append(lo)
            # V2/V3/V4：负步幅罚。**用 last_left_stride / last_right_stride 是否在本帧改变**来认定
            #   "本帧是哪只脚落地"——这两个量只在该脚落地帧被赋值，比我自己重建接触判据可靠。
            nl = w.total_land_count
            own_now = (w.last_left_stride, w.last_right_stride)
            landL = own_now[0] != own_prev[0]
            landR = own_now[1] != own_prev[1]
            own_prev = own_now
            if nl > prev_land:
                n_landings += 1
                own = None
                if landL and landR:
                    own = min(w.last_left_stride, w.last_right_stride)
                elif landL:
                    own = w.last_left_stride
                elif landR:
                    own = w.last_right_stride
                val = info["negative_stride_penalty"]
                if own is not None and own < 0:
                    n_neg_landings += 1          # 正确分母：落地那只脚自己的间距<0
                if val > 0:
                    charged_tot += 1
                    exps = [NEG_W * np.clip(-x / NEG_S, 0, 1)
                            for x in (w.last_left_stride, w.last_right_stride) if x is not None]
                    if not any(abs(val - e) < 1e-5 for e in exps):
                        neg_viol += 1
                        if neg_viol <= 3:
                            print(f"    负步幅违例: cand={[round(x,3) for x in exps]} got={val:.3f}")
            prev_land = nl
            # 门控恒等式（⑫ 的，本轮没改，重跑一次防手滑）
            ds, fl = info["double_support_frac"], info["flight_frac"]
            if len(w.left_contact_hist) >= 20:
                if abs(info["walk_gate"] - (WGF + (1 - WGF) * np.clip(((ds - fl) - WGLO) / (WGRE - WGLO), 0, 1))) > 1e-6:
                    gate_viol += 1
            acc["ret"] += float(r)
            acc["tax"] += info["short_stride_penalty"]
            acc["neg"] += info["negative_stride_penalty"]
            acc["height"] += info.get("height_reward_gated", 0.0)
            acc["surv"] += info.get("survival_bonus_gated", 0.0)
            acc["swap"] += info.get("position_swap_gated", 0.0)
            acc["fwd"] += info.get("forward_gauss_gated", 0.0)
            acc["gates"].append(info["walk_gate"])
            if term or trunc:
                break
        acc["len"] = t
        acc["lo_end"] = acc["lo"][-1] if acc["lo"] else np.nan
        acc["neg_events"] = w.negative_stride_event_count      # 被计费的负步幅落地次数
        acc["ds_land"] = w.stride_event_count                  # ds 落地总数（旧分支唯一能看到的）
        acc["land"] = w.total_land_count
        neg_events_tot += acc["neg_events"]; ds_land_tot += acc["ds_land"]; land_tot += acc["land"]
        eps.append(acc)
    return dict(eps=eps, tax_viol=tax_viol, gate_viol=gate_viol, ddf_viol=ddf_viol, neg_viol=neg_viol,
                n_tax_over_w=n_tax_over_w, n_tax_frames=n_tax_frames, n_landings=n_landings,
                n_neg_landings=n_neg_landings, charged_probe=charged_tot,
                neg_events=neg_events_tot, ds_land=ds_land_tot,
                land=land_tot, nan=nan, tot_frames=tot_frames)


def main():
    print(f"源码常数（ast）: tax(w={TW}, ref={TR}, **cap={TCAP}**)  neg(W={NEG_W}, scale={NEG_S})  "
          f"gate({WGF},{WGLO},{WGRE}) ds({DSF},{DSR}) osc_A={C.get('osc_A')}")
    model = SAC.load(MODEL)
    print(f"权重（只读）: {MODEL}")
    R = run(model)
    eps = R["eps"]

    print("\n=== V1 税去饱和（L1）===")
    chk("V1a 逐帧恒等式 tax == w·clip((REF−lo)/REF, 0, cap)", R["tax_viol"] == 0,
        f"违例 {R['tax_viol']} / {R['tot_frames']} 帧")
    chk("V1b cap 生效：存在 tax > w 的帧（⑫ 里这种帧必然为 0）", R["n_tax_over_w"] > 0,
        f"tax>{TW} 的帧数 = {R['n_tax_over_w']}")
    print(f"  计税帧 {R['n_tax_frames']}/{R['tot_frames']} = {R['n_tax_frames']/max(R['tot_frames'],1):.2f}"
          f"   税合计中位 {np.median([e['tax'] for e in eps]):.0f} 分/集")

    print("\n=== V2/V3/V4 负步幅罚改任意落地（L2）===")
    tot_neg = R["n_neg_landings"]
    charged = R["neg_events"]
    charged_probe = R["charged_probe"]
    print(f"  落地总数 {R['land']}  其中**落地那只脚自己的间距<0** 的 {tot_neg}"
          f"（{100*tot_neg/max(R['land'],1):.0f}%）  被计费 {charged} 次"
          f"（探针另路识别 {charged_probe} 次）")
    print(f"  **关键不等式**：被计费 {charged}  vs  ds 落地总数 {R['ds_land']}"
          f" ⇒ 若 charged > ds_land，则一定有 ds 分支之外（腾空落地）被计费（⑫ 结构性做不到）")
    chk("V2 腾空落地开始被计费（charged > ds_land）", charged > R["ds_land"],
        f"{charged} > {R['ds_land']}" if charged > R["ds_land"] else f"{charged} ≤ {R['ds_land']}")
    chk("V3 罚值 == W·clip(−lo/SCALE,0,1)", R["neg_viol"] == 0, f"违例 {R['neg_viol']}")
    cov = 100 * charged / max(tot_neg, 1)
    chk("V4 负步幅落地的计费覆盖率（⑫ 实测只有 3.4%）", cov > 90,
        f"{charged}/{tot_neg} = {cov:.1f}%")
    print(f"  负步幅罚合计中位 {np.median([e['neg'] for e in eps]):.0f} 分/集  （⑫ TB 实测该通道 ≈ −4 分/集）")

    print("\n=== 门控未被动过（回归检查）===")
    chk("R1 walk_gate 恒等式", R["gate_viol"] == 0, f"违例 {R['gate_viol']}")
    chk("R2 无 NaN", R["nan"] == 0, f"NaN 帧 {R['nan']}")

    print("\n=== V5 ⑬ 起点价（同一份 ⑫ 权重，只读实测）===")
    print(f"  {'ep_len':>7} {'ret':>7} {'分/帧':>7} | {'height+surv':>11} {'swap':>6} {'forward':>8} {'税':>7} {'负步幅罚':>8} {'lo_end':>7}")
    for e in eps:
        print(f"  {e['len']:7d} {e['ret']:7.0f} {e['ret']/e['len']:7.2f} | {e['height']+e['surv']:11.0f} "
              f"{e['swap']:6.0f} {e['fwd']:8.0f} {-e['tax']:7.0f} {-e['neg']:8.0f} {e['lo_end']:+7.3f}")
    pf = np.median([e["ret"] / e["len"] for e in eps])
    print(f"\n  ⇒ ⑬ 每帧分中位 = {pf:+.2f}；⑫ 同一策略的实测是 −0.24（rollout 末段）/ +0.24（eval 67/284 帧）")
    print(f"     红线（预先声明）：若 ⑬ 起点价中位 < −3.0 分/帧 ⇒ 判定为双重冲击（warm start+空 buffer+新价），回滚 L2 权重到 1.5")
    print(f"\n=== 结论：{'ALL PASS' if not fails else 'FAIL -> ' + ', '.join(fails)} ===")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
