"""⑮ 定价候选的**地形表**（TB 实测标定版）。

不预测训练结果（这类预测在本项目已被证伪四次）。只回答两件可核查的事：
  A. 在**当前实测工作点**上，"前进相关"的钱与"原地也能拿到"的钱各是多少（TB 的 *_gated 通道直接读，不建模）
  B. 把 ⑮ 改价的三条通道按同一批**实测观测量**重算一遍，比值变成多少

关键更正（本版为什么存在）：
  `forward_gate = clip(min(last_left_stride, last_right_stride)/0.15)`（源码 :1372）——用的是
  **两条腿里落后那条**的前移量，不是落点间距；`stride_bonus` 也乘同一个 min。所以一条腿当柱子
  会把两条前进通道一起清零。上一版拿 lo=0.049 去算，把 ⑮ 的收益估高了。
  本脚本的 st_lag 直接来自 jsonl 的 ev_step_adv_{L,R}_mean（每脚起飞→落地前移，与 last_*_stride 同物）。

第二件要查的事：v=0.3 处有**三道二值门同时开关**（height :1391 / survival :1394 / 碎步税 :1774）。
  脚本末尾做"跨崖扫描"：从实测 v 往下挪 0.1 越过 0.3，净分是涨还是跌。涨 ⇒ ⑮ 草案自己造了个
  "停下更划算"的坑（税退钱 > 高度门掉钱），必须先把它抹平再上场。
"""
import ast
import collections
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = "/home/guo/workspace/humanoid_walk"
SRC = os.path.join(ROOT, "train_humanoid_sac.py")
JSONL = os.path.join(ROOT, "eval_logs", "eval_gait.jsonl")
DT = 0.015
TAIL = 40          # jsonl 取末 N 集（⑭ 之后）
TB_POINTS = 3000   # TB 每个 tag 取末 N 个采样（record_mean = 每步随机策略的均值）


def consts():
    txt = open(SRC, encoding="utf-8").read()
    tree = ast.parse(txt)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WalkRewardWrapper")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    C = {}
    for node in ast.walk(init):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                v = node.value
                try:
                    C[t.attr] = -float(v.operand.value) if (isinstance(v, ast.UnaryOp)
                                                            and isinstance(v.operand, ast.Constant)) else float(v.value)
                except Exception:
                    pass
    for nm in ("STRIDE_FLOOR", "STRIDE_TOP", "STRIDE_WEIGHT", "STRIDE_CAD_REF", "NEGATIVE_STRIDE_WEIGHT"):
        m = re.search(nm + r"\s*=\s*([0-9.]+)", txt)
        if m:
            C[nm] = float(m.group(1))
    return C


C = consts()


# ---------- TB：流式读，只留需要的 tag 的末段，内存有界 ----------
def tb_find():
    best = None
    for f in glob.glob(os.path.join(ROOT, "sac_humanoid_walk", "**", "events.out.tfevents*"), recursive=True):
        if os.path.getsize(f) < 5000:
            continue
        with open(f, "rb") as fh:
            head = fh.read(4_000_000)
        if b"gait/swap_gate" in head and b"gait/forward_gauss_gated" in head:
            if best is None or os.path.getmtime(f) > os.path.getmtime(best):
                best = f
    return best


def tb_read(path, tags):
    from tensorboard.backend.event_processing import event_file_loader
    out = collections.defaultdict(lambda: collections.deque(maxlen=TB_POINTS))
    last_step = 0
    for ev in event_file_loader.EventFileLoader(path).Load():
        if not ev.summary.value:
            continue
        last_step = ev.step
        for v in ev.summary.value:
            if v.tag in tags:
                val = None
                if v.tensor.float_val:
                    val = float(v.tensor.float_val[0])
                elif v.tensor.double_val:
                    val = float(v.tensor.double_val[0])
                elif v.HasField("value"):
                    val = float(v.value)
                if val is not None:
                    out[v.tag].append(val)
    return {k: np.array(v) for k, v in out.items()}, last_step


TAGS = ["gait/forward_gauss_gated", "gait/stride_bonus_gated", "gait/position_swap_gated",
        "gait/phase_reward_gated", "gait/reach_reward_gated", "gait/height_reward_gated",
        "gait/survival_bonus_gated", "gait/swing_min_reward_gated", "gait/swing_forward_reward",
        "gait/contact_phase_reward", "gait/net_balance", "gait/walk_gate", "gait/flat_gate",
        "gait/dense_ds_factor", "gait/forward_gate", "gait/swap_gate", "gait/swap_rate",
        "gait/tiny_step_frame_frac", "gait/speed", "gait/short_stride_penalty",
        "gait/last_left_stride", "gait/last_right_stride",
        "gait/forward_gauss", "gait/stride_bonus", "gait/position_swap_reward",
        "gait/phase_reward", "gait/reach_reward", "gait/stride_bonus_rate", "gait/ds_landing_rate",
        "gait/negative_stride_penalty", "gait/alternate_paid", "gait/alternate_penalty",
        "gait/fall_penalty", "gait/ctrl_cost"]


# ---------- jsonl：实测观测量 ----------
def obs():
    rows = [json.loads(l) for l in open(JSONL)]
    rows = [r for r in rows if "swap_rate" in r and (r.get("ep_len") or 0) > 200][-TAIL:]
    g = lambda k: np.nanmedian([float(r.get(k, np.nan)) for r in rows])
    advL, advR = g("ev_step_adv_L_mean"), g("ev_step_adv_R_mean")
    sp = min(g("ev_spacing_L_p50"), g("ev_spacing_R_p50"))
    return dict(n=len(rows), v_end=g("speed"), st_lag=min(advL, advR), st_mean=0.5 * (advL + advR),
                advL=advL, advR=advR, lo=sp, ds=g("double_support_frac"), fl=g("flight_frac"),
                swap=g("swap_rate"), land_s=g("land_per_s"), ep_len=g("ep_len"),
                negfrac=0.5 * (g("ev_spacing_L_negfrac") + g("ev_spacing_R_negfrac")))


def wg_of(P, ds, fl):
    return P["wg_f"] + (1 - P["wg_f"]) * np.clip(((ds - fl) - P["wg_lo"]) / (P["wg_ref"] - P["wg_lo"]), 0, 1)


def main():
    O = obs()
    tf = tb_find()
    if tf is None:
        print("找不到含 gait/swap_gate 的事件文件 ⇒ ⑭ 的 TB 仪表没写进去（这本身是 FAIL）")
        return 1
    M, last_step = tb_read(tf, TAGS)
    print(f"TB={os.path.relpath(tf, ROOT)}  末步={last_step/1e6:.2f}M env 步")
    print(f"jsonl 取末 {O['n']} 集(ep_len>200) ⇒ 观测量：末帧v={O['v_end']:.3f} "
          f"st_lag(落后腿前移)={O['st_lag']*1000:.1f}mm st_mean={O['st_mean']*1000:.1f}mm "
          f"(L {O['advL']*1000:.0f}/R {O['advR']*1000:.0f}) lo={O['lo']:.3f} ds={O['ds']:.3f} "
          f"fl={O['fl']:.3f} swap={O['swap']:.3f} 落地/s={O['land_s']:.2f} ep_len={O['ep_len']:.0f}")
    print(f"人体参照：每脚前移 448mm ⇒ 落后腿只有人体的 {O['st_lag']/0.448*100:.0f}%")
    # ⚠ jsonl 的 speed 是**末帧**速度（已知陷阱）。每帧均速只能取 TB。
    O["v"] = float(np.mean(M["gait/speed"])) if "gait/speed" in M else O["v_end"]
    sl = float(np.mean(M["gait/last_left_stride"])) if "gait/last_left_stride" in M else np.nan
    sr = float(np.mean(M["gait/last_right_stride"])) if "gait/last_right_stride" in M else np.nan
    print(f"每帧均速(TB) v={O['v']:.3f} ｜碎步税/高度门共用的 0.3 门槛在它{'下方' if O['v']<0.3 else '上方'} "
          f"{abs(O['v']-0.3):.3f}")
    print(f"forward_gate 的两个被门量(TB 每帧均值)：last_left_stride={sl*100:.1f}cm last_right_stride={sr*100:.1f}cm")

    FG_REF = C["forward_gate_ref"]
    fg_m = float(np.mean(M["gait/forward_gate"]))
    wgv = float(np.mean(M["gait/walk_gate"]))
    print("\n=== A. 实测账本（TB *_gated，分/帧，随机策略 rollout）===")
    S = {}
    ADV = ["gait/forward_gauss_gated", "gait/stride_bonus_gated", "gait/position_swap_gated"]
    IPL = ["gait/phase_reward_gated", "gait/reach_reward_gated", "gait/height_reward_gated",
           "gait/survival_bonus_gated", "gait/swing_min_reward_gated", "gait/swing_forward_reward",
           "gait/contact_phase_reward", "gait/net_balance"]
    PEN = ["gait/short_stride_penalty", "gait/negative_stride_penalty", "gait/alternate_paid",
           "gait/ctrl_cost"]   # fall_penalty 无 tag：末帧 −50/集长，不在每帧账里
    for nm, grp in (("前进相关", ADV), ("原地可得", IPL), ("扣项", PEN)):
        s = 0.0
        for t in grp:
            v = float(np.mean(M[t])) if t in M else np.nan
            if np.isfinite(v):
                s += v
            print(f"  {t.replace('gait/',''):26s} {v:+8.4f}" if np.isfinite(v) else f"  {t.replace('gait/',''):26s}   (无此 tag)")
        S[nm] = s
        print(f"  —— {nm} 合计 {s:+.4f} 分/帧")
    print(f"\n  比值 原地:前进 = {S['原地可得']/max(S['前进相关'],1e-9):.0f}:1")
    print(f"  门控实况 walk_gate={np.mean(M['gait/walk_gate']):.3f} flat_gate={np.mean(M['gait/flat_gate']):.3f} "
          f"dense_ds={np.mean(M['gait/dense_ds_factor']):.3f} forward_gate={np.mean(M['gait/forward_gate']):.3f} "
          f"swap_gate={np.mean(M['gait/swap_gate']):.3f}")

    print("\n=== A′. 门梯：每条前进通道在哪一级被乘掉（TB 每帧均值）===")
    def gm(t):
        return float(np.mean(M[t])) if t in M else float("nan")
    fg_raw, fg_g = gm("gait/forward_gauss"), gm("gait/forward_gauss_gated")
    sb_raw, sb_g = gm("gait/stride_bonus"), gm("gait/stride_bonus_gated")
    ps_raw, ps_g = gm("gait/position_swap_reward"), gm("gait/position_swap_gated")
    print(f"  forward      未乘 walk_gate={fg_raw:+.4f}  →  实付={fg_g:+.4f}   ×{fg_g/max(fg_raw,1e-9):.3f}"
          f"   （这一级乘的就是 walk_gate={wgv:.3f}）")
    print(f"  stride_bonus 未乘 walk_gate={sb_raw:+.4f}  →  实付={sb_g:+.4f}   ×{sb_g/max(sb_raw,1e-9):.3f}")
    print(f"  pos_swap     未乘 walk_gate={ps_raw:+.4f}  →  实付={ps_g:+.4f}   ×{ps_g/max(ps_raw,1e-9):.3f}")
    print(f"  ⇒ 三条前进通道都带 forward_gate={fg_m:.3f}（挂在 min(左,右)落地间距上，两腿各有 "
          f"{O['negfrac']*100:.0f}% 的落地是负的），再乘 walk_gate={wgv:.3f}"
          f" ⇒ 合成折扣 = {fg_m*wgv:.4f}")

    # ---------- B. ⑮ 候选在同一批实测观测量上的反事实 ----------
    # 门取值一律用 TB 的**每帧实测均值**（随机策略 rollout），与 A 段账本同口径；
    # jsonl 是 det 集中位数，两套数字不混用（⑫ 之后的硬规矩）。
    FWD_W = 0.4
    cad_disc = min(1.0, C["STRIDE_CAD_REF"] / max(O["land_s"], 1e-9))
    per_land_to_frame = O["land_s"] * DT   # 落地事件分/落地 → 分/帧
    now_fwd = FWD_W * np.exp(-((O["v"] - 0.95) ** 2) / (2 * 0.28 ** 2)) * fg_m * wgv
    now_stride = np.clip((st_lag_gate - C["STRIDE_FLOOR"]) / (C["STRIDE_TOP"] - C["STRIDE_FLOOR"]), 0, 1) \
        * C["STRIDE_WEIGHT"] * cad_disc * wgv * per_land_to_frame
    print("\n=== B. ⑮ 候选：把三条通道按实测观测量重算（其余通道不动）===")
    print(f"  门看到的落后腿步幅 = forward_gate({fg_m:.3f})×0.15 = {st_lag_gate*100:.1f}cm"
          f"   ｜jsonl 每脚前移 min={O['st_lag']*100:.1f}cm mean={O['st_mean']*100:.1f}cm"
          f"（两者不一致时以门自己看到的为准，因为它才是被乘的量）")
    print(f"  自检 模型 forward_gauss_gated={now_fwd:+.4f} vs TB={float(np.mean(M['gait/forward_gauss_gated'])):+.4f}"
          f"；模型 stride_bonus_gated={now_stride:+.4f} vs TB={float(np.mean(M['gait/stride_bonus_gated'])):+.4f}"
          f" ⇒ 账本对得上才继续")
    ramp_v = np.clip(O["v"] / 0.95, 0, 1)
    variants = {
        "⑮a 草案：fwd_ref→0.05(min腿) 不乘 walk_gate  gauss→斜坡":
            (FWD_W * ramp_v * np.clip(st_lag_gate / 0.05, 0, 1), C["STRIDE_FLOOR"]),
        "⑮b 门槛改看两腿平均(柱子腿不再清零) ref 仍 0.15":
            (FWD_W * ramp_v * np.clip(O["st_mean"] / FG_REF, 0, 1), C["STRIDE_FLOOR"]),
        "⑮c 看平均 + ref 0.05 + stride floor 0.06":
            (FWD_W * ramp_v * np.clip(O["st_mean"] / 0.05, 0, 1), 0.06),
        "⑮d fwd 只看身体速度(前移的门整个交给税与负步幅)":
            (FWD_W * ramp_v, C["STRIDE_FLOOR"]),
    }
    for nm, (val, sfl) in variants.items():
        sb = np.clip((st_lag_gate - sfl) / (C["STRIDE_TOP"] - sfl), 0, 1) \
            * C["STRIDE_WEIGHT"] * cad_disc * wgv * per_land_to_frame
        tot = val + sb + float(np.mean(M["gait/position_swap_gated"]))
        print(f"  {nm:58s} fwd={val:+.4f} stride={sb:+.4f} ⇒ 前进合计={tot:+.4f}  "
              f"原地:前进={S['原地可得']/max(tot,1e-9):5.0f}:1")
    print(f"  现值参照：fwd={float(np.mean(M['gait/forward_gauss_gated'])):+.4f} "
          f"stride={float(np.mean(M['gait/stride_bonus_gated'])):+.4f} ⇒ 前进合计={S['前进相关']:+.4f}")

    # ---------- C. v=0.3 三道二值门的跨崖扫描 ----------
    v0 = O["v"]
    lo_v, hi_v = max(v0 - 0.10, 0.02), v0 + 0.10
    hs = float(np.mean(M["gait/height_reward_gated"])) + float(np.mean(M["gait/survival_bonus_gated"]))
    if "gait/short_stride_penalty" in M:
        tax, tax_src = float(np.mean(M["gait/short_stride_penalty"])), "TB 实测"
    else:
        # 该 tag 从未加进 TB（欠账，等这轮停了再补）。用 计费帧占比 × 在该帧的税率 估算：
        frac = float(np.mean(M["gait/tiny_step_frame_frac"]))
        rate = C["tiny_step_weight"] * np.clip((C["tiny_step_ref"] - O["lo"]) / C["tiny_step_ref"], 0, C["tiny_step_cap"])
        tax, tax_src = -frac * rate, f"估算=计费帧占比({frac:.3f})×税率({rate:.3f})，非实测"
    print(f"\n=== C. 跨崖扫描：0.3 是 height(:1391)/survival(:1394)/碎步税(:1774) 三门**共用**的二值阈值")
    print(f"    当前每帧均速 v={v0:.3f}（距崖 {abs(v0-0.3):.3f}）；height+survival 实付 {hs:+.4f}/帧，"
          f"碎步税实付 {tax:+.4f}/帧（{tax_src}）")
    print("    net(v) = 前进 + 高度/生存 + 碎步税 三条只跟 v 有关的通道之和（分/帧，其余通道与 v 无关）\n")

    def net(v, mode, height_ref, tax_mode):
        if mode == "NOW":
            f = FWD_W * np.exp(-((v - 0.95) ** 2) / (2 * 0.28 ** 2)) * fg_m * wgv
        else:
            f = FWD_W * np.clip(v / 0.95, 0, 1) * np.clip(st_lag_gate / 0.05, 0, 1)
        h = hs * (1.0 if v > 0.3 else 0.0) if height_ref is None else hs * np.clip(v / height_ref, 0, 1)
        t = tax * (1.0 if v > 0.3 else 0.0) if tax_mode is None else tax * np.clip(v / tax_mode, 0, 1)
        return f + h + t

    combos = (("⑬⑭ 现行（三门全二值）", "NOW", None, None),
              ("⑮a 草案（高度门线性到 0.8，税门**仍是 0.3 二值**）", "NX", 0.8, None),
              ("⑮a′ 修正（高度门与税门同一条 0.8 斜坡）", "NX", 0.8, 0.8))
    for nm, mode, href, tref in combos:
        a, b, c = net(lo_v, mode, href, tref), net(v0, mode, href, tref), net(hi_v, mode, href, tref)
        # 真正的危险是 v=0.3 处的**断点**（±0.1 可能根本没跨过去），所以骑跨 0.3 各取一次
        up, dn = net(0.301, mode, href, tref), net(0.299, mode, href, tref)
        jump = dn - up
        ok = "PASS" if jump < 0 else "**FAIL：跌破 0.3 反而更赚**"
        print(f"  {nm}")
        print(f"    net({lo_v:.2f})={a:+.4f} net({v0:.2f})={b:+.4f} net({hi_v:.2f})={c:+.4f}"
              f"  ｜跨崖 net(0.301)={up:+.4f} → net(0.299)={dn:+.4f} 跳变={jump:+.4f} ⇒ {ok}")
    print(f"  另注：det 集末帧均速={O['v_end']:.3f} 已在 0.3 之下，而 TB 每帧均速={v0:.3f} 在之上 ⇒ "
          f"策略群体**正骑在这道二值门上开关**（⑫ 时代记录过：骑门开关会让地板失效、Q 塌）。")

    print("\n注：A 段全是实测；B/C 段是同一批实测观测量的反事实算术，不是对训练结果的预测。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
