"""批次⑯（T2：时钟窗口 → 触地即止）的验证。

两件事，都必须做：
  P1 **价格不变性的机械证明**：把当前文件里的 T2 驱动块换成 ⑬⑭ 备份里的 T1 块，
     得到"重构的 ⑮ 文件"；然后逐行 diff，要求**唯一差异就是那个块**。
     这样才敢说 ⑯ 相对 ⑮ 只改了驱动定律（⑮ 源文件当时没单独留副本，只能这样重建）。
  P2 **行为 A/B**：同一份权重（⑮ 的 best）、同一批 seed、两套驱动定律各跑 det rollout，
     看 D1 覆盖率 / D2 落地大腿前倾 / D3 reach_R / 存活。轨迹必然不同（这正是自变量），
     所以这里比的是分布，不是逐帧恒等。

顺带守 ⑪ 的两条结构不变量：同一帧不得两腿都在驱动、驱动占空比不能失控。
"""
import difflib
import importlib.util
import os
import re
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")

import gymnasium as gym
import mujoco
from stable_baselines3 import SAC

NEW_SRC = "train_humanoid_sac.py"
T1_SRC = "train_humanoid_sac_batch13_14_1_86M.py"      # ⑯ 没动过模板 ⇒ 它的 T1 块就是 ⑮ 用的那块
RECON = "/tmp/train_b15_recon.py"
MODEL = os.environ.get("PROBE_MODEL", "best_model/best_model.zip")   # ⑮ best
SEEDS = tuple(int(x) for x in os.environ.get("T2_SEEDS", "123,2026,7,31,55,77,91,101").split(","))
FR = 1500

t_new = open(NEW_SRC, encoding="utf-8").read()
t_t1 = open(T1_SRC, encoding="utf-8").read()

# ---- 切块：从 `phase = (self.total_steps % self.gait_period)` 到 `self.osc_active = (...)` 结束 ----
BLOCK = re.compile(r"[ ]{12}phase = \(self\.total_steps % self\.gait_period\)[^\n]*\n"
                   r"(?:.*\n)*?[ ]{12}self\.osc_active = \(act_flag\[0\], act_flag\[1\]\)\n")
m_new, m_t1 = BLOCK.search(t_new), BLOCK.search(t_t1)
assert m_new and m_t1, "切块失败：模板块边界没匹配上，先改正则再谈验证"
recon_txt = t_new[:m_new.start()] + m_t1.group(0) + t_new[m_new.end():]
open(RECON, "w", encoding="utf-8").write(recon_txt)

d = list(difflib.unified_diff(t_new.splitlines(), recon_txt.splitlines(), lineterm="", n=0))
changed = [x for x in d if x.startswith(("+", "-")) and not x.startswith(("+++", "---"))]
# 不靠关键字猜"哪些行属于驱动定律"——直接验证所有差异都落在被替换的那段字符区间内。
sm = difflib.SequenceMatcher(None, t_new, recon_txt, autojunk=False)
outside = []
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag == "equal":
        continue
    if not (m_new.start() <= i1 and i2 <= m_new.end()):
        outside.append((tag, i1, i2, t_new[max(0, i1 - 40):i2 + 40].replace("\n", "⏎")[:200]))
print(f"P1 重构 diff：改动 {len(changed)} 行，差异区间越界 {len(outside)} 处")
for o in outside:
    print("   越界:", o)
assert not outside, "重构越界 ⇒ 除了驱动定律还改了别的东西，⑯ 的归因不成立"
print("  ⇒ 全部差异都落在被替换的模板块内 ✓（⑯ 相对 ⑮ 只换了 T1→T2，价格一行没动）")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def run(W, seed, model, want_detail=True):
    raw = gym.make("Humanoid-v4", max_episode_steps=FR)
    W.patch_mirror_symmetry(raw.unwrapped.model)
    w = W.WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=True)
    m, dd = w.unwrapped.model, w.unwrapped.data
    hidx = {s: int(m.jnt_qposadr[m.joint(f"{s}_hip_y").id]) for s in ("left", "right")}
    gl = {s: [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(f"{s}_foot").id]
          for s in ("left", "right")}
    obs, _ = w.reset(seed=seed)
    both = drive_fr = frames = 0
    land_tot = land_in = 0
    land_ang = {"left": [], "right": []}
    prev = {"left": True, "right": True}
    while True:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = w.step(np.asarray(a, dtype=np.float64))
        frames += 1
        al, ar = info.get("osc_active_L", 0), info.get("osc_active_R", 0)
        both += (al > 0 and ar > 0)
        drive_fr += (al > 0 or ar > 0)
        # ⚠ 判"触地"必须用**奖励自己用的那把尺**（_foot_contact_state 的真接触），
        #   不能用"球底离地<2cm"的 proximity —— :1188 的注释写明后者会把 62%/38% 的腾空误判成支撑。
        _hitL, _hitR, _, _ = w._foot_contact_state(m, dd)
        for s, act, hit in (("left", al, _hitL), ("right", ar, _hitR)):
            if hit and not prev[s]:
                land_ang[s].append(-float(np.degrees(dd.qpos[hidx[s]])))    # +=大腿朝前（同 CMU 手号）
                land_tot += 1
                land_in += (act > 0)
            prev[s] = hit
        if term or trunc:
            break
    ext = land_in / max(land_tot, 1)
    inner = float(info.get("osc_land_cover", 0.0))
    if flag_cover_mismatch is not None and abs(ext - inner) > 1e-9:
        flag_cover_mismatch.append((seed, ext, inner))
    return dict(frames=frames, both=both, duty=drive_fr / max(frames, 1),
                cover=ext, cover_inner=inner,
                cap=float(info.get("osc_cap_cut", 0.0)),
                landings=int(info.get("ep_landings", 0) or 0),
                land=land_ang, ret=float(np.nan),
                land_s=float(info.get("land_per_s", np.nan)),
                sp_L=float(info.get("ev_spacing_L_p50", np.nan)),
                sp_R=float(info.get("ev_spacing_R_p50", np.nan)),
                reach_L=float(info.get("ev_reach_L_mean", np.nan)),
                reach_R=float(info.get("ev_reach_R_mean", np.nan)),
                swap=float(info.get("swap_rate", np.nan)),
                ds=float(info.get("double_support_frac", np.nan)),
                fl=float(info.get("flight_frac", np.nan)))


model = SAC.load(MODEL)
OLD = load(RECON, "t1mod")
NEW = load(NEW_SRC, "t2mod")
print(f"\nP2 同权重 {MODEL} · {len(SEEDS)} 个 seed · det rollout（T1=⑮ 的重构，T2=当前）\n")
res = {}
flag_cover_mismatch = None
res["T1"] = [run(OLD, sd, model) for sd in SEEDS]          # T1 没有内部计数器 ⇒ 只信外部算法
flag_cover_mismatch = []
res["T2"] = [run(NEW, sd, model) for sd in SEEDS]
print(f"  内部/外部覆盖率对账（只 T2 有内部计数）：{len(flag_cover_mismatch)} 处不一致 "
      f"⇒ {'一致 ✓' if not flag_cover_mismatch else flag_cover_mismatch}")
flag_cover_mismatch = None
med = lambda rs, k: float(np.median([r[k] for r in rs if np.isfinite(r[k])]))
print(f"  {'指标':22s}{'T1（⑮）':>12}{'T2（⑯）':>12}{'Δ':>10}   人体参照")
REF = {"cover": None, "land_L": 28.8, "land_R": 28.8, "reach_R": 0.233, "ds": 0.27, "fl": 0.0,
       "land_s": 2.82, "swap": None, "frames": None, "duty": None, "both": 0.0, "cap": 0.0}
rows = [("ep_len", "frames"), ("驱动占空比", "duty"), ("两腿同驱帧数", "both"),
        ("D1 触地被驱动覆盖率", "cover"), ("被上限截断次数", "cap"),
        ("落地间距 L", "sp_L"), ("落地间距 R", "sp_R"), ("reach L", "reach_L"),
        ("reach R（D3）", "reach_R"), ("swap_rate", "swap"), ("ds", "ds"), ("flight", "fl"),
        ("落地/s", "land_s")]
for nm, k in rows:
    a, b = med(res["T1"], k), med(res["T2"], k)
    print(f"  {nm:22s}{a:12.3f}{b:12.3f}{b-a:+10.3f}   {('人体 '+format(REF[k],'.2f')) if REF.get(k) is not None else ''}")
for s in ("left", "right"):
    a = np.median([v for r in res["T1"] for v in r["land"][s]] or [np.nan])
    b = np.median([v for r in res["T2"] for v in r["land"][s]] or [np.nan])
    print(f"  {'D2 落地大腿前倾 '+s:22s}{a:12.1f}{b:12.1f}{b-a:+10.1f}   人体 28.8 (p10 25.4)")
print("\n结构不变量：T2 的『同一帧两腿都在驱动』必须为 0 帧（⑪ 的反相保证）；"
      "T1 的那条由时钟半周期保证，T2 由『双腾空只留腾空更久的那条』仲裁保证。")
