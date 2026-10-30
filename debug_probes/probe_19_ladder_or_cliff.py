"""⑰ 落地前的最后两件测量：VB 是梯子还是悬崖；#2 该写成什么形式。

A. **支撑前脚操作量的局部分布**。VB 把 `phase_reward` 的 1339 分/集搬到
   `clip((脚x − CoMx)/D) × 该脚在支撑` 上。march 盆在 D=0.20 下只累计到 0.2/1499 ⇒
   **它现在一分都拿不到，而且拿不到不是"钱少"是"梯度为零"**：策略没有理由往那个方向挪第一步。
   所以要扫 D：如果 D 小到让 march 盆也开始有钱，那 VB 就变成"给现盆打折"，不是悬崖；
   如果 D 多大都还是 0，那 VB 就是**盲谷**，必须配一条残值（保留一部分 phase）才敢 warm start。
   同时要给出**触地瞬间**的脚-身体前后关系（那才是"迈出去"的定义量）。

B. **#2 的正确形式**。我 ⑯·a 写的"双腾空谁都不驱动"把模板在触地前关掉了
   （`osc_land_cover` 0.209→0.006）。但当时撤掉的 ⑪ 原规则"驱动腾空更久的那条腿"
   **本身就是镜像等变的**（交换左右标签，判据跟着交换）；真正破坏等变性的只有"两腿腾空数
   相等时用绝对时钟 phase<0.5 打破平手"那一句。所以正确形式是
   **腾空更久者驱动；完全平手时谁都不驱动**（平手=同一帧双脚踏出，交换标签仍是平手 ⇒ 等变）。
   这里量的是它的覆盖率与截断数，等变性由构造保证，另附一个逐帧对称性自检。
"""
import importlib.util
import os
import sys
import gc
import warnings

import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")

SRC = "train_humanoid_sac.py"
txt = open(SRC, encoding="utf-8").read()
ANCHOR = 'want["left"] = want["right"] = False'
LONGER = ('_dl, _dr = self._osc_air["left"], self._osc_air["right"]\n'
          '                want["left"], want["right"] = (_dl > _dr), (_dr > _dl)')
VPATH = "/tmp/train_b19_longer_air.py"
assert txt.count(ANCHOR) == 1
open(VPATH, "w", encoding="utf-8").write(txt.replace(ANCHOR, LONGER, 1))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


W0 = load(SRC, "b19_current")
W1 = load(VPATH, "b19_longer")
MODEL = os.environ.get("PROBE_MODEL", "best_model/best_model.zip")
SEEDS = tuple(int(x) for x in os.environ.get("S18_SEEDS", "123,2026,7,31,55,77,91,101").split(","))
FR = 1500
model = SAC.load(MODEL)
del model.replay_buffer
gc.collect()


def roll(Wmod, seed, rec_cover=False):
    raw = gym.make("Humanoid-v4", max_episode_steps=FR)
    Wmod.patch_mirror_symmetry(raw.unwrapped.model)
    w = Wmod.WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=True)
    w.reset(seed=seed)
    m, d = w.unwrapped.model, w.unwrapped.data
    sup = {s: [] for s in ("left", "right")}          # 支撑帧的 脚x − CoMx
    at_land = {s: [] for s in ("left", "right")}      # 触地瞬间同量
    lead_arr = []
    land = in_drive = 0
    prev = (False, False)
    both_driven = 0
    n = 0
    for _ in range(FR):
        a, _ = model.predict(w.unwrapped._get_obs(), deterministic=True)
        _, r, ter, tru, info = w.step(a)
        if ter or tru:
            break
        n += 1
        dr = getattr(w, "osc_being_driven", (False, False))
        both_driven += int(dr[0] and dr[1])
        hitL, hitR, _, _ = w._foot_contact_state(m, d)
        com = d.qpos[0]
        fx = {"left": d.body("left_foot").xpos[0], "right": d.body("right_foot").xpos[0]}
        if hitL:
            sup["left"].append(fx["left"] - com)
        if hitR:
            sup["right"].append(fx["right"] - com)
        if (not prev[0] and hitL) or (not prev[1] and hitR):
            land += 1
            lead_arr.append(dr[0] or dr[1])          # 触地这帧是否仍在驱动窗口内
            if not prev[0] and hitL:
                at_land["left"].append(fx["left"] - com)
            if not prev[1] and hitR:
                at_land["right"].append(fx["right"] - com)
        prev = (hitL, hitR)
    return dict(n=n, sup=sup, at_land=at_land, cover=np.mean(lead_arr) if lead_arr else np.nan,
                land=land, both=both_driven,
                cap=float(np.mean([getattr(w, "osc_cap_cut", 0)])) if rec_cover else np.nan)


print("=== A. 支撑前脚操作量：march 盆在三种分母下各挣多少（det 实测，8 seeds）===")
R = [roll(W0, s) for s in SEEDS]
for s_ in ("left", "right"):
    v = np.array([x for r in R for x in r["sup"][s_]])
    w_ = np.array([x for r in R for x in r["at_land"][s_]])
    print(f"  {s_:5s} 支撑帧 脚x−CoMx (m) p10/p50/p90 = {np.percentile(v,10):+.3f}/{np.median(v):+.3f}/"
          f"{np.percentile(v,90):+.3f}  最大 {v.max():+.3f}   |  触地瞬间 p10/p50/p90 = "
          f"{np.percentile(w_,10):+.3f}/{np.median(w_):+.3f}/{np.percentile(w_,90):+.3f}")
for D in (0.20, 0.10, 0.05):
    tot = []
    for r in R:
        f = sum(np.clip(np.array(r["sup"][s_]) / D, 0, 1).sum() for s_ in ("left", "right"))
        tot.append(f)
    print(f"  D={D:.2f} ⇒ 面值1的'支撑前脚' = {np.median(tot):7.1f} / {np.median([r['n'] for r in R]):.0f} 帧"
          f"   换算成 phase 那笔预算(1339 分/集) = {np.median(tot)/max(0.30*np.median([r['n'] for r in R]),1e-9)*1339:6.0f} 分/集")
print("  读法：右列≈0 ⇒ 该分母下现盆**拿不到梯度**，VB 是盲谷，warm start 必须配残值；")
print("        右列给出现盆 30% 以上 ⇒ 分母太松，VB 等于给现盆打折。")

print("\n=== B. #2 的两种形式：覆盖率 / 双驱动违规 ===")
print(f"{'驱动仲裁':28s} {'存活帧':>7s} {'落地数':>7s} {'触地被驱动覆盖':>14s} {'双腿同驱帧':>10s}")
for Wm, nm in ((W0, "现状 ⑯·a：双腾空谁都不驱"), (W1, "⑰ 草案：腾空更久者驱，平手不驱")):
    Rb = [roll(Wm, s, rec_cover=True) for s in SEEDS]
    print(f"{nm:28s} {np.median([r['n'] for r in Rb]):7.0f} {np.median([r['land'] for r in Rb]):7.0f} "
          f"{np.nanmedian([r['cover'] for r in Rb]):14.3f} {np.median([r['both'] for r in Rb]):10.0f}")
print("  覆盖率 = 触地那一帧模板仍在驱动该腿的比例（⑪ 的 T1 实测 0.086，⑯ 现状 0.006，eval 早期 0.209）")
print("  '双腿同驱帧' 必须恒为 0，否则 ⑪ 的反相不变量被破坏（⑯·a 当初就是为它改的）")
