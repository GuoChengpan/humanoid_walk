"""⑰ 决策前第 5 件测量：模板强度 k 的"作者权 / 覆盖 / 收入 / 交替"四方扫描。

为什么这轮要扫它：#2 修好之后覆盖率从 0.006 跳到 0.994，意味着**整个腾空期的髋指令都是模板给的**
（`osc_k_start=1.0` 是完全覆盖：action[i] ← −A，策略在这个关节上被完全改写）。
所以"这是纯 RL 还是我们导入步态"这个问题现在是有物理答案的，答案就在 k 上。

量四个东西，同一个标本（⑯ 权重 det）、同一批 seed、同一套价格：
  1 覆盖：触地那一帧模板仍在驱动该腿的比例（#2 的收益，必须守住）
  2 作者权：被驱动帧上**模板改写掉多少指令**，单位换成 N·m（hip_y gear=300，ctrlrange ±0.4 ⇒ ±120 N·m），
    并给"策略本来想往反方向踢"的比例——这是"RL 还剩多少决定权"的直接读数
  3 收入：现价净分/集 + 各通道，另附**新通道 place**（偏移式 clip((脚−CoM+0.06)/0.26 ×支撑，面值1）
    看每一格配置能为 ⑰ 的付款对象挣多少，这是"值不值得续训"的量化前提
  4 交替/步态结构：duty、ds、land/s、v、髋反相 activity、触地脚−CoM 的 p50/p90（迈没迈出去）

参照行：现状 ⑯·a（双腾空谁都不驱）k=1.0 —— 它就是 ⑯ 实际跑完 4M 的那套。
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
VPATH = "/tmp/train_b19b_longer.py"
assert txt.count(ANCHOR) == 1
open(VPATH, "w", encoding="utf-8").write(txt.replace(ANCHOR, LONGER, 1))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


W_CUR = load(SRC, "b19b_cur")
W_LNG = load(VPATH, "b19b_longer")
MODEL = os.environ.get("PROBE_MODEL",
                       "best_model/tmp_model_data/SAC_19/19_5_厥着腚高速原地踏步/best_model.zip")
SEEDS = tuple(int(x) for x in os.environ.get("S18_SEEDS", "123,2026,7,31,55,77,91,101").split(","))
FR = 1500
DT = 0.015
GEAR_HIP = 300.0

model = SAC.load(MODEL)
del model.replay_buffer
gc.collect()


def roll(Wmod, k, A, seed):
    raw = gym.make("Humanoid-v4", max_episode_steps=FR)
    Wmod.patch_mirror_symmetry(raw.unwrapped.model)
    w = Wmod.WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=True)
    w.osc_k_start = k
    w.osc_A = A
    w.reset(seed=seed)
    m, d = w.unwrapped.model, w.unwrapped.data
    al = m.jnt_qposadr[m.joint("left_hip_y").id]
    ar = m.jnt_qposadr[m.joint("right_hip_y").id]
    land = cov = dbl = 0
    ovr = []            # 被改写掉的指令（归一化单位）
    disagree = 0
    drv_frames = 0
    sup, at_land, jl, jr, cL, cR, ds, fl, vx = [], [], [], [], 0, 0, 0, 0, []
    r_sum = 0.0
    ph_sum = st_sum = rc_sum = place_sum = 0.0
    prev = (False, False)
    n = 0
    for _ in range(FR):
        a, _ = model.predict(w.unwrapped._get_obs(), deterministic=True)
        # 步前接触状态（与 wrapper :1311 同一判据同一时刻）
        hL0, hR0, _, _ = w._foot_contact_state(m, d)
        dr = w.osc_being_driven if n > 0 else (False, False)
        for si, s in ((0, "left"), (1, "right")):
            if si < len(dr) and dr[si]:
                drv_frames += 1
                pre = float(a[w.osc_aidx[s]])
                post = float(np.clip((1 - k) * pre + k * (-A), -1, 1))
                ovr.append(abs(post - pre))
                if (pre > 0) != (post > 0) and abs(post - pre) > 1e-3:
                    disagree += 1
        _, r, ter, tru, info = w.step(a)
        if ter or tru:
            break
        n += 1
        r_sum += r
        ph_sum += float(info.get("phase_reward_gated", 0.0) or 0.0)
        st_sum += float(info.get("stride_bonus_gated", 0.0) or 0.0)
        rc_sum += float(info.get("reach_reward_gated", 0.0) or 0.0)
        hitL, hitR, _, _ = w._foot_contact_state(m, d)
        L = ((not prev[0] and hitL) or (not prev[1] and hitR))
        land += int(L)
        if L:
            cov += int(w.osc_being_driven[0] or w.osc_being_driven[1])
            fx = d.body("left_foot").xpos[0] if (not prev[0] and hitL) else d.body("right_foot").xpos[0]
            at_land.append(fx - d.qpos[0])
        dbl += int(w.osc_being_driven[0] and w.osc_being_driven[1])
        ds += int(hitL and hitR)
        fl += int((not hitL) and (not hitR))
        cL += int(hitL); cR += int(hitR)
        com = d.qpos[0]
        if hitL:
            sup.append(d.body("left_foot").xpos[0] - com)
        if hitR:
            sup.append(d.body("right_foot").xpos[0] - com)
        place_sum += (max(0.0, min(1.0, (d.body("left_foot").xpos[0] - com + 0.06) / 0.26)) * int(hitL)
                      + max(0.0, min(1.0, (d.body("right_foot").xpos[0] - com + 0.06) / 0.26)) * int(hitR))
        jl.append(d.qpos[al]); jr.append(d.qpos[ar])
        vx.append(d.qvel[0])
        prev = (hitL, hitR)
    ap = np.array(at_land) if at_land else np.array([np.nan])
    sp = np.array(sup) if sup else np.array([np.nan])
    # 髋反相 activity（:893 的口径，50 帧窗）
    anti = []
    for i in range(50, n):
        wl = np.array(jl[i - 50:i]); wr = np.array(jr[i - 50:i])
        anti.append((1 - np.corrcoef(wl - wl.mean(), wr - wr.mean())[0, 1]) / 2)
    return dict(n=n, cover=cov / max(land, 1), dbl=dbl, land=land, duty=(cL + cR) / max(2 * n, 1),
                ds=ds / max(n, 1), flight=fl / max(n, 1), v=float(np.mean(vx)),
                land_s=land / max(n * DT, 1e-9), ret=r_sum, ph=ph_sum, st=st_sum, rc=rc_sum,
                place=place_sum, ovr=float(np.mean(ovr)) * GEAR_HIP * 0.4 if ovr else 0.0,
                ovr_norm=float(np.mean(ovr)) if ovr else 0.0,
                dis=disagree / max(drv_frames, 1), drv_frac=drv_frames / max(2 * n, 1),
                anti=float(np.mean(anti)) if anti else np.nan,
                land_p50=float(np.nanmedian(ap)), land_p90=float(np.nanpercentile(ap, 90)),
                land_max=float(np.nanmax(ap)), sup_p90=float(np.nanpercentile(sp, 90)))


CFGS = [("现状 ⑯·a  k=1.00 A=0.15  （⑯ 实跑）", W_CUR, 1.0, 0.15),
        ("#2 更久者驱 k=1.00 A=0.15", W_LNG, 1.0, 0.15),
        ("#2 更久者驱 k=0.50 A=0.15", W_LNG, 0.5, 0.15),
        ("#2 更久者驱 k=0.25 A=0.15", W_LNG, 0.25, 0.15),
        ("#2 更久者驱 k=1.00 A=0.35（幅度对照）", W_LNG, 1.0, 0.35),
        ("#2 更久者驱 k=0.00 A=0.15（纯RL对照）", W_LNG, 0.0, 0.15)]

print(f"{'配置':40s} {'存活':>5s} {'覆盖':>6s} {'同驱':>4s} {'改写N·m':>8s} {'反着来':>7s} {'驱动帧占比':>9s}")
res = {}
for label, Wm, k, A in CFGS:
    r = [roll(Wm, k, A, s) for s in SEEDS]
    med = lambda key, rr=r: float(np.median([x[key] for x in rr]))   # 必须绑 rr=r，否则闭包共享同一个 r
    res[label] = med
    print(f"{label:40s} {med('n'):5.0f} {med('cover'):6.3f} {med('dbl'):4.0f} {med('ovr'):8.1f} "
          f"{med('dis'):7.2f} {med('drv_frac'):9.3f}")
    if med("dbl") > 0:
        print(f"    !! 双腿同驱帧 {med('dbl'):.0f} > 0 ⇒ ⑪ 的反相不变量被破坏，这格不可用")

print(f"\n{'配置':40s} {'duty':>6s} {'ds':>6s} {'腾空':>6s} {'land/s':>7s} {'v':>6s} {'髋反相':>7s}")
for label, *_ in CFGS:
    med = res[label]
    print(f"{label:40s} {med('duty'):6.3f} {med('ds'):6.3f} {med('flight'):6.3f} {med('land_s'):7.2f} "
          f"{med('v'):6.3f} {med('anti'):7.3f}")

print(f"\n{'配置':40s} {'净收入':>7s} {'phase':>7s} {'stride':>7s} {'reach':>7s} {'place新道':>9s} "
      f"{'触地脚−CoM p50/p90/max':>22s}")
for label, *_ in CFGS:
    med = res[label]
    print(f"{label:40s} {med('ret'):7.0f} {med('ph'):7.0f} {med('st'):7.0f} {med('rc'):7.0f} "
          f"{med('place'):9.0f}   {med('land_p50'):+6.3f}/{med('land_p90'):+6.3f}/{med('land_max'):+6.3f}")
print("\n人参照：触地瞬间 脚在 CoM 前方 +0.233 m（CMU 9-clip IC lead）。place 列是偏移式新通道在面值 1 下的分/集。")
