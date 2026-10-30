"""⑰ 决策前第 6 件测量：把驱动窗口**延到触地之后 N 帧**（落地并保持前伸）。

为什么这是纯 RL 路线上最后一个没试过的梯度来源：k 扫描（probe_19b）给出两条硬事实——
  · k≤0.5（保留作者权）时，触地脚−CoM 的最大值 ≤ +0.005 m ⇒ **放置从不发生**；
  · 只有 k=1.0 出现过 +0.065 m，但那是模板整个腾空期接管换来的。
人走路在 initial contact 之后**还在继续屈髋**（把身体拉到脚上），而我们这个装置没有踝、
不能蹬地，"把身体拉到前脚上"是它唯一可能的推进机制。T2 恰恰在触地那一帧把驱动**关掉**了。
所以这里试的是：脚落地后再把髋往前带 N 帧，会（a）出现放置吗，（b）出现双支撑吗，（c）稳不稳。

实现（只改驱动窗口，价格一行不动）：
  want[side] = 腾空窗口(现状) **或** 落地后 1..N 帧的保持窗口；
  两个窗口同时想驱动时**腾空者优先**（否则就变成两髋同时前摆=蹲跳，破坏 ⑪ 的反相不变量）；
  双腾空仍按 #2「腾空更久者驱动，完全平手不驱」。

**不变性自检**：N=0 时新增的 or 分支恒假 ⇒ 结果必须与 probe_19b 的
「#2 更久者驱 k=1.0」一行逐指标相同（cover 0.994 / 净收入 1380 / place 90 / 触地 max +0.065）。
对不上就是补丁越界了，整表作废。
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
A2 = 'want["left"] = want["right"] = False'
assert txt.count(A2) == 1
WANT = '''            want = {"left": (not _hitL) and 1 <= self._osc_air["left"] <= self.osc_wmax,
                    "right": (not _hitR) and 1 <= self._osc_air["right"] <= self.osc_wmax}'''
assert txt.count(WANT) == 1, "落地保持补丁的锚点没匹配上"
NEW = '''            self._osc_post = getattr(self, "_osc_post", {"left": 0, "right": 0})
            for s, hit in (("left", _hitL), ("right", _hitR)):
                self._osc_post[s] = self._osc_post[s] + 1 if hit else 0
            _pn = getattr(self, "osc_post_n", 0)
            want = {"left": (((not _hitL) and 1 <= self._osc_air["left"] <= self.osc_wmax) or
                             (_hitL and 1 <= self._osc_post["left"] <= _pn)),
                    "right": (((not _hitR) and 1 <= self._osc_air["right"] <= self.osc_wmax) or
                              (_hitR and 1 <= self._osc_post["right"] <= _pn))}
            if want["left"] and want["right"]:
                if _hitL != _hitR:
                    want["left"], want["right"] = _hitR, _hitL        # 腾空者优先
                else:
                    _dl, _dr = self._osc_air["left"], self._osc_air["right"]
                    want["left"], want["right"] = (_dl > _dr), (_dr > _dl)'''
VPATH = "/tmp/train_b19c_posthold.py"
src2 = txt.replace(WANT, NEW, 1).replace(A2, ('_dl, _dr = self._osc_air["left"], self._osc_air["right"]\n'
                                             '                want["left"], want["right"] = (_dl > _dr), (_dr > _dl)'), 1)
# 原 A2 那条语句现在落在被替换块的 if 分支里，需要确认它已被上面第二个 replace 覆盖掉
assert 'want["left"] = want["right"] = False' not in src2, "双腾空仲裁没被替换干净"
open(VPATH, "w", encoding="utf-8").write(src2)

spec = importlib.util.spec_from_file_location("b19c", VPATH)
W = importlib.util.module_from_spec(spec)
sys.modules["b19c"] = W
spec.loader.exec_module(W)

MODEL = os.environ.get("PROBE_MODEL",
                       "best_model/tmp_model_data/SAC_19/19_5_厥着腚高速原地踏步/best_model.zip")
SEEDS = tuple(int(x) for x in os.environ.get("S18_SEEDS", "123,2026,7,31,55,77,91,101").split(","))
FR, DT, GEAR = 1500, 0.015, 300.0
model = SAC.load(MODEL)
del model.replay_buffer
gc.collect()


def roll(N, k, A, seed):
    raw = gym.make("Humanoid-v4", max_episode_steps=FR)
    W.patch_mirror_symmetry(raw.unwrapped.model)
    w = W.WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=True)
    w.osc_k_start, w.osc_A, w.osc_post_n = k, A, N
    w.reset(seed=seed)
    m, d = w.unwrapped.model, w.unwrapped.data
    al = m.jnt_qposadr[m.joint("left_hip_y").id]
    ar = m.jnt_qposadr[m.joint("right_hip_y").id]
    land = cov = hold = dbl = n = 0
    cL = cR = ds = fl = 0
    sup, at_land, jl, jr, vx, ovr = [], [], [], [], [], []
    r_sum = ph = st = rc = plc = 0.0
    prev = (False, False)
    for _ in range(FR):
        a, _ = model.predict(w.unwrapped._get_obs(), deterministic=True)
        hL0, hR0, _, _ = w._foot_contact_state(m, d)
        dr = w.osc_being_driven if n > 0 else (False, False)
        for si, (s, h) in ((0, ("left", hL0)), (1, ("right", hR0))):
            if si < len(dr) and dr[si]:
                if h:
                    hold += 1                      # 这一帧脚踩在地上却还在被往前带
                pre = float(a[w.osc_aidx[s]])
                ovr.append(abs(float(np.clip((1 - k) * pre + k * (-A), -1, 1)) - pre))
        dbl += int(dr[0] and dr[1]) if n > 0 else 0
        _, r, ter, tru, info = w.step(a)
        if ter or tru:
            break
        n += 1
        r_sum += r
        ph += float(info.get("phase_reward_gated", 0.0) or 0.0)
        st += float(info.get("stride_bonus_gated", 0.0) or 0.0)
        rc += float(info.get("reach_reward_gated", 0.0) or 0.0)
        hitL, hitR, _, _ = w._foot_contact_state(m, d)
        cov += int((((not prev[0] and hitL) or (not prev[1] and hitR))
                    and (w.osc_being_driven[0] or w.osc_being_driven[1])))
        land += int((not prev[0] and hitL) or (not prev[1] and hitR))
        if (not prev[0] and hitL) or (not prev[1] and hitR):
            at_land.append((d.body("left_foot" if not prev[0] and hitL else "right_foot").xpos[0]) - d.qpos[0])
        ds += int(hitL and hitR); fl += int((not hitL) and (not hitR))
        cL += int(hitL); cR += int(hitR)
        com = d.qpos[0]
        if hitL:
            sup.append(d.body("left_foot").xpos[0] - com)
        if hitR:
            sup.append(d.body("right_foot").xpos[0] - com)
        plc += (max(0.0, min(1.0, (d.body("left_foot").xpos[0] - com + 0.06) / 0.26)) * int(hitL)
                + max(0.0, min(1.0, (d.body("right_foot").xpos[0] - com + 0.06) / 0.26)) * int(hitR))
        jl.append(d.qpos[al]); jr.append(d.qpos[ar]); vx.append(d.qvel[0])
        prev = (hitL, hitR)
    ap = np.array(at_land) if at_land else np.array([np.nan])
    anti = [0.5 * (1 - np.corrcoef(np.array(jl[i - 50:i]) - np.mean(jl[i - 50:i]),
                                   np.array(jr[i - 50:i]) - np.mean(jr[i - 50:i]))[0, 1]) for i in range(50, n)]
    return dict(n=n, cover=cov / max(land, 1), hold=hold / max(n, 1), dbl=dbl, land=land,
                duty=(cL + cR) / max(2 * n, 1), ds=ds / max(n, 1), flight=fl / max(n, 1),
                v=float(np.mean(vx)), land_s=land / max(n * DT, 1e-9), ret=r_sum, ph=ph, st=st, rc=rc,
                plc=plc, ovr=float(np.mean(ovr)) * GEAR * 0.4 if ovr else 0.0,
                anti=float(np.mean(anti)) if anti else np.nan,
                l_p50=float(np.nanmedian(ap)), l_p90=float(np.nanpercentile(ap, 90)),
                l_max=float(np.nanmax(ap)), sup_max=float(np.max(sup)) if sup else np.nan)


CFGS = [("N=0  k=1.00（不变性自检=probe_19b 那行）", 0, 1.0, 0.15),
        ("N=3  k=1.00", 3, 1.0, 0.15),
        ("N=6  k=1.00", 6, 1.0, 0.15),
        ("N=10 k=1.00", 10, 1.0, 0.15),
        ("N=6  k=0.50（保持+部分作者权）", 6, 0.5, 0.15),
        ("N=15 k=1.00（过头检查）", 15, 1.0, 0.15)]
res = {}
print(f"{'配置':42s} {'存活':>5s} {'覆盖':>6s} {'保持帧占比':>10s} {'同驱':>4s} {'改写N·m':>8s} "
      f"{'v':>6s} {'duty':>6s} {'ds':>6s} {'land/s':>7s}")
for label, N, k, A in CFGS:
    r = [roll(N, k, A, s) for s in SEEDS]
    med = lambda key, rr=r: float(np.median([x[key] for x in rr]))
    res[label] = med
    print(f"{label:42s} {med('n'):5.0f} {med('cover'):6.3f} {med('hold'):10.3f} {med('dbl'):4.0f} "
          f"{med('ovr'):8.1f} {med('v'):6.3f} {med('duty'):6.3f} {med('ds'):6.3f} {med('land_s'):7.2f}")
    if med("dbl") > 0:
        print(f"    !! 双腿同驱 {med('dbl'):.0f} 帧 ⇒ 反相不变量被破坏")

print(f"\n{'配置':42s} {'净收入':>7s} {'phase':>7s} {'stride':>6s} {'reach':>6s} {'place梯子':>9s} "
      f"{'触地脚−CoM p50/p90/max':>22s} {'支撑max':>8s}")
for label, *_ in CFGS:
    med = res[label]
    print(f"{label:42s} {med('ret'):7.0f} {med('ph'):7.0f} {med('st'):6.0f} {med('rc'):6.0f} "
          f"{med('plc'):9.0f}   {med('l_p50'):+6.3f}/{med('l_p90'):+6.3f}/{med('l_max'):+6.3f} "
          f"{med('sup_max'):+8.3f}")
print("\n人参照：触地瞬间脚在 CoM 前方 +0.233 m。N=0 行必须与 probe_19b 的 k=1.0 行一致，否则全表作废。")
