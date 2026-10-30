"""⑰ 第 6 件测量（v2）：落地后保持驱动 N 帧 —— 两种优先序对照。

v1 的失败要写出来（否则下一次还会犯）：我加的仲裁是"**腾空者优先**"，而这个步态的绝大多数帧
是单支撑（一腿在地上、一腿在天上），于是"地上那条腿的保持窗口"永远输给"天上那条"，
**保持其实一次都没生效**——表现就是 N=10 与 N=15 两行和 N=0 逐指标完全相同。
v1 还有个计数错位：`osc_being_driven` 是在 step 内、用**步前**接触状态写的，我却拿它跟
**上一步之后**的接触状态配对 ⇒ "保持帧占比"连 N=0 都读到 0.111（那是触地后第一帧的占比）。

v2 两件事：
  · 优先序做成开关：mode=0 腾空者优先（=v1，预期不生效，作对照），mode=1 **支撑者优先**
    （保持真正落地，代价是那一帧腾空腿不被驱动 ⇒ 仍满足"每帧至多一腿被驱"）
  · hold 计数改成正确配对：步前接触 × 本步实际使用的决策（step 之后读回）
  · 顺带把**单次支撑帧长**量出来：若 stance 只有 ~6 帧，N>6 本来就必然等于 N=6，读数要有这个预期
"""
import importlib.util
import re
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
# 一次性替换整段：want 的判据 + 它后面那段 ⑯·a 双腾空仲裁（含注释与 osc_double_air 计数）。
PAT = re.compile(r"[ ]{12}want = \{\"left\": \(not _hitL\)[\s\S]*?self\.osc_double_air \+= 1\n")
m = PAT.search(txt)
assert m, "锚段没匹配上：want+双腾空仲裁的整段要先核对再替换"
NEW = ('            self._osc_post = getattr(self, "_osc_post", {"left": 0, "right": 0})\n'
       '            for s, hit in (("left", _hitL), ("right", _hitR)):\n'
       '                self._osc_post[s] = self._osc_post[s] + 1 if hit else 0\n'
       '            _pn = getattr(self, "osc_post_n", 0)\n'
       '            _pm = getattr(self, "osc_post_mode", 0)\n'
       '            want = {"left": (((not _hitL) and 1 <= self._osc_air["left"] <= self.osc_wmax) or\n'
       '                             (_hitL and 1 <= self._osc_post["left"] <= _pn)),\n'
       '                    "right": (((not _hitR) and 1 <= self._osc_air["right"] <= self.osc_wmax) or\n'
       '                              (_hitR and 1 <= self._osc_post["right"] <= _pn))}\n'
       '            if want["left"] and want["right"]:\n'
       '                if _hitL != _hitR:\n'
       '                    if _pm == 0:\n'
       '                        want["left"], want["right"] = _hitR, _hitL      # 腾空者优先（v1，预期不生效）\n'
       '                    else:\n'
       '                        want["left"], want["right"] = _hitL, _hitR      # 支撑者优先（保持生效）\n'
       '                elif _hitL and _hitR:\n'
       '                    want["left"] = want["right"] = False                # 双落地同驱=蹲跳，禁\n'
       '                else:\n'
       '                    _dl, _dr = self._osc_air["left"], self._osc_air["right"]\n'
       '                    want["left"], want["right"] = (_dl > _dr), (_dr > _dl)\n'
       '                    self.osc_double_air += 1\n')
src2 = txt[:m.start()] + NEW + txt[m.end():]
import re as _r
assert _r.search(r'want = \{\"left\": \(not _hitL\)', src2) is None, "原 want 判据还在"
assert src2.count("osc_double_air += 1") == 1, "osc_double_air 计数被复制或多份 ⇒ 替换边界不对"
assert "\u3010\u4fee\u6b63 \u246f\u00b7a" not in src2, "⑯·a 的注释块还在，说明 span 没吃干净"
open("/tmp/train_b19d_posthold2.py","w",encoding="utf-8").write(src2)
VPATH = "/tmp/train_b19d_posthold2.py"
open(VPATH, "w", encoding="utf-8").write(src2)

spec = importlib.util.spec_from_file_location("b19d", VPATH)
W = importlib.util.module_from_spec(spec)
sys.modules["b19d"] = W
spec.loader.exec_module(W)

MODEL = os.environ.get("PROBE_MODEL",
                       "best_model/tmp_model_data/SAC_19/19_5_厥着腚高速原地踏步/best_model.zip")
SEEDS = tuple(int(x) for x in os.environ.get("S18_SEEDS", "123,2026,7,31,55,77,91,101").split(","))
FR, DT, GEAR = 1500, 0.015, 300.0
model = SAC.load(MODEL)
del model.replay_buffer
gc.collect()


def roll(N, mode, k, A, seed):
    raw = gym.make("Humanoid-v4", max_episode_steps=FR)
    W.patch_mirror_symmetry(raw.unwrapped.model)
    w = W.WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=True)
    w.osc_k_start, w.osc_A, w.osc_post_n, w.osc_post_mode = k, A, N, mode
    w.reset(seed=seed)
    m, d = w.unwrapped.model, w.unwrapped.data
    al = m.jnt_qposadr[m.joint("left_hip_y").id]
    ar = m.jnt_qposadr[m.joint("right_hip_y").id]
    land = cov = hold = dbl = n = driven = 0
    cL = cR = ds = fl = 0
    stance_run = []
    cur_stance = 0
    sup, at_land, jl, jr, vx, ovr = [], [], [], [], [], []
    r_sum = ph = st = rc = plc = 0.0
    prev = (False, False)
    for _ in range(FR):
        a, _ = model.predict(w.unwrapped._get_obs(), deterministic=True)
        hL0, hR0, _, _ = w._foot_contact_state(m, d)          # 步前：本步 step 内部用的就是这一对
        _, r, ter, tru, info = w.step(a)
        if ter or tru:
            break
        dr = w.osc_being_driven                                # 本步**实际使用**的决策，与本步的接触口径配对
        for si, (s, h) in ((0, ("left", hL0)), (1, ("right", hR0))):
            if dr[si]:
                driven += 1
                if h:
                    hold += 1                                  # 脚在地上却仍被往前带 = 保持真的生效
                pre = float(a[w.osc_aidx[s]])
                ovr.append(abs(float(np.clip((1 - k) * pre + k * (-A), -1, 1)) - pre))
        dbl += int(dr[0] and dr[1])
        n += 1
        r_sum += r
        ph += float(info.get("phase_reward_gated", 0.0) or 0.0)
        st += float(info.get("stride_bonus_gated", 0.0) or 0.0)
        rc += float(info.get("reach_reward_gated", 0.0) or 0.0)
        hitL, hitR, _, _ = w._foot_contact_state(m, d)
        L = (not prev[0] and hitL) or (not prev[1] and hitR)
        land += int(L)
        cov += int(L and (dr[0] or dr[1]))
        if L:
            at_land.append((d.body("left_foot" if not prev[0] and hitL else "right_foot").xpos[0]) - d.qpos[0])
        ds += int(hitL and hitR); fl += int((not hitL) and (not hitR))
        cL += int(hitL); cR += int(hitR)
        if hitL or hitR:
            cur_stance += 1
        elif cur_stance:
            stance_run.append(cur_stance); cur_stance = 0
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
    return dict(n=n, cover=cov / max(land, 1), hold=hold / max(n, 1), drv=driven / max(2 * n, 1),
                dbl=dbl, land=land, duty=(cL + cR) / max(2 * n, 1), stance=float(np.median(stance_run)) if stance_run else np.nan,
                ds=ds / max(n, 1), flight=fl / max(n, 1), v=float(np.mean(vx)),
                land_s=land / max(n * DT, 1e-9), ret=r_sum, ph=ph, st=st, rc=rc, plc=plc,
                ovr=float(np.mean(ovr)) * GEAR * 0.4 if ovr else 0.0,
                anti=float(np.mean(anti)) if anti else np.nan,
                l_p50=float(np.nanmedian(ap)), l_p90=float(np.nanpercentile(ap, 90)),
                l_max=float(np.nanmax(ap)), sup_max=float(np.max(sup)) if sup else np.nan)


CFGS = [("mode=空 N=0  k=1.00（不变性自检）", 0, 0, 1.0, 0.15),
        ("mode=空 N=6  k=1.00", 6, 0, 1.0, 0.15),
        ("mode=撑 N=3  k=1.00", 3, 1, 1.0, 0.15),
        ("mode=撑 N=6  k=1.00", 6, 1, 1.0, 0.15),
        ("mode=撑 N=12 k=1.00", 12, 1, 1.0, 0.15),
        ("mode=撑 N=6  k=0.50", 6, 1, 0.5, 0.15)]
res = {}
print(f"{'配置':38s} {'存活':>5s} {'驱动帧':>6s} {'保持帧':>6s} {'同驱':>4s} {'单次stance':>10s} "
      f"{'改写N·m':>8s} {'v':>6s} {'duty':>6s} {'ds':>6s} {'land/s':>7s}")
for label, N, md, k, A in CFGS:
    r = [roll(N, md, k, A, s) for s in SEEDS]
    med = lambda key, rr=r: float(np.median([x[key] for x in rr]))
    res[label] = med
    print(f"{label:38s} {med('n'):5.0f} {med('drv'):6.3f} {med('hold'):6.3f} {med('dbl'):4.0f} "
          f"{med('stance'):10.1f} {med('ovr'):8.1f} {med('v'):6.3f} {med('duty'):6.3f} {med('ds'):6.3f} "
          f"{med('land_s'):7.2f}")
    if med("dbl") > 0:
        print(f"    !! 双腿同驱 {med('dbl'):.0f} 帧 ⇒ 反相不变量被破坏，这格不可用")

print(f"\n{'配置':38s} {'净收入':>7s} {'phase':>7s} {'stride':>6s} {'reach':>6s} {'place梯子':>9s} "
      f"{'髋反相':>7s} {'触地脚−CoM p50/p90/max':>22s}")
for label, *_ in CFGS:
    med = res[label]
    print(f"{label:38s} {med('ret'):7.0f} {med('ph'):7.0f} {med('st'):6.0f} {med('rc'):6.0f} "
          f"{med('plc'):9.0f} {med('anti'):7.3f}   {med('l_p50'):+6.3f}/{med('l_p90'):+6.3f}/{med('l_max'):+6.3f}")
print("\n人参照：触地瞬间脚在 CoM 前方 +0.233 m。第一行必须与 probe_19b 的「#2 k=1.0」一致"
      "（cover .994 / 净 1380 / phase 1000 / place 90 / max +0.065）。")
