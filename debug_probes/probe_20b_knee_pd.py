"""⑰ 第 7 件测量 v2：膝用 **PD 伺服**在摆动后段拉向伸直（v1 的恒力矩不够格当这个仪器）。

v1（probe_20）的结论为什么不能算"膝假设已否证"：它注入的是**恒力矩** `action = ±c`。
摆动期小腿的惯性力矩可以远大于 20–40 N·m，恒力矩既守不住角度也不构成"伸到某角"的指令；
实测也确实是反的——越早开始伸（onset=4/6），触地膝屈反而从 20.8° 涨到 42.9/49.5°、92–110 帧摔死。
而同一套仪器里真正管用的是 18f 按住腰用的那种 **PD**（能把 abdomen_y 稳稳钉在目标角上）。

所以 v2 只做一件事：把注入换成 PD，`τ = kp(θ_target − θ) − kd·θ̇`，ctrl = τ/gear 再截 ±0.4。
补丁只插**一行**调用（`self.knee_fn(...)`），避免 v1 那种多行字符串替换静默失配——
v1 的表与改 PD 后的表**逐格相同**，那就是替换没生效的铁证，所以我另起一支而不是继续打补丁。

口径（全部向量式，正=该节在前方）：触地大腿 +13 = 前倾 13°；**触地小腿 −5.5 = 脚在膝后 5.5°（后甩）**；
膝屈 = 大腿向量与小腿向量夹角。人 IC：大腿 +28.8°、小腿 +3.4°、膝屈 25.4°、脚在 CoM 前 +0.233 m。

不变性自检：onset=off 必须复现 probe_19b 的「#2 k=1.0」行（净 1380 / phase 1000 / stride 393 /
reach 119 / place 90 / 触地 max +0.065 / 反相 0.690）。
判据（观测前钉死）：PASS = 存活 ≥ 1000 且 触地小腿比基线(−5.5°)向 0 靠近 ≥ 4°
且 触地脚−CoM max > +0.065 且 净收入 ≥ 0.7×1380。
"""
import importlib.util
import os
import re
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
PAT = re.compile(r'[ ]{12}want = \{"left": \(not _hitL\)[\s\S]*?self\.osc_double_air \+= 1\n')
m = PAT.search(txt)
assert m, "want+双腾空仲裁整段没匹配上"
NEW_HIP = ('            want = {"left": (not _hitL) and 1 <= self._osc_air["left"] <= self.osc_wmax,\n'
           '                    "right": (not _hitR) and 1 <= self._osc_air["right"] <= self.osc_wmax}\n'
           '            if want["left"] and want["right"]:\n'
           '                _dl, _dr = self._osc_air["left"], self._osc_air["right"]\n'
           '                want["left"], want["right"] = (_dl > _dr), (_dr > _dl)\n'
           '                self.osc_double_air += 1\n')
# 只插一行：在 act_flag = [] 之前（它紧跟在 osc_double_air += 1 之后，缩进 12 空格）
KNEE_LINE = ('            _kn = getattr(self, "knee_onset", 0)\n'
             '            if _kn:\n'
             '                for _s, _h in (("left", _hitL), ("right", _hitR)):\n'
             '                    if (not _h) and self._osc_air[_s] >= _kn:\n'
             '                        action[self.knee_aidx[_s]] = self.knee_fn(self, _s)\n')
assert txt[m.end():m.end() + 40].lstrip().startswith("act_flag = []"), "锚点后不是 act_flag，切片位置要重核"
src2 = txt[:m.start()] + NEW_HIP + KNEE_LINE + txt[m.end():]
assert src2.count("self.knee_fn(self, _s)") == 1 and "phase < 0.5" not in src2.split("act_flag")[0].split("want =")[1]
VPATH = "/tmp/train_b20b_knee_pd.py"
open(VPATH, "w", encoding="utf-8").write(src2)
spec = importlib.util.spec_from_file_location("b20b", VPATH)
W = importlib.util.module_from_spec(spec)
sys.modules["b20b"] = W
spec.loader.exec_module(W)

MODEL = os.environ.get("PROBE_MODEL",
                       "best_model/tmp_model_data/SAC_19/19_5_厥着腚高速原地踏步/best_model.zip")
SEEDS = tuple(int(x) for x in os.environ.get("S18_SEEDS", "123,2026,7,31,55,77,91,101").split(","))
FR, DT = 1500, 0.015
model = SAC.load(MODEL)
del model.replay_buffer
gc.collect()


def roll(onset, target_deg, kp, kd, seed):
    raw = gym.make("Humanoid-v4", max_episode_steps=FR)
    W.patch_mirror_symmetry(raw.unwrapped.model)
    w = W.WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=True)
    w.osc_k_start, w.osc_A = 1.0, 0.15
    m = w.unwrapped.model
    w.knee_aidx = {s: [u for u in range(m.nu)
                       if m.joint(m.actuator_trnid[u][0]).name == f"{s}_knee"][0] for s in ("left", "right")}
    w.knee_onset = onset

    def fn(win, s):
        d = win.env.unwrapped.data
        qa = m.jnt_qposadr[m.joint(f"{s}_knee").id]
        dv = m.jnt_dofadr[m.joint(f"{s}_knee").id]
        tau = kp * (np.radians(target_deg) - d.qpos[qa]) - kd * d.qvel[dv]
        return float(np.clip(tau / m.actuator_gear[w.knee_aidx[s]][0], -0.4, 0.4))

    w.knee_fn = fn
    w.reset(seed=seed)
    d = w.unwrapped.data
    land = n = cL = cR = ds = fl = 0
    thigh_at, shank_at, flex_at, foot_off, sat = [], [], [], [], []
    jl, jr, vx = [], [], []
    r_sum = ph = st = rc = plc = 0.0
    prev = (False, False)
    for _ in range(FR):
        a, _ = model.predict(w.unwrapped._get_obs(), deterministic=True)
        _, r, ter, tru, info = w.step(a)
        if ter or tru:
            break
        n += 1
        r_sum += r
        ph += float(info.get("phase_reward_gated", 0.0) or 0.0)
        st += float(info.get("stride_bonus_gated", 0.0) or 0.0)
        rc += float(info.get("reach_reward_gated", 0.0) or 0.0)
        hitL, hitR, _, _ = w._foot_contact_state(m, d)
        ds += int(hitL and hitR); fl += int((not hitL) and (not hitR))
        cL += int(hitL); cR += int(hitR)
        com = d.qpos[0]
        for s_, h_, pk in (("left", hitL, prev[0]), ("right", hitR, prev[1])):
            fx = d.body(f"{s_}_foot").xpos[0]
            plc += (max(0.0, min(1.0, (fx - com + 0.06) / 0.26)) if h_ else 0.0)
            if h_ and not pk:                                   # 本帧这条腿刚落地
                th = d.xpos[m.body(f"{s_}_shin").id] - d.xpos[m.body(f"{s_}_thigh").id]
                sk = d.xpos[m.body(f"{s_}_foot").id] - d.xpos[m.body(f"{s_}_shin").id]
                thigh_at.append(np.degrees(np.arctan2(th[0], -th[2])))
                shank_at.append(np.degrees(np.arctan2(sk[0], -sk[2])))
                foot_off.append(fx - com)
                cs = np.dot(th, sk) / max(np.linalg.norm(th) * np.linalg.norm(sk), 1e-9)
                flex_at.append(np.degrees(np.arccos(np.clip(cs, -1, 1))))
                land += 1
        jl.append(np.degrees(d.qpos[m.jnt_qposadr[m.joint("left_hip_y").id]]))
        jr.append(np.degrees(d.qpos[m.jnt_qposadr[m.joint("right_hip_y").id]]))
        vx.append(d.qvel[0])
        prev = (hitL, hitR)
    anti = [0.5 * (1 - np.corrcoef(np.array(jl[i - 50:i]) - np.mean(jl[i - 50:i]),
                                   np.array(jr[i - 50:i]) - np.mean(jr[i - 50:i]))[0, 1]) for i in range(50, n)]
    fo = np.array(foot_off) if foot_off else np.array([np.nan])
    med = lambda v: float(np.median(v)) if len(v) else np.nan
    return dict(n=n, land_s=land / max(n * DT, 1e-9), duty=(cL + cR) / max(2 * n, 1),
                ds=ds / max(n, 1), flight=fl / max(n, 1), v=float(np.mean(vx)), ret=r_sum,
                ph=ph, st=st, rc=rc, plc=plc, anti=float(np.mean(anti)) if anti else np.nan,
                thigh=med(thigh_at), shank=med(shank_at), flex=med(flex_at),
                fo_p50=med(fo), fo_max=float(np.nanmax(fo)))


CFGS = [("onset=off（不变性自检）", 0, -5.0, 120.0, 15.0),
        ("onset=3 tgt=−5° kp=120", 3, -5.0, 120.0, 15.0),
        ("onset=6 tgt=−5° kp=120", 6, -5.0, 120.0, 15.0),
        ("onset=9 tgt=−5° kp=120", 9, -5.0, 120.0, 15.0),
        ("onset=6 tgt=−5° kp=250（更硬）", 6, -5.0, 250.0, 25.0),
        ("onset=6 tgt=−20°（只半伸）", 6, -20.0, 120.0, 15.0)]
res = {}
print("=== 膝 PD 伺服扫描（qpos 负=屈，tgt −5° ≈ 几乎直；onset=摆动第几帧起伺服）===")
print(f"{'配置':30s} {'存活':>5s} {'duty':>6s} {'ds':>6s} {'land/s':>7s} {'v':>6s} "
      f"{'触地大腿':>9s} {'触地小腿':>9s} {'触地膝屈':>9s} {'触地脚−CoM p50/max':>20s}")
for label, on, tg, kp, kd in CFGS:
    r = [roll(on, tg, kp, kd, s) for s in SEEDS]
    med = lambda key, rr=r: float(np.median([x[key] for x in rr]))
    res[label] = med
    print(f"{label:30s} {med('n'):5.0f} {med('duty'):6.3f} {med('ds'):6.3f} {med('land_s'):7.2f} "
          f"{med('v'):6.3f} {med('thigh'):+9.1f} {med('shank'):+9.1f} {med('flex'):9.1f} "
          f"  {med('fo_p50'):+6.3f}/{med('fo_max'):+6.3f}")
print(f"\n{'配置':30s} {'净收入':>7s} {'phase':>7s} {'stride':>6s} {'reach':>6s} {'place梯子':>9s} {'髋反相':>7s}")
for label, *_ in CFGS:
    med = res[label]
    print(f"{label:30s} {med('ret'):7.0f} {med('ph'):7.0f} {med('st'):6.0f} {med('rc'):6.0f} "
          f"{med('plc'):9.0f} {med('anti'):7.3f}")
base = res["onset=off（不变性自检）"]
print("\n判据核算（PASS 需四条同时成立）：")
for label, *_ in CFGS[1:]:
    med = res[label]
    ok = (med("n") >= 1000 and med("shank") - base("shank") >= 4.0
          and med("fo_max") > base("fo_max") and med("ret") >= 0.7 * base("ret"))
    print(f"  {label:30s} 存活{med('n'):5.0f} 小腿Δ{med('shank')-base('shank'):+5.1f}° "
          f"max{med('fo_max'):+.3f} 净{med('ret'):6.0f}  ⇒ {'**PASS**' if ok else 'fail'}")
