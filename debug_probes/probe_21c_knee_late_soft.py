"""⑰ 补测：摆动末段膝「半伸」写死（probe_21 的 B 段是空跑，这里用**带膝注入的模块**重做）。

probe_21 的 B 段九行与基线逐位相同 ⇒ 因为我复用了 S17 模块，它**只替换了双腾空仲裁，没有插入膝驱动行**，
`w.knee_onset` 是个没人读的属性。那不等于"无效应"，等于没测。这里建一个同时含
#2 仲裁 + 膝注入 的模块，并把未测过的区域（**晚介入 + 只半伸 + 软增益**）扫掉。

已知（probe_20b，恒定：早/硬是错的）：onset=3/6/9 拉到 −5°、kp=120 ⇒ 触地膝屈反而 41.6-47.7°、
存活 77-282 帧；而 onset=9 交出全场最高触地脚−CoM max **+0.104**。所以假设是：
**"只在摆动最后两三帧、把膝从当前屈曲位往回带一点点"才有救，拉太早/拉太直都是往前栽。**

判据（观测前钉死）：PASS = 存活 ≥ 1200 且 触地脚−CoM max > 基线(+0.065) 且 净收入 ≥ 0.7×基线。
不变性自检：off 行必须等于 probe_21 的 W16×S17（存活 1499 / 净 1380 / 触地 max +0.065 / 膝屈 20.8）。
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

CUR = "train_humanoid_sac.py"
W16 = "best_model/tmp_model_data/SAC_19/19_5_厥着腚高速原地踏步/best_model.zip"
FR, DT = 1500, 0.015
SEEDS = tuple(int(x) for x in os.environ.get("S18_SEEDS", "123,2026,7,31,55,77,91,101").split(","))

txt = open(CUR, encoding="utf-8").read()
PAT = re.compile(r'[ ]{12}want = \{"left": \(not _hitL\)[\s\S]*?self\.osc_double_air \+= 1\n')
m = PAT.search(txt)
assert m
assert txt[m.end():m.end() + 60].lstrip().startswith("act_flag = []"), "锚点后不是 act_flag"
NEW = ('            want = {"left": (not _hitL) and 1 <= self._osc_air["left"] <= self.osc_wmax,\n'
       '                    "right": (not _hitR) and 1 <= self._osc_air["right"] <= self.osc_wmax}\n'
       '            if want["left"] and want["right"]:\n'
       '                _dl, _dr = self._osc_air["left"], self._osc_air["right"]\n'
       '                want["left"], want["right"] = (_dl > _dr), (_dr > _dl)\n'
       '                self.osc_double_air += 1\n'
       '            _kn = getattr(self, "knee_onset", 0)\n'
       '            if _kn:\n'
       '                for _s, _h in (("left", _hitL), ("right", _hitR)):\n'
       '                    if (not _h) and self._osc_air[_s] >= _kn:\n'
       '                        action[self.knee_aidx[_s]] = self.knee_fn(self, _s)\n')
out = txt[:m.start()] + NEW + txt[m.end():]
assert out.count("self.knee_fn(self, _s)") == 1 and "原来这里用" not in out
VP = "/tmp/train_b21c_knee.py"
open(VP, "w", encoding="utf-8").write(out)
spec = importlib.util.spec_from_file_location("b21c", VP)
Wm = importlib.util.module_from_spec(spec)
sys.modules["b21c"] = Wm
spec.loader.exec_module(Wm)

mo = SAC.load(W16)
del mo.replay_buffer
gc.collect()


def roll(onset, target, kp, seed):
    raw = gym.make("Humanoid-v4", max_episode_steps=FR)
    Wm.patch_mirror_symmetry(raw.unwrapped.model)
    w = Wm.WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=True)
    w.osc_k_start, w.osc_A = 1.0, 0.15
    mmd = w.unwrapped.model
    w.knee_onset = onset
    w.knee_aidx = {s: [u for u in range(mmd.nu)
                       if mmd.joint(mmd.actuator_trnid[u][0]).name == f"{s}_knee"][0] for s in ("left", "right")}

    def fn(win, s):
        d = win.env.unwrapped.data
        qa = mmd.jnt_qposadr[mmd.joint(f"{s}_knee").id]
        dv = mmd.jnt_dofadr[mmd.joint(f"{s}_knee").id]
        tau = kp * (np.radians(target) - d.qpos[qa]) - 8.0 * d.qvel[dv]
        return float(np.clip(tau / mmd.actuator_gear[win.knee_aidx[s]][0], -0.4, 0.4))
    w.knee_fn = fn
    w.reset(seed=seed)
    d = w.unwrapped.data
    n = land = cL = cR = ds = fl = 0
    thigh_at, shank_at, flex_at, foot_off = [], [], [], []
    vx = []
    r_sum = ph = st = plc = 0.0
    prev = (False, False)
    for _ in range(FR):
        a, _ = mo.predict(w.unwrapped._get_obs(), deterministic=True)
        _, r, ter, tru, info = w.step(a)
        if ter or tru:
            break
        n += 1
        r_sum += r
        ph += float(info.get("phase_reward_gated", 0.0) or 0.0)
        st += float(info.get("stride_bonus_gated", 0.0) or 0.0)
        hitL, hitR, _, _ = w._foot_contact_state(mmd, d)
        ds += int(hitL and hitR); fl += int((not hitL) and (not hitR))
        cL += int(hitL); cR += int(hitR)
        com = d.qpos[0]
        for s_, h_, pk in (("left", hitL, prev[0]), ("right", hitR, prev[1])):
            fx = d.body(f"{s_}_foot").xpos[0]
            plc += max(0.0, min(1.0, (fx - com + 0.06) / 0.26)) if h_ else 0.0
            if h_ and not pk:
                th = d.xpos[mmd.body(f"{s_}_shin").id] - d.xpos[mmd.body(f"{s_}_thigh").id]
                sk = d.xpos[mmd.body(f"{s_}_foot").id] - d.xpos[mmd.body(f"{s_}_shin").id]
                thigh_at.append(np.degrees(np.arctan2(th[0], -th[2])))
                shank_at.append(np.degrees(np.arctan2(sk[0], -sk[2])))
                foot_off.append(fx - com)
                cs = np.dot(th, sk) / max(np.linalg.norm(th) * np.linalg.norm(sk), 1e-9)
                flex_at.append(np.degrees(np.arccos(np.clip(cs, -1, 1))))
                land += 1
        vx.append(d.qvel[0])
        prev = (hitL, hitR)
    med = lambda v: float(np.median(v)) if len(v) else np.nan
    fo = np.array(foot_off) if foot_off else np.array([np.nan])
    return dict(n=n, duty=(cL + cR) / max(2 * n, 1), ds=ds / max(n, 1), land_s=land / max(n * DT, 1e-9),
                v=float(np.mean(vx)), ret=r_sum, ph=ph, st=st, plc=plc,
                thigh=med(thigh_at), shank=med(shank_at), flex=med(flex_at),
                fo_p50=med(fo), fo_max=float(np.nanmax(fo)))


print("=== 摆动末段膝半伸（W16 权重 × #2+膝注入模块；onset 从摆动第几帧起）===")
print(f"{'膝伺服配置':34s} {'存活':>5s} {'duty':>6s} {'ds':>6s} {'land/s':>7s} {'v':>6s} "
      f"{'触地大腿':>9s} {'触地小腿':>9s} {'膝屈':>6s} {'触地脚−CoM p50/max':>20s} {'净收入':>7s} {'place':>6s}")


def line(label, onset, target, kp):
    r = [roll(onset, target, kp, s) for s in SEEDS]
    med = lambda k, rr=r: float(np.median([x[k] for x in rr]))
    print(f"{label:34s} {med('n'):5.0f} {med('duty'):6.3f} {med('ds'):6.3f} {med('land_s'):7.2f} "
          f"{med('v'):6.3f} {med('thigh'):+9.1f} {med('shank'):+9.1f} {med('flex'):6.1f} "
          f"  {med('fo_p50'):+6.3f}/{med('fo_max'):+6.3f} {med('ret'):7.0f} {med('plc'):6.0f}")
    return med


base = line("off（不变性自检）", 0, 0.0, 0.0)
for onset in (7, 9, 11):
    for target, kp in ((-15.0, 60.0), (-25.0, 60.0), (-25.0, 25.0), (-35.0, 25.0)):
        med = line(f"onset={onset} tgt={target:+.0f}° kp={kp:.0f}", onset, target, kp)
        ok = med("n") >= 1200 and med("fo_max") > base("fo_max") and med("ret") >= 0.7 * base("ret")
        print(f"      ⇒ 判据：存活{med('n'):.0f} max{med('fo_max'):+.3f}(基线{base('fo_max'):+.3f}) "
              f"净{med('ret'):.0f}(基线{base('ret'):.0f})  ⇒ {'**PASS**' if ok else 'fail'}")
print("\n口径：向量式，正=该节在前方 ⇒ 触地小腿 −5.5 = 脚在膝后 5.5°；膝屈=大腿与小腿向量夹角。")
print("人 IC：大腿 +28.8、小腿 +3.4、膝屈 25.4、脚在 CoM 前 +0.233 m。")
