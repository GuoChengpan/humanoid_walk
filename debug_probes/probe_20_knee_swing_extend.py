"""⑰ 决策前第 7 件测量：摆动末段驱动**膝伸**（knee_pre），看小腿能不能在触地前甩直。

诊断来自上一轮（probe_18f 补的放置列 + 1499 帧姿态分布）：
  · 大腿世界角 −21.4/−15.3°（前摆 21°/15°，与人 IC +28.8° 同量级）⇒ 髋幅度不是瓶颈
  · 小腿世界角 **+11.9/+21.1°（向后甩）**，人 IC 是 +3.4°（略前）⇒ **脚被甩在膝盖后面**
  · 把腰按直到中立并不能救放置（触地脚−CoM max 反而 −0.018）⇒ 腰也不是瓶颈
  · 而膝在奖励里**一个项都没有**，模板也只驱动髋
所以这是唯一直接对着病灶的机构改动：摆动后段给膝一个伸的力矩。

它与已被否证的"落地后保持屈髋"的区别要说清：那个是落地后继续把大腿往前带（60 帧内摔死），
这个是**落地前把小腿甩直**，作用在另一个关节、另一个相位。

两件事必须先做：
  S1 符号标定：humanoid.xml 的膝是 **motor**（gear 200，ctrlrange ±0.4 ⇒ ±80 N·m），
     不是位置伺服 ⇒ "往伸的方向"必须实测，不能靠 range="−160 −2" 猜。
  S2 不变性自检：onset=0（关闭）必须逐指标复现 probe_19b 的「#2 更久者驱 k=1.0」行
     （净 1380 / phase 1000 / stride 393 / reach 119 / place 90 / 触地 max +0.065 / 反相 0.690）。
     对不上就说明补丁越界，整表作废。

判据（观测前钉死，人参照 IC 小腿 +3.4°、膝屈 25.4°）：
  PASS = 存活 ≥ 800 帧 且 触地小腿世界角比基线向"竖直/略前"移动 ≥ 5°
         且 触地脚−CoM max 比基线（+0.065）更大 且 净收入不低于基线的 70%
  只满足前两条但净收入崩 ⇒ 说明它把别的通道打掉了，需要看是哪一条。
"""
import importlib.util
import os
import re
import sys
import gc
import warnings

import numpy as np
import gymnasium as gym
import mujoco
from stable_baselines3 import SAC

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")

# ---------------- S1 符号标定 ----------------
ref = gym.make("Humanoid-v4")
rm, rd = ref.unwrapped.model, ref.unwrapped.data
KNEE_ADR = rm.jnt_qposadr[rm.joint("left_knee").id]
KNEE_CTRL = [u for u in range(rm.nu) if rm.joint(rm.actuator_trnid[u][0]).name == "left_knee"][0]
GEAR = rm.actuator_gear[KNEE_CTRL][0]


def flexion():
    """解剖学膝屈 = 大腿向量与小腿向量的夹角（memory 里定过：不是关节坐标）。"""
    th = rd.xpos[rm.body("left_shin").id] - rd.xpos[rm.body("left_thigh").id]
    sh = rd.xpos[rm.body("left_foot").id] - rd.xpos[rm.body("left_shin").id]
    c = np.dot(th, sh) / max(np.linalg.norm(th) * np.linalg.norm(sh), 1e-9)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


print("=== S1 膝 ctrl 符号标定（悬空、其余关节为 0，积分 30 步）===")
for sgn in (+1, -1):
    rd.qpos[:] = 0.0
    rd.qpos[2] = 1.4
    rd.qpos[3:7] = [1, 0, 0, 0]
    rd.qpos[KNEE_ADR] = np.radians(-30.0)          # 先弯 30°
    rd.qvel[:] = 0.0
    rd.ctrl[:] = 0.0
    rd.ctrl[KNEE_CTRL] = sgn * 0.15
    mujoco.mj_forward(rm, rd)
    f0 = flexion()
    for _ in range(30):
        mujoco.mj_step(rm, rd)
    f1 = flexion()
    print(f"  ctrl={sgn*0.15:+.2f} (τ={sgn*0.15*GEAR:+.0f} N·m)：膝屈 {f0:.1f}° → {f1:.1f}°"
          f"  ⇒ {'**伸**' if f1 < f0 - 0.5 else ('**屈**' if f1 > f0 + 0.5 else '无反应')}")
EXT_SIGN = None
for sgn in (+1, -1):
    rd.qpos[:] = 0.0; rd.qpos[2] = 1.4; rd.qpos[3:7] = [1, 0, 0, 0]
    rd.qpos[KNEE_ADR] = np.radians(-30.0); rd.qvel[:] = 0.0; rd.ctrl[:] = 0.0
    rd.ctrl[KNEE_CTRL] = sgn * 0.15
    mujoco.mj_forward(rm, rd); f0 = flexion()
    for _ in range(30):
        mujoco.mj_step(rm, rd)
    if flexion() < f0 - 0.5:
        EXT_SIGN = sgn
        break
assert EXT_SIGN is not None, "两个方向都没让膝伸，说明标定姿势不对（可能被限位/被体重卡住）"
print(f"  ⇒ 伸膝方向 = ctrl {EXT_SIGN:+.2f}，下面用 |c| ∈ {{0.10, 0.20}} × {EXT_SIGN}")

# ---------------- 打补丁：摆动后段驱动膝伸 ----------------
SRC = "train_humanoid_sac.py"
txt = open(SRC, encoding="utf-8").read()
PAT = re.compile(r'[ ]{12}want = \{"left": \(not _hitL\)[\s\S]*?self\.osc_double_air \+= 1\n')
m = PAT.search(txt)
assert m, "want+双腾空仲裁整段没匹配上"
# 紧跟在 osc_double_air 行后面就是 `act_flag = []`，直接用 m.end() 切片，
# 不要用 index('act_flag = []') —— 那会把这一行的 12 个前导空格一起吃掉，产生 IndentationError
INS = ('            _kon = getattr(self, "knee_onset", 0)\n'
       '            _kc = getattr(self, "knee_ctrl", 0.0)\n'
       '            if _kon:\n'
       '                for _s, _h in (("left", _hitL), ("right", _hitR)):\n'
       '                    if (not _h) and self._osc_air[_s] >= _kon:\n'
       '                        action[self.knee_aidx[_s]] = _kc      # 模板接管：摆动后段伸膝\n')
# 先换成 #2 的仲裁（腾空更久者驱动，平手不驱），再在 act_flag 之前插入膝驱动
src2 = txt[:m.start()] + '''            want = {"left": (not _hitL) and 1 <= self._osc_air["left"] <= self.osc_wmax,
                    "right": (not _hitR) and 1 <= self._osc_air["right"] <= self.osc_wmax}
            if want["left"] and want["right"]:
                _dl, _dr = self._osc_air["left"], self._osc_air["right"]
                want["left"], want["right"] = (_dl > _dr), (_dr > _dl)
                self.osc_double_air += 1
''' + INS + txt[m.end():]
assert "phase < 0.5" not in src2.split("act_flag = []")[0].split("want = {")[1], "旧仲裁没替换干净"
assert src2.count('action[self.knee_aidx[_s]] = _kc') == 1
VPATH = "/tmp/train_b20_knee.py"
open(VPATH, "w", encoding="utf-8").write(src2)
spec = importlib.util.spec_from_file_location("b20", VPATH)
W = importlib.util.module_from_spec(spec)
sys.modules["b20"] = W
spec.loader.exec_module(W)

MODEL = os.environ.get("PROBE_MODEL",
                       "best_model/tmp_model_data/SAC_19/19_5_厥着腚高速原地踏步/best_model.zip")
SEEDS = tuple(int(x) for x in os.environ.get("S18_SEEDS", "123,2026,7,31,55,77,91,101").split(","))
FR, DT = 1500, 0.015
model = SAC.load(MODEL)
del model.replay_buffer
gc.collect()


def roll(onset, c, seed):
    raw = gym.make("Humanoid-v4", max_episode_steps=FR)
    W.patch_mirror_symmetry(raw.unwrapped.model)
    w = W.WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=True)
    w.osc_k_start, w.osc_A = 1.0, 0.15
    w.knee_onset = onset
    w.knee_ctrl = c
    mm = w.unwrapped.model
    w.knee_aidx = {s: [u for u in range(mm.nu)
                       if mm.joint(mm.actuator_trnid[u][0]).name == f"{s}_knee"][0] for s in ("left", "right")}
    w.knee_qadr = {s: mm.jnt_qposadr[mm.joint(f"{s}_knee").id] for s in ("left", "right")}
    w.knee_dadr = {s: mm.jnt_dofadr[mm.joint(f"{s}_knee").id] for s in ("left", "right")}
    w.knee_gear = {s: mm.actuator_gear[w.knee_aidx[s]][0] for s in ("left", "right")}
    w.knee_kp, w.knee_kd = 120.0, 15.0
    w.knee_target = np.radians(-5.0)      # qpos 负=屈 ⇒ −5° ≈ 几乎伸直
    w.reset(seed=seed)
    m, d = w.unwrapped.model, w.unwrapped.data
    land = 0
    shank_at, thigh_at, footoff_at, flex_at = [], [], [], []
    jl, jr, sup, vx = [], [], [], []
    r_sum = ph = st = rc = plc = 0.0
    cL = cR = ds = fl = n = 0
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
        for s_, h_ in (("left", hitL), ("right", hitR)):
            if h_:
                sup.append(d.body(f"{s_}_foot").xpos[0] - com)
                plc_v = max(0.0, min(1.0, (d.body(f"{s_}_foot").xpos[0] - com + 0.06) / 0.26))
            else:
                plc_v = 0.0
            plc += plc_v
            if (not prev[0 if s_ == "left" else 1]) and h_:      # 本帧这条腿刚落地
                th = d.xpos[m.body(f"{s_}_shin").id] - d.xpos[m.body(f"{s_}_thigh").id]
                sk = d.xpos[m.body(f"{s_}_foot").id] - d.xpos[m.body(f"{s_}_shin").id]
                thigh_at.append(np.degrees(np.arctan2(th[0], -th[2])))
                shank_at.append(np.degrees(np.arctan2(sk[0], -sk[2])))
                footoff_at.append(d.body(f"{s_}_foot").xpos[0] - com)
                cs = np.dot(th, sk) / max(np.linalg.norm(th) * np.linalg.norm(sk), 1e-9)
                flex_at.append(np.degrees(np.arccos(np.clip(cs, -1, 1))))
                land += 1
        jl.append(np.degrees(d.qpos[m.jnt_qposadr[m.joint("left_hip_y").id]]))
        jr.append(np.degrees(d.qpos[m.jnt_qposadr[m.joint("right_hip_y").id]]))
        vx.append(d.qvel[0])
        prev = (hitL, hitR)
    anti = [0.5 * (1 - np.corrcoef(np.array(jl[i - 50:i]) - np.mean(jl[i - 50:i]),
                                   np.array(jr[i - 50:i]) - np.mean(jr[i - 50:i]))[0, 1]) for i in range(50, n)]
    fo = np.array(footoff_at) if footoff_at else np.array([np.nan])
    return dict(n=n, land=land, duty=(cL + cR) / max(2 * n, 1), ds=ds / max(n, 1),
                flight=fl / max(n, 1), v=float(np.mean(vx)), land_s=land / max(n * DT, 1e-9),
                ret=r_sum, ph=ph, st=st, rc=rc, plc=plc,
                anti=float(np.mean(anti)) if anti else np.nan,
                shank=float(np.median(shank_at)) if shank_at else np.nan,     # + 为后甩
                thigh=float(np.median(thigh_at)) if thigh_at else np.nan,
                flex=float(np.median(flex_at)) if flex_at else np.nan,
                fo_p50=float(np.nanmedian(fo)), fo_max=float(np.nanmax(fo)),
                sup_max=float(np.max(sup)) if sup else np.nan)


CFGS = [("onset=off（不变性自检）", 0, 0.0),
        ("onset=4  c=%+0.2f" % (EXT_SIGN * 0.10), 4, EXT_SIGN * 0.10),
        ("onset=6  c=%+0.2f" % (EXT_SIGN * 0.10), 6, EXT_SIGN * 0.10),
        ("onset=9  c=%+0.2f" % (EXT_SIGN * 0.10), 9, EXT_SIGN * 0.10),
        ("onset=6  c=%+0.2f" % (EXT_SIGN * 0.20), 6, EXT_SIGN * 0.20),
        ("onset=9  c=%+0.2f" % (EXT_SIGN * 0.20), 9, EXT_SIGN * 0.20)]
res = {}
print(f"\n=== S2 扫描（伸膝方向 ctrl={EXT_SIGN:+.2f}；onset=摆动第几帧开始伸）===")
print(f"{'配置':30s} {'存活':>5s} {'duty':>6s} {'ds':>6s} {'land/s':>7s} {'v':>6s} "
      f"{'触地大腿':>9s} {'触地小腿':>9s} {'触地膝屈':>9s} {'触地脚−CoM p50/max':>20s}")
for label, onset, c in CFGS:
    r = [roll(onset, c, s) for s in SEEDS]
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
print("\n人参照：IC 大腿 +28.8°前、小腿 +3.4°前、膝屈 25.4°、脚在 CoM 前 +0.233 m。"
      "口径（向量式，正=该节在前方）：触地大腿 +13 = 大腿前倾13°；**触地小腿 −5.5 = 脚在膝盖后面 5.5°（后甩）**。")
