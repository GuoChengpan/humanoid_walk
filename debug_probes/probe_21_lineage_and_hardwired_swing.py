"""⑰ 决策前第 8 件测量：(A) ⑮ 与 ⑯ 两个权重在"⑰ 接口"下的盆地质量；(B) 把摆动规律写死能不能救放置。

A 的动机（用户的判断：⑮ 的局部陷阱可能没那么强）：光比"谁训得久"没有用，要比的是**在同一个
⑰ 接口下，谁的起点更接近放置**。所以四格都跑（权重 × 接口）：
  W15=19_4/best_model.zip（md5 53e87b25…，我 memory 里记成 5e87b25a 是截错了，这里更正）
  W16=19_5/best_model.zip（md5 19854b7e…）
  S15=⑮ 源快照（T1 时钟驱动 + ⑮ 价格）——"它当年到底学成什么样"
  S17=当前文件 + #2 仲裁（T2 触地驱动 + ⑮ 价格）——"续训睁眼看到的第一帧"
     注意：S15/S17 的**价格完全相同**（⑮ 的那套），差别只在驱动定律 ⇒ 同接口下的两权重差才是纯权重差。

B 的动机（用户：让它自己找好像找不出来，一直在套利）：probe_20b 里
  onset=9（只在摆动最后 ~3 帧伸膝）交出了**全场最高的触地脚−CoM max +0.104**，却因伺服太硬
  （kp=120 拉到 −5°）只活 282 帧。这不是"伸膝不成立"，是"伸太早 + 拉太直"。
  所以扫第二个网格：更晚介入(onset 8/10/12) × 只半伸(target −15/−25/−35) × 更软的增益。
  大腿那一路已经等于写死了：k=1.0 时模板把髋按在 −A=−0.15 直到触地，正是人"IC 处于大腿前摆峰值"的规律。
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

W15 = "best_model/tmp_model_data/SAC_19/19_4_调权重_结果还是原地踏步速度慢了/best_model.zip"
W16 = "best_model/tmp_model_data/SAC_19/19_5_厥着腚高速原地踏步/best_model.zip"
S15 = "best_model/tmp_model_data/SAC_19/19_4_调权重_结果还是原地踏步速度慢了/train_humanoid_sac.py"
CUR = "train_humanoid_sac.py"
FR, DT = 1500, 0.015
SEEDS = tuple(int(x) for x in os.environ.get("S18_SEEDS", "123,2026,7,31,55,77,91,101").split(","))


def build_s17():
    """当前文件 + #2 仲裁（腾空更久者驱动，平手不驱）。价格一行不动。"""
    txt = open(CUR, encoding="utf-8").read()
    PAT = re.compile(r'[ ]{12}want = \{"left": \(not _hitL\)[\s\S]*?self\.osc_double_air \+= 1\n')
    m = PAT.search(txt)
    assert m, "want+双腾空仲裁整段没匹配上"
    NEW = ('            want = {"left": (not _hitL) and 1 <= self._osc_air["left"] <= self.osc_wmax,\n'
           '                    "right": (not _hitR) and 1 <= self._osc_air["right"] <= self.osc_wmax}\n'
           '            if want["left"] and want["right"]:\n'
           '                _dl, _dr = self._osc_air["left"], self._osc_air["right"]\n'
           '                want["left"], want["right"] = (_dl > _dr), (_dr > _dl)\n'
           '                self.osc_double_air += 1\n')
    out = txt[:m.start()] + NEW + txt[m.end():]
    assert "原来这里用" not in out, "⑯·a 的仲裁块还在 span 里"
    # 注意 :841 的 `self.gait_phase < 0.5` 是 position_swap_reward 的节奏目标（**绝对时钟**），
    # 与本补丁无关，不能当成残留。它是全项目还剩的一处时钟依赖付款通道（实测 58 分/集）。
    p = "/tmp/train_b21_s17.py"
    open(p, "w", encoding="utf-8").write(out)
    return p


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


S17 = build_s17()
MODS = {"S15 ⑮源(T1时钟+⑮价格)": load(S15, "b21_s15"), "S17 ⑰接口(#2仲裁+⑮价格)": load(S17, "b21_s17")}
MODELS = {}


def get_model(p):
    if p not in MODELS:
        mo = SAC.load(p)
        del mo.replay_buffer
        gc.collect()
        MODELS[p] = mo
    return MODELS[p]


def roll(Wm, mp, onset=0, target=-5.0, kp=0.0, seed=0):
    mo = get_model(mp)
    raw = gym.make("Humanoid-v4", max_episode_steps=FR)
    Wm.patch_mirror_symmetry(raw.unwrapped.model)
    w = Wm.WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=True)
    m = w.unwrapped.model
    w.knee_onset = onset
    if onset:
        w.knee_aidx = {s: [u for u in range(m.nu)
                           if m.joint(m.actuator_trnid[u][0]).name == f"{s}_knee"][0] for s in ("left", "right")}

        def fn(win, s, _t=target, _kp=kp):
            d = win.env.unwrapped.data
            qa = m.jnt_qposadr[m.joint(f"{s}_knee").id]
            dv = m.jnt_dofadr[m.joint(f"{s}_knee").id]
            tau = _kp * (np.radians(_t) - d.qpos[qa]) - 8.0 * d.qvel[dv]
            return float(np.clip(tau / m.actuator_gear[win.knee_aidx[s]][0], -0.4, 0.4))
        w.knee_fn = fn
    w.reset(seed=seed)
    d = w.unwrapped.data
    abd_id = m.jnt_qposadr[m.joint("abdomen_y").id]
    land = n = cL = cR = ds = fl = 0
    thigh_at, shank_at, flex_at, foot_off = [], [], [], []
    jl, jr, vx, abd = [], [], [], []
    r_sum = ph = st = rc = plc = 0.0
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
        rc += float(info.get("reach_reward_gated", 0.0) or 0.0)
        hitL, hitR, _, _ = w._foot_contact_state(m, d)
        ds += int(hitL and hitR); fl += int((not hitL) and (not hitR))
        cL += int(hitL); cR += int(hitR)
        com = d.qpos[0]
        abd.append(np.degrees(d.qpos[abd_id]))
        for s_, h_, pk in (("left", hitL, prev[0]), ("right", hitR, prev[1])):
            fx = d.body(f"{s_}_foot").xpos[0]
            plc += (max(0.0, min(1.0, (fx - com + 0.06) / 0.26)) if h_ else 0.0)
            if h_ and not pk:
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
    med = lambda v: float(np.median(v)) if len(v) else np.nan
    fo = np.array(foot_off) if foot_off else np.array([np.nan])
    ab = np.array(abd) if abd else np.array([np.nan])
    return dict(n=n, duty=(cL + cR) / max(2 * n, 1), ds=ds / max(n, 1), flight=fl / max(n, 1),
                land_s=land / max(n * DT, 1e-9), v=float(np.mean(vx)), ret=r_sum, ph=ph, st=st,
                rc=rc, plc=plc, anti=float(np.mean(anti)) if anti else np.nan,
                thigh=med(thigh_at), shank=med(shank_at), flex=med(flex_at),
                fo_p50=med(fo), fo_max=float(np.nanmax(fo)),
                abd_at_limit=float(np.mean(np.abs(np.abs(ab) - 30.0) < 2.0)))


HDR = (f"{'配置':40s} {'存活':>5s} {'duty':>6s} {'ds':>6s} {'land/s':>7s} {'v':>6s} "
       f"{'触地大腿':>9s} {'触地小腿':>9s} {'膝屈':>6s} {'触地脚−CoM p50/max':>20s} {'腰顶限位%':>9s}")


def show(label, runs):
    med = lambda key: float(np.median([x[key] for x in runs]))
    print(f"{label:40s} {med('n'):5.0f} {med('duty'):6.3f} {med('ds'):6.3f} {med('land_s'):7.2f} "
          f"{med('v'):6.3f} {med('thigh'):+9.1f} {med('shank'):+9.1f} {med('flex'):6.1f} "
          f"  {med('fo_p50'):+6.3f}/{med('fo_max'):+6.3f} {med('abd_at_limit')*100:9.1f}")
    return med


print("=== A. 权重 × 接口（价格两边都是 ⑮ 那套，所以同接口下的两行差 = 纯权重差）===")
print(HDR)
A = {}
for mlabel, Wm in MODS.items():
    for wlabel, wp in (("W15 ⑮权重", W15), ("W16 ⑯权重", W16)):
        A[(mlabel, wlabel)] = show(f"{mlabel} × {wlabel}", [roll(Wm, wp, seed=s) for s in SEEDS])
print(f"\n{'配置':40s} {'净收入':>7s} {'phase':>7s} {'stride':>6s} {'reach':>6s} {'place梯子':>9s} {'髋反相':>7s}")
for (mlabel, wlabel), med in A.items():
    print(f"{mlabel + ' × ' + wlabel:40s} {med('ret'):7.0f} {med('ph'):7.0f} {med('st'):6.0f} "
          f"{med('rc'):6.0f} {med('plc'):9.0f} {med('anti'):7.3f}")

print("\n=== B. 把摆动末段的膝「半伸」写死：晚介入 × 软增益 × 只伸到 −15/−25/−35 ===")
print(HDR.replace("配置", "膝伺服(W16 × S17)"))
Wm17 = MODS["S17 ⑰接口(#2仲裁+⑮价格)"]
BASE = [roll(Wm17, W16, seed=s) for s in SEEDS]
show("off（基线 = W16×S17 上面那行）", BASE)
bm = float(np.median([x["fo_max"] for x in BASE]))
for onset in (8, 10, 12):
    for target in (-15.0, -25.0, -35.0):
        rr = [roll(Wm17, W16, onset=onset, target=target, kp=60.0, seed=s) for s in SEEDS]
        med = show(f"onset={onset} tgt={target:+.0f}° kp=60", rr)
        ok = med("n") >= 1000 and med("fo_max") > bm and med("ret") >= 0.7 * float(np.median([x["ret"] for x in BASE]))
        print(f"      ⇒ 存活{med('n'):.0f} max{med('fo_max'):+.3f}(基线{bm:+.3f}) 净{med('ret'):.0f} "
              f"{'**PASS**' if ok else 'fail'}")
print("\n人参照：IC 大腿 +28.8°、小腿 +3.4°(本表口径为向量式，正=向前 ⇒ 人约 +3.4)、膝屈 25.4°、脚在 CoM 前 +0.233 m。")


print("\n=== C. from-scratch 可行性：早期策略（零动作 / 随机动作）在 #2 接口下能活多久 ===")
print("   ⑦ 的旧实测：k≥0.45 与 FULL 覆盖在 25-214 步内摔 ⇒ 若这里也活不到 300 帧，")
print("   『从头训 + k=1.0』就不能直接开，必须 k 课程（先 ≤0.35 再升）。")
rng = np.random.default_rng(7)


def survive(Wm, mode, k, A, seed):
    raw = gym.make("Humanoid-v4", max_episode_steps=FR)
    Wm.patch_mirror_symmetry(raw.unwrapped.model)
    w = Wm.WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=True)
    w.osc_k_start, w.osc_A = k, A
    w.reset(seed=seed)
    n = 0
    adv = 0.0
    for _ in range(FR):
        if mode == "zero":
            a = np.zeros(w.action_space.shape, dtype=np.float32)
        else:
            a = rng.uniform(-1, 1, w.action_space.shape).astype(np.float32)
        _, r, ter, tru, _ = w.step(a)
        if ter or tru:
            break
        n += 1
        adv += w.unwrapped.data.qpos[0]
    return n, adv / max(n, 1)


print(f"{'动作 × k':34s} {'k=1.00 存活':>12s} {'k=0.50 存活':>12s} {'k=0.25 存活':>12s}")
for mode in ("zero", "random"):
    row = []
    for k in (1.0, 0.5, 0.25):
        ns = [survive(MODS["S17 ⑰接口(#2仲裁+⑮价格)"], mode, k, 0.15, s)[0] for s in SEEDS[:6]]
        row.append(float(np.median(ns)))
    print(f"{mode + ' 动作':34s} {row[0]:12.0f} {row[1]:12.0f} {row[2]:12.0f}"
          f"   （6 seeds 中位；前进位移另计）")
print("   读法：三列都 <300 ⇒ from-scratch 带不动 k=1.0，⑰ 必须配 k 课程或改续训。")
