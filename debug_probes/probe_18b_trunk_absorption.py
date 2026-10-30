"""⑰ 决策前的第 2 件测量：躯干把髋关节的摆动"吃掉"了多少，以及这是不是平衡驱动的。

背景（都已实测）：⑯ 末段 det 评估里
  · `ev_hip_at_land` 左/右 37.2° / 33.5°（**pelvis 相对**的 hip_y 关节角，而 phase_reward 就吃它）
  · 世界系矢状大腿角只有 14.9° / 7.1°（人 IC 实测 +28.8°）
  · `abdomen_y` +25~26°，接近 ±30° 限位
⇒ 关节摆得很开、脚却没往前走，中间差的量很可能被躯干（自由关节俯仰 + 腰）吸收了。
这决定 ⑰ 的两条候选哪条成立：把 phase_reward 换成世界系观测量，还是给躯干倾角加税。
但"加税"有个前提要先验：如果躯干俯仰是**平衡需要**（把 CoM 压在支撑脚上），
那它和 CoM−支撑脚误差应当同步；如果只是套利，它应当是个近似恒定的偏置。

控制项（按 memory 的规矩，新探针必须先复现一个已知读数）：
  `ev_hip_at_land`、`ev_reach_*` 这两列 jsonl 里有现成的中位数，探针用同一批 det rollout
  复现它们 ⇒ 复现不上就是取数下标/口径错了，下面的表一律不读。
"""
import importlib.util
import json
import os
import sys
import gc

import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")

SRC = "train_humanoid_sac.py"
spec = importlib.util.spec_from_file_location("b18c_main", SRC)
W = importlib.util.module_from_spec(spec)
sys.modules["b18c_main"] = W
spec.loader.exec_module(W)

MODEL = os.environ.get("PROBE_MODEL", "best_model/best_model.zip")
SEEDS = tuple(int(x) for x in os.environ.get("S18_SEEDS", "123,2026,7,31,55,77,91,101").split(","))
FR = int(os.environ.get("S18_FRAMES", "1500"))

model = SAC.load(MODEL)
del model.replay_buffer
gc.collect()


def roll(seed):
    raw = gym.make("Humanoid-v4", max_episode_steps=FR)
    W.patch_mirror_symmetry(raw.unwrapped.model)
    w = W.WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=True)
    w.reset(seed=seed)
    m, d = w.unwrapped.model, w.unwrapped.data
    q = lambda name: float(d.qpos[m.jnt_qposadr[m.joint(name).id]])
    ix = {s: dict(hip=m.jnt_qposadr[m.joint(f"{s}_hip_y").id],
                  knee=m.jnt_qposadr[m.joint(f"{s}_knee").id],
                  thigh=m.body(f"{s}_thigh").id, leg=m.body(f"{s}_shin").id,
                  foot=m.body(f"{s}_foot").id) for s in ("left", "right")}
    has = lambda n: (n in [m.joint(i).name for i in range(m.njnt)])
    abd = [n for n in ("abdomen_y", "abdomen_x", "abdomen_z") if has(n)]
    ne = [n for n in ("neck_y", "neck_x", "neck_z") if has(n)]
    rows = []
    for _ in range(FR):
        a, _ = model.predict(w.unwrapped._get_obs(), deterministic=True)
        _, _, ter, tru, info = w.step(a)
        if ter or tru:
            break
        rec = dict(t=w.total_steps, z=d.qpos[2], x=d.qpos[0], vx=d.qvel[0])
        # 世界系矢状大腿角：thigh→leg 向量，dx>0 = 膝盖在髋的前方 = 大腿前倾
        for s, k in ix.items():
            tp, lp = d.xpos[k["thigh"]], d.xpos[k["leg"]]
            v = lp - tp
            rec[f"world_{s}"] = float(np.degrees(np.arctan2(v[0], -v[2])))
            rec[f"joint_{s}"] = float(np.degrees(d.qpos[k["hip"]]))
            rec[f"reach_{s}"] = float(d.xpos[k["foot"]][0] - tp[0])
            rec[f"hipq_{s}"] = float(-np.degrees(d.qpos[k["hip"]]))   # jsonl ev_hip_at_land 同式
        rec["abd"] = {n: float(np.degrees(d.qpos[m.jnt_qposadr[m.joint(n).id]])) for n in abd}
        rec["neck"] = {n: float(np.degrees(d.qpos[m.jnt_qposadr[m.joint(n).id]])) for n in ne}
        # 躯干（自由关节）俯仰：由 root 四元数取 pitch
        qw, qx, qy, qz = d.qpos[3:7]
        rec["root_pitch"] = float(np.degrees(np.arcsin(np.clip(2 * (qw * qy - qx * qz), -1, 1))))
        hitL, hitR, _, _ = w._foot_contact_state(m, d)
        rec["hit"] = (int(hitL), int(hitR))
        fx = [d.xpos[ix[s]["foot"]][0] for s, h in (("left", hitL), ("right", hitR)) if h]
        rec["sup_err"] = (d.qpos[0] - float(np.mean(fx))) if fx else np.nan
        rows.append(rec)
    return rows


allr = [roll(s) for s in SEEDS]
rows = [r for rs in allr for r in rs if len(rs) > 200]
if not rows:
    sys.exit("全部 rollout 都在 200 帧内结束，无法测量")
print(f"{len(allr)} 个 seed，共 {sum(len(r) for r in allr)} 帧")


def g(key):
    return np.array([r[key] for r in rows], dtype=float)


hit = np.array([r["hit"] for r in rows])
print("\n=== 控制 1：复现 jsonl 已知读数（同一权重、同一 det 口径）===")
js = [json.loads(l) for l in open("eval_logs/eval_gait.jsonl", encoding="utf-8") if l.strip()]
js = [r for r in js if r.get("b16")][-270:]
jm = lambda k: float(np.nanmedian([r.get(k, np.nan) for r in js]))
# 探针的 hipq/reach 是逐帧的，jsonl 的是**落地事件**均值 ⇒ 只要求同量级与同号，不要求逐位相等
print(f"  hip_y 关节角(度)  探针 L {np.median(g('hipq_left')):+6.1f} R {np.median(g('hipq_right')):+6.1f}"
      f"   |  jsonl ev_hip_at_land L {jm('ev_hip_at_land_L_mean'):+.1f} R {jm('ev_hip_at_land_R_mean'):+.1f}")
print(f"  reach(米)        探针 L {np.median(g('reach_left')):+6.3f} R {np.median(g('reach_right')):+6.3f}"
      f"   |  jsonl ev_reach  L {jm('ev_reach_L_mean'):+.3f} R {jm('ev_reach_R_mean'):+.3f}")
print(f"  接触占比         探针 L {hit[:,0].mean():6.3f} R {hit[:,1].mean():6.3f}  "
      f"⇒ duty_sum={hit.sum(1).mean():.3f}（人 1.146；ds>0 的**充要**条件是每腿 duty>0.5）")

print("\n=== 控制 2：躯干各段的摆幅与均值（度）===")
for k in ["root_pitch"] + [f"abd.{n}" for n in rows[0]["abd"]] + [f"neck.{n}" for n in rows[0]["neck"] if rows[0]["neck"]]:
    pre, _, name = k.partition(".")
    xs = np.array([r[pre][name] if name else r[k] for r in rows], dtype=float)
    print(f"  {k:16s} 均值 {xs.mean():+7.1f}  p10/p90 {np.percentile(xs,10):+7.1f}/{np.percentile(xs,90):+7.1f}"
          f"  峰值摆幅 {xs.max()-xs.min():6.1f}")

print("\n=== A. 吸收率：关节摆幅 vs 世界系大腿摆幅（同一腿、同一帧集）===")
print("  链：torso(自由关节 pitch) → abdomen_y → pelvis(焊在 lwaist 上) → hip_y → 大腿")
abd_y = np.array([r["abd"]["abdomen_y"] for r in rows], dtype=float)
rp = g("root_pitch")
for s in ("left", "right"):
    j, wd = g(f"joint_{s}"), g(f"world_{s}")
    amp_j = np.percentile(j, 90) - np.percentile(j, 10)
    amp_w = np.percentile(wd, 90) - np.percentile(wd, 10)
    b = np.polyfit(j, wd, 1)
    chain = rp + abd_y + j
    print(f"  {s:5s} 关节 p10-p90 {amp_j:5.1f}° → 世界 {amp_w:5.1f}° ⇒ **传递率 {amp_w/max(amp_j,1e-9):.2f}**"
          f"   斜率 {b[0]:+.2f}  R={np.corrcoef(j, wd)[0,1]:+.2f}")
    print(f"        三段和(root+abdomen_y+hip_y) 与实测世界大腿角的残差 p50 |Δ|="
          f"{np.median(np.abs(chain - wd)):.1f}°  corr={np.corrcoef(chain, wd)[0,1]:+.2f}"
          f"   三段自身 p50：root {np.median(rp):+.1f}° abdomen_y {np.median(abd_y):+.1f}° hip_y {np.median(j):+.1f}°")
print(f"  人参照：IC 世界大腿 +28.8°（p10 25.4），膝屈 25.4°，小腿 +3.4°")

print("\n=== B. 躯干俯仰是平衡驱动还是套利偏置？（对 CoM−支撑脚 误差回归）===")
se = g("sup_err")
ok = ~np.isnan(se)
for k in ["root_pitch", "abd.abdomen_y", "joint_left", "joint_right"]:
    pre, _, name = k.partition(".")
    v = (np.array([r[pre][name] for r in rows], dtype=float) if name
         else np.array([r[k] for r in rows], dtype=float))
    c = np.corrcoef(se[ok], v[ok])[0, 1]
    sl = np.polyfit(se[ok], v[ok], 1)[0]
    print(f"  {k:16s} 与支撑误差 corr={c:+.2f} 斜率={sl:+.2f}°/m"
          f"   误差本身 p10/p50/p90 = {np.percentile(se[ok],10):+.3f}/{np.median(se[ok]):+.3f}/"
          f"{np.percentile(se[ok],90):+.3f} m")
print(f"  前进速度 mean {g('vx').mean():+.3f} m/s   躯干高度 z p10/p50/p90 = "
      f"{np.percentile(g('z'),10):.3f}/{np.median(g('z')):.3f}/{np.percentile(g('z'),90):.3f} m")

print("\n=== C. 判据（观测前钉死）===")
print("  B 类结论：|corr(root_pitch, sup_err)| ≥ 0.5 且斜率符号 = 误差为正时躯干前倾（撑住 CoM）")
print("            ⇒ 你说对，撅腚是平衡需要，不能上税，只能改掉让它没必要的驱动方式。")
print("  A 类结论：corr < 0.3 且 root_pitch/abdomen_y 均值离 0 很远而摆幅小")
print("            ⇒ 是恒定偏置（套利/姿势），phase_reward 必须换世界系口径。")
