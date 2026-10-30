"""左腿机制链只读探针（批次C'标定 + 弹性判决）。

口径：接触与滑移全部走 WalkRewardWrapper._foot_contact_state（与训练同一份代码）。
不再用 body_z<0.08 —— 那个判据实为"脚球底离地<0.105m"，实测把 52% 的腾空算成了支撑。

按摆动事件粒度分别测左右腿：髋行程 / 摆动时长 / 屈髋驱动(旧奖励口径) / 踝相对躯干前向进展 /
clearance(球底真离地) / 末期伸膝 / 落点超前量 reach(相对躯干、相对同侧髋) /
支撑段真滑移(接触点切向速度积分；脚是球 geom 能滚，故踝位移≠滑行) / 单步真实前进 step_advance。

只读：不改训练进程；轨迹落盘 debug_probes/trace_944k.npz 供离线复算。
"""
import os
import sys
import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

# 模型路径不要硬编码快照名：磁盘上有 3 个同名不同 lineage 的 best_model_944000.zip，
# 而快照目录会被反复移动（本会话就被移动过一次，导致探针直接找不到文件）。
# 默认用当前 warm-start 文件；要测别的模型就 PROBE_MODEL=/path/xx.zip python3 ...
_DEFAULT_MODEL = "/home/guo/workspace/humanoid_walk/best_model/best_model.zip"
MODEL = os.environ.get("PROBE_MODEL", _DEFAULT_MODEL)
if not os.path.exists(MODEL):
    raise SystemExit(f"探针找不到模型: {MODEL}\n用 PROBE_MODEL=... 指定一个存在的 zip")
TRACE = "/home/guo/workspace/humanoid_walk/debug_probes/trace_probe.npz"
N_EP = 5
EP_HORIZON = 1500         # 必须与训练一致（gym.make 的 max_episode_steps）
MAX_STEPS = 8000
DT = 0.015                # frame_skip=5 × timestep=0.003，已实测核对


def build_env():
    def _f():
        raw = gym.make("Humanoid-v4", max_episode_steps=EP_HORIZON)
        patch_mirror_symmetry(raw.unwrapped.model)     # 与训练同一任务定义
        return WalkRewardWrapper(raw, swing_drive_weight=2.0)
    return DummyVecEnv([_f])


def load_model(vec):
    try:
        m = SAC.load(MODEL, env=vec, device="cpu")
        print("[load] SAC.load 成功")
        import gc
        m.replay_buffer = None      # 探针只用 policy，1.5M buffer 的虚拟分配不带走
        gc.collect()
        return m
    except Exception as e:
        print(f"[load] SAC.load 失败: {type(e).__name__}: {e}")
    import zipfile
    m = SAC("MlpPolicy", vec, policy_kwargs={"net_arch": [512, 256], "activation_fn": nn.Tanh},
            device="cpu", seed=0)
    with zipfile.ZipFile(MODEL) as z:
        sd = th.load(z.open("policy.pth"), map_location="cpu", weights_only=False)
    missing, unexpected = m.policy.load_state_dict(sd, strict=False)
    print(f"[load] 兜底权重: missing={len(missing)} unexpected={len(unexpected)}")
    return m


def run(vec, model):
    wrapper = vec.envs[0]
    raw = wrapper.unwrapped
    m, data = raw.model, raw.data
    joints = ["left_hip_y", "right_hip_y", "left_hip_x", "right_hip_x",
              "left_hip_z", "right_hip_z", "left_knee", "right_knee"]
    jid = {n: m.joint(n).id for n in joints}
    qadr = {k: int(m.jnt_qposadr[v]) for k, v in jid.items()}
    dofadr = {k: int(m.jnt_dofadr[v]) for k, v in jid.items()}
    fl, fr = m.body("left_foot").id, m.body("right_foot").id
    tl, tr_ = m.body("left_thigh").id, m.body("right_thigh").id
    gl = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == fl)
    gr = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == fr)
    rad = float(m.geom(gl).size[0])
    torso = m.body("torso").id

    recs, ep = [], 0
    obs = vec.reset()
    for _ in range(MAX_STEPS):
        act, _ = model.predict(obs, deterministic=True)
        obs, rew, done, info = vec.step(act)
        # 与训练用同一个函数：真接触 + 接触点切向滑移
        cL, cR, slipL, slipR = wrapper._foot_contact_state(m, data)
        recs.append(dict(
            ep=ep, t=len(recs), rew=float(rew[0]), bd=bool(done[0]),
            torso_x=float(data.xpos[torso][0]), torso_z=float(data.xpos[torso][2]),
            hipLx=float(data.xpos[tl][0]), hipRx=float(data.xpos[tr_][0]),
            lx=float(data.xpos[fl][0]), rx=float(data.xpos[fr][0]),
            clearL=float(data.geom_xpos[gl][2] - rad),
            clearR=float(data.geom_xpos[gr][2] - rad),
            hipL=float(data.qpos[qadr["left_hip_y"]]), hipR=float(data.qpos[qadr["right_hip_y"]]),
            kneeL=float(data.qpos[qadr["left_knee"]]), kneeR=float(data.qpos[qadr["right_knee"]]),
            hxL=float(data.qpos[qadr["left_hip_x"]]), hxR=float(data.qpos[qadr["right_hip_x"]]),
            hzL=float(data.qpos[qadr["left_hip_z"]]), hzR=float(data.qpos[qadr["right_hip_z"]]),
            vL=float(data.qvel[dofadr["left_hip_y"]]), vR=float(data.qvel[dofadr["right_hip_y"]]),
            cL=cL, cR=cR, slipL=slipL, slipR=slipR,
        ))
        if done[0]:
            ep += 1
            if ep >= N_EP:
                break
    keys = list(recs[0].keys())
    arr = {k: np.array([r[k] for r in recs],
                       dtype=np.int64 if k in ("ep", "t") else np.float64) for k in keys}
    np.savez(TRACE, **arr)
    print(f"[trace] {len(recs)} 步 / {ep} 集 落盘 -> {TRACE}")
    return recs


def swings(recs, side):
    """离地 -> 落地 的摆动段（只在同一集内切段）。"""
    c = [bool(r[f"c{side}"]) for r in recs]
    fx = np.array([r[{"L": "lx", "R": "rx"}[side]] for r in recs])
    tx = np.array([r["torso_x"] for r in recs])
    hpx = np.array([r[{"L": "hipLx", "R": "hipRx"}[side]] for r in recs])
    hip = np.array([r[{"L": "hipL", "R": "hipR"}[side]] for r in recs])
    knee = np.array([r[{"L": "kneeL", "R": "kneeR"}[side]] for r in recs])
    clr = np.array([r[{"L": "clearL", "R": "clearR"}[side]] for r in recs])
    hxt = np.array([r[{"L": "hxL", "R": "hxR"}[side]] for r in recs])
    hzt = np.array([r[{"L": "hzL", "R": "hzR"}[side]] for r in recs])
    out, i = [], 0
    while i < len(recs) - 2:
        if c[i] and not c[i + 1] and recs[i]["ep"] == recs[i + 1]["ep"]:
            j = i + 1
            while j < len(recs) - 1 and not c[j]:
                j += 1
            a, b = i + 1, j                        # [a,b) 空中，b = 落地帧
            if b - a >= 3 and recs[a]["ep"] == recs[b - 1]["ep"]:
                rel = fx[a:b] - tx[a:b]
                relh = fx[a:b] - hpx[a:b]
                n = b - a
                k3 = max(1, n // 3)
                pos = np.maximum(0.0, np.diff(rel))
                out.append(dict(
                    n_steps=n, dur=n * DT,
                    drive_old=float(np.maximum(0.0, -np.diff(hip[a:b])).sum() / n),
                    drive_new=float(pos.sum() / n),
                    hip_rom=float(np.degrees(hip[a:b].max() - hip[a:b].min())),
                    clearance=float(clr[a:b].min()),
                    reach=float(rel[-1]), reach_hip=float(relh[-1]),
                    knee_ext_last3=float(np.degrees(knee[b - 1] - knee[max(a, b - 1 - k3)])),
                    frac_last3=float(pos[-k3:].sum() / max(pos.sum(), 1e-9)),
                    hx_tax=float(np.clip(np.abs(hxt[a:b]) - 0.15, 0, None).mean() * 1.5),
                    hz_tax=float(np.clip(np.abs(hzt[a:b]) - 0.10, 0, None).mean() * 1.5),
                ))
            i = j
        else:
            i += 1
    return out


def stances(recs, side):
    """落地 -> 离地 的支撑段：真滑移积分 vs 踝位移（球脚能滚，踝位移不等于滑行）。"""
    c = [bool(r[f"c{side}"]) for r in recs]
    fx = np.array([r[{"L": "lx", "R": "rx"}[side]] for r in recs])
    tx = np.array([r["torso_x"] for r in recs])
    sl = np.array([r[{"L": "slipL", "R": "slipR"}[side]] for r in recs])
    out, i = [], 0
    while i < len(recs) - 2:
        if not c[i] and c[i + 1] and recs[i]["ep"] == recs[i + 1]["ep"]:
            j = i + 1
            while j < len(recs) - 1 and c[j]:
                j += 1
            a, b = i + 1, j
            if b - a >= 3 and recs[a]["ep"] == recs[b - 1]["ep"]:
                out.append(dict(n_steps=b - a,
                                slip=float((sl[a:b] * DT).sum()),
                                ankle_travel=float(abs(fx[b - 1] - fx[a])),
                                advance=float(tx[b - 1] - tx[a])))
            i = j
        else:
            i += 1
    return out


def q(v, p):
    return float(np.percentile(v, p)) if len(v) else float("nan")


def med(a):
    return float(np.median(a)) if len(a) else float("nan")


def main():
    vec = build_env()
    model = load_model(vec)
    recs = run(vec, model)
    recs = [r for r in recs if not r["bd"]]        # done 帧记的是 reset 后的状态，脏样本
    ep = np.array([r["ep"] for r in recs])
    tx = np.array([r["torso_x"] for r in recs])
    tz = np.array([r["torso_z"] for r in recs])
    print(f"\n采样 {len(recs)} 步 / {ep.max()+1} 集 (det, horizon={EP_HORIZON})")

    sp = (np.abs(np.diff(tx)) / DT)[ep[1:] == ep[:-1]]
    print(f"speed: mean={sp.mean():.3f} p50={np.median(sp):.3f} p90={q(sp,90):.3f} max={sp.max():.3f} "
          f"| >1.4={100*np.mean(sp>1.4):.0f}%  >1.6={100*np.mean(sp>1.6):.0f}%")
    print(f"torso_z: mean={tz.mean():.3f} min={tz.min():.3f} std={tz.std():.3f} | 每集步数: "
          + ",".join(str(int((ep == k).sum())) for k in range(int(ep.max()) + 1)))

    aL = np.array([bool(r["cL"]) for r in recs]); aR = np.array([bool(r["cR"]) for r in recs])
    print(f"\n相位(真接触): 左接触={aL.mean():.3f} 右接触={aR.mean():.3f} "
          f"双支撑={np.mean(aL & aR):.3f} 双腾空={np.mean(~aL & ~aR):.3f}")

    lx = np.array([r["lx"] for r in recs]); rx = np.array([r["rx"] for r in recs])
    hipLx = np.array([r["hipLx"] for r in recs]); hipRx = np.array([r["hipRx"] for r in recs])
    land = [i for i in range(1, len(recs))
            if ep[i] == ep[i - 1] and ((not aL[i - 1] and aL[i]) or (not aR[i - 1] and aR[i]))]
    if len(land) > 4:
        land = np.array(land)
        who = np.array(["L" if (not aL[i - 1] and aL[i]) else "R" for i in land])
        pair = np.where(ep[land][1:] == ep[land][:-1])[0]
        adv, dtt = np.diff(tx[land])[pair], (np.diff(land) * DT)[pair]
        w2 = who[1:][pair]
        print(f"\n落地 {len(land)} 次 | 间隔={np.mean(np.diff(land)):.1f}步={dtt.mean():.3f}s "
              f"| 步频={1/dtt.mean():.2f} 步/s ({60/dtt.mean():.0f} 步/分)")
        for s in "LR":
            k = w2 == s
            if k.sum() < 2:
                continue
            print(f"  到{s}腿落地: 单步真实前进={adv[k].mean():.3f}±{adv[k].std():.3f} m (n={k.sum()}) "
                  f"| 该步均速={adv[k].mean()/dtt[k].mean():.2f} m/s")
        print(f"  reach@落地 相对躯干: 左={(lx[land]-tx[land])[who=='L'].mean():+.3f} "
              f"右={(rx[land]-tx[land])[who=='R'].mean():+.3f} m")
        print(f"  reach@落地 相对同侧髋: 左={(lx[land]-hipLx[land])[who=='L'].mean():+.3f} "
              f"右={(rx[land]-hipRx[land])[who=='R'].mean():+.3f} m  (正=迈到髋前方)")
        print(f"  两脚间距(现 stride 口径): 左落地={(lx[land]-rx[land])[who=='L'].mean():+.3f} "
              f"右落地={(rx[land]-lx[land])[who=='R'].mean():+.3f} m")

    SW = {s: swings(recs, s) for s in "LR"}
    ST = {s: stances(recs, s) for s in "LR"}
    print("\n=== 摆动段（真接触口径）===")
    print("%-8s %5s %7s %9s %9s %7s %8s %8s %8s %9s %7s %7s %7s" % (
        "leg", "n", "dur_s", "drive_old", "drive_new", "rom°", "clear", "reach",
        "reach髋", "kneeExt°", "fracL3", "hx_tax", "hz_tax"))
    for s in "LR":
        if not SW[s]:
            print(f"{s}: 无摆动段")
            continue
        g = lambda k: np.array([w[k] for w in SW[s]])
        print("%-8s %5d %7.3f %9.4f %9.4f %7.1f %8.3f %8.3f %8.3f %9.1f %7.2f %7.3f %7.3f" % (
            s + " med", len(SW[s]), med(g("dur")), med(g("drive_old")), med(g("drive_new")),
            med(g("hip_rom")), med(g("clearance")), med(g("reach")), med(g("reach_hip")),
            med(g("knee_ext_last3")), med(g("frac_last3")), med(g("hx_tax")), med(g("hz_tax"))))
        print("%-8s %5s %7.3f %9.4f %9.4f %7.1f %8.3f %8.3f %8.3f %9.1f %7.2f %7.3f %7.3f" % (
            s + " p90", "", q(g("dur"), 90), q(g("drive_old"), 90), q(g("drive_new"), 90),
            q(g("hip_rom"), 90), q(g("clearance"), 10), q(g("reach"), 90), q(g("reach_hip"), 90),
            q(g("knee_ext_last3"), 90), q(g("frac_last3"), 90), q(g("hx_tax"), 90), q(g("hz_tax"), 90)))

    print("\n=== 支撑段：真滑移 vs 踝位移 ===")
    for s in "LR":
        if not ST[s]:
            continue
        sl = np.array([w["slip"] for w in ST[s]])
        an = np.array([w["ankle_travel"] for w in ST[s]])
        ad = np.array([w["advance"] for w in ST[s]])
        print(f"  {s}: n={len(ST[s])} 时长={np.mean([w['n_steps'] for w in ST[s]])*DT:.3f}s "
              f"真滑移 mean={sl.mean():.3f} p50={med(sl):.3f} | 踝位移={an.mean():.3f} "
              f"(滑移/踝位移={sl.mean()/max(an.mean(),1e-9):.2f}) | 期间身体前进={ad.mean():.3f} m")

    print("\n=== 判别力 + 与 reach(相对髋) 的事件级相关（同一策略内）===")
    print("%-20s %9s %9s %8s %8s %8s" % ("observable", "L med", "R med", "R/L", "corrL", "corrR"))
    for name, k in (("drive_old 屈髋", "drive_old"), ("drive_new 踝进展", "drive_new"),
                    ("hip_rom 行程°", "hip_rom"), ("kneeExt 末1/3°", "knee_ext_last3"),
                    ("clearance", "clearance"), ("swing时长", "dur"), ("frac_last3", "frac_last3")):
        if not SW["L"] or not SW["R"]:
            break
        a = np.array([w[k] for w in SW["L"]]); b = np.array([w[k] for w in SW["R"]])
        rL = np.array([w["reach_hip"] for w in SW["L"]])
        rR = np.array([w["reach_hip"] for w in SW["R"]])
        c1 = np.corrcoef(a, rL)[0, 1] if a.std() > 1e-9 else float("nan")
        c2 = np.corrcoef(b, rR)[0, 1] if b.std() > 1e-9 else float("nan")
        print("%-20s %9.4f %9.4f %8.1f %8.2f %8.2f" % (
            name, med(a), med(b), med(b) / max(abs(med(a)), 1e-9), c1, c2))

    if SW["L"] and SW["R"]:
        print("\n=== 批次C' 标定候选分布 ===")
        for key, lab in (("reach_hip", "落点超前量(相对髋,m)"), ("knee_ext_last3", "末期伸膝°"),
                         ("drive_new", "踝相对躯干进展(m/步)"), ("hip_rom", "髋行程°")):
            a = np.array([w[key] for w in SW["L"]]); b = np.array([w[key] for w in SW["R"]])
            print(f"{lab:26s} 左 p10={q(a,10):+.3f} p50={med(a):+.3f} p75={q(a,75):+.3f} "
                  f"p90={q(a,90):+.3f} | 右 p10={q(b,10):+.3f} p50={med(b):+.3f} p90={q(b,90):+.3f}")
        print("标定原则：floor 放左腿 p50 略上（不给白拿），top 放左右差距的 60-70% 处（够得着、不满奖）")
    vec.close()


if __name__ == "__main__":
    main()
