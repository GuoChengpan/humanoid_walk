"""读 V9 归档（"实现交替慢跑"）到底做了什么 —— 物理口径判定，不看它自己的奖励数字。

为什么要单独一支探针：V9 是 8 月 30 日的存档，它的 `best_extra_logs.jsonl` 记的是**当时的奖励**，
而当时的奖励函数和现在完全不同（通道、权重、门都不一样）。所以"它是不是交替慢跑"**不能**用它的
ret / 它的 ev_* 列来判断，只能用物理量重新量一遍。

判据（跑之前钉死，全部用真接触判定 = MuJoCo 的 d.contact 里"脚 geom × floor geom"）：
  慢跑（jog）= 有腾空：flight_frac ≥ 0.15
  交替        = 相邻两次落地来自不同脚的比例 ≥ 0.8（不是同脚连跳）
  真前进      = 净前进速度 ≥ 0.5 m/s，且**单步前进**（同一脚两次落地之间躯干的位移）≥ 0.15 m
  对称        = |step_L − step_R| / max ≤ 0.30
四条同时满足才算"实现了交替慢跑"。差任何一条就只报"差在哪"，不改口。

内存安全：不走 `SAC.load`（它会装配 9.14 GB 的 replay buffer），直接 torch.load zip 里的 policy.pth。
"""
import io
import json
import os
import sys
import zipfile

import numpy as np
import gymnasium as gym
import mujoco
import torch

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")
torch.serialization.add_safe_globals([]) if False else None

CKPT = sys.argv[1] if len(sys.argv) > 1 else "best_model_saved/V9/V9-实现交替慢跑/best_model.zip"
SEEDS = tuple(int(x) for x in os.environ.get("V9_SEEDS", "7,31,55,123,2026").split(","))
STEPS = 1000


def load_policy(path, obs_space, act_space):
    """只重建 actor（SACPolicy），不碰 replay buffer ⇒ 不申请 9.14 GB。"""
    import torch.nn as nn
    from stable_baselines3.sac.policies import SACPolicy
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        blob = z.read("policy.pth")
        try:
            meta = json.loads(z.read("data").decode("utf-8"))
        except Exception:
            meta = {}
        ver = z.read("_stable_baselines3_version").decode() if "_stable_baselines3_version" in names else "?"
    sd = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
    in_dim = int(sd["actor.latent_pi.0.weight"].shape[1])
    hid = [int(sd[k].shape[0]) for k in ("actor.latent_pi.0.weight", "actor.latent_pi.2.weight") if k in sd]
    print(f"  存档 SB3 版本 {ver}   num_timesteps={meta.get('num_timesteps','?')}"
          f"   actor 输入维 {in_dim}（本 env obs={obs_space.shape[0]}）  隐藏层 {hid}")
    assert in_dim == obs_space.shape[0], "obs 维度不匹配：这份权重不是给这个 env 的，先别跑"
    pol = SACPolicy(obs_space, act_space, lambda x: 1e-4, net_arch=hid, activation_fn=nn.Tanh, use_sde=False)
    missing = pol.load_state_dict(sd, strict=False)
    if missing.missing_keys or missing.unexpected_keys:
        print(f"  ⚠ 键不完全对齐 missing={missing.missing_keys[:4]} unexpected={missing.unexpected_keys[:4]}")
    return pol.to("cpu").eval(), names, meta


env0 = gym.make("Humanoid-v4")
print(f"载入 {CKPT}")
policy, znames, meta = load_policy(CKPT, env0.observation_space, env0.action_space)
print(f"  zip 内容: {[n for n in znames if n != 'pytorch_variables.pth']}")
env0.close()
print(f"  obs 维度 {env0.observation_space.shape}  act {env0.action_space.shape} ⇒ 与该版本兼容性检查通过与否看下面 rollout")
env0.close()

from mirror_map import patch_mirror_symmetry


def roll(seed, patch):
    env = gym.make("Humanoid-v4", max_episode_steps=STEPS)
    if patch:
        patch_mirror_symmetry(env.unwrapped.model)
    m, d = env.unwrapped.model, env.unwrapped.data
    gl = {s: [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(f"{s}_foot").id] for s in ("left", "right")}
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    obs, _ = env.reset(seed=seed)
    x0 = d.qpos[0]
    hits = []
    lander = []
    z_list, vx_inst = [], []
    last_land_x = {}
    steps = {s: [] for s in ("left", "right")}
    for t in range(STEPS):
        a, _ = policy.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        prev = hits[-1] if hits else (False, False)          # 上一帧的接触 = 本帧的"步前"状态
        hit = {"left": False, "right": False}
        for i in range(d.ncon):
            ga, gb = d.contact[i].geom1, d.contact[i].geom2
            for s in ("left", "right"):
                if (ga in gl[s] and gb == floor) or (gb in gl[s] and ga == floor):
                    hit[s] = True
        hits.append((hit["left"], hit["right"]))
        for s_i, s in ((0, "left"), (1, "right")):
            if hit[s] and not prev[s_i]:                     # 上升沿 = 本帧落地
                lander.append(s)
                if s in last_land_x:
                    steps[s].append(d.qpos[0] - last_land_x[s])
                last_land_x[s] = d.qpos[0]
        z_list.append(d.qpos[2]); vx_inst.append(d.qvel[0])
        if term or trunc:
            break
    H = np.array(hits, dtype=bool)
    n = len(H)
    dt = m.opt.timestep * env.unwrapped.frame_skip
    both = (H[:, 0] & H[:, 1]).mean()
    flight = ((~H[:, 0]) & (~H[:, 1])).mean()
    order = np.array(lander)
    alt = float((order[1:] != order[:-1]).mean()) if len(order) > 1 else float("nan")
    v_net = (d.qpos[0] - x0) / max(n * dt, 1e-9)
    sl = {s: (float(np.median(steps[s])) if steps[s] else float("nan")) for s in ("left", "right")}
    n_land_s = len(lander) / max(n * dt, 1e-9)
    return dict(n=n, dt=dt, duty_L=H[:, 0].mean(), duty_R=H[:, 1].mean(), ds=both, flight=flight,
                land_s=n_land_s, v_net=v_net, v_mean=float(np.mean(vx_inst)), z=float(np.median(z_list)),
                step_L=sl["left"], step_R=sl["right"], alt=alt, n_land=len(lander))


for patch in (True, False):
    rs = [roll(s, patch) for s in SEEDS]
    med = lambda k: float(np.median([r[k] for r in rs]))
    print(f"\n=== mirror 对称修补 = {'开' if patch else '关'}（{len(SEEDS)} seeds 中位）===")
    print(f"  存活 {med('n'):5.0f}/{STEPS} 帧 (dt={rs[0]['dt']:.4f}s)   净前进 v_net {med('v_net'):+.3f} m/s   瞬时 v 均值 {med('v_mean'):+.3f}"
          f"   躯干高 z {med('z'):.3f}")
    print(f"  duty L/R {med('duty_L'):.3f}/{med('duty_R'):.3f}   双支撑 ds {med('ds'):.3f}   腾空 flight {med('flight'):.3f}")
    print(f"  落地 {med('land_s'):.2f} 次/s（共 {med('n_land'):.0f} 次）   交替率(相邻落地异脚) {med('alt'):.3f}")
    print(f"  单步前进 L {med('step_L'):+.4f} m   R {med('step_R'):+.4f} m")
    ok = {"腾空≥0.15": med("flight") >= 0.15, "交替≥0.80": med("alt") >= 0.80,
          "v_net≥0.5": med("v_net") >= 0.5, "单步≥0.15": min(med("step_L"), med("step_R")) >= 0.15,
          "对称≤0.30": abs(med("step_L") - med("step_R")) / max(abs(med("step_L")), abs(med("step_R")), 1e-9) <= 0.30}
    print("  判据：" + "  ".join(f"{'✔' if v else '✘'}{k}" for k, v in ok.items())
          + f"   ⇒ {'**确实实现了交替慢跑**' if all(ok.values()) else '不成立（见上面差的项）'}")
