"""2M 跑完的终审：
  1. 各阶段快照的等变性缺口演变（1472K warm-start 为基线）——actor 对称性有没有随镜像数据生长
  2. 最终最优（1792K）步态审计：确定性 3 seed + 随机 3 seed，左/右摆幅与步幅 vs warm-start
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, gymnasium as gym
from stable_baselines3 import SAC
from mirror_map import MirrorMap, patch_mirror_symmetry
from train_humanoid_sac import WalkRewardWrapper

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP_DIR = os.path.join(ROOT, "best_model", "best_model_snapshots")
SNAPS = [
    ("1472K_warmstart", os.path.join(ROOT, "best_model/tmp_model_data/17_6/17_6_d_的最终版_加positions_swap/best_model_snapshots/best_model_1472000.zip")),
    ("208K", os.path.join(SNAP_DIR, "best_model_208000.zip")),
    ("624K", os.path.join(SNAP_DIR, "best_model_624000.zip")),
    ("1216K", os.path.join(SNAP_DIR, "best_model_1216000.zip")),
    ("1792K_best(=best_model.zip)", os.path.join(ROOT, "best_model/best_model.zip")),
]


def rollout(model, seed, det):
    e = gym.make("Humanoid-v4", max_episode_steps=1500)
    patch_mirror_symmetry(e.unwrapped.model)
    e = WalkRewardWrapper(e)
    obs, _ = e.reset(seed=seed)
    tot, n, done, info = 0.0, 0, False, {}
    while not done:
        a, _ = model.predict(obs, deterministic=det)
        obs, r, term, trunc, info = e.step(a)
        tot += float(r); n += 1
        done = term or trunc
    e.close()
    return n, tot, info.get("mean_left_swing"), info.get("mean_right_swing"), \
        info.get("last_left_stride"), info.get("last_right_stride")


def eq_gap(model, mm, seed=2):
    """一条确定性轨迹上的等变性缺口：median ||a(M(s)) − mirror(a(s))|| 及相对值"""
    e = gym.make("Humanoid-v4", max_episode_steps=1500)
    patch_mirror_symmetry(e.unwrapped.model)
    e = WalkRewardWrapper(e)
    obs, _ = e.reset(seed=seed)
    O, A = [], []
    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        O.append(obs.copy()); A.append(a.copy())
        obs, r, term, trunc, _ = e.step(a)
        done = term or trunc
    e.close()
    if len(O) < 20:
        return None
    idx = np.linspace(0, len(O) - 1, min(150, len(O))).astype(int)
    gaps, norms = [], []
    for t in idx:
        am, _ = model.predict(mm.mirror_obs(O[t])[None], deterministic=True)
        tgt = mm.mirror_action(A[t])
        gaps.append(np.linalg.norm(am[0] - tgt))
        norms.append(np.linalg.norm(tgt))
    gaps, norms = np.array(gaps), np.array(norms)
    return float(np.median(gaps)), float(np.median(gaps / np.maximum(norms, 1e-6)))


print("== 等变性缺口演变 ==")
for name, path in SNAPS:
    if not os.path.exists(path):
        print(f"{name}: 文件缺失 {path}")
        continue
    model = SAC.load(path, device="cpu")
    e = gym.make("Humanoid-v4", max_episode_steps=1500)
    patch_mirror_symmetry(e.unwrapped.model)
    mm = MirrorMap(e.unwrapped.model)
    e.close()
    g = eq_gap(model, mm)
    print(f"{name:28s} 等变性缺口: median={g[0]:.3f} 相对={g[1]:.2f}" if g else f"{name}: 轨迹太短")

print("\n== 最终最优（1792K）步态审计 ==")
model = SAC.load(SNAPS[-1][1], device="cpu")
print(f"num_timesteps={model.num_timesteps}")
det = [rollout(model, s, True) for s in (1, 2, 3)]
for n, r in zip((1, 2, 3), det):
    print(f"  确定性 seed{n}: {r[0]} 步, {r[1]:.0f} 分, 摆幅 左{r[2]:.1f}/右{r[3]:.1f}, 步幅 左{r[4]:.3f}/右{r[5]:.3f}")
sto = [rollout(model, s, False) for s in (1, 2, 3)]
for n, r in zip((1, 2, 3), sto):
    print(f"  随机std seed{n}: {r[0]} 步, {r[1]:.0f} 分, 摆幅 左{r[2]:.1f}/右{r[3]:.1f}")
surv = sum(1 for r in sto if r[0] >= 200)
print(f"随机存活率(≥200步): {surv}/3")
