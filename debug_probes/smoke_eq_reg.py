"""方案A v2 干跑：归一化混合 λ 的 A/B 机制验证（单进程、小模型、不碰用户 checkpoint）
同一 warm-start(592K)、同种子、同环境，唯一变量 = eq_mix_lambda (0 / 0.2 / 0.4)。
在训练前录制的固定探针状态集上测等变缺口：λ>0 必须显著收敛，λ=0 基本不动。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, gymnasium as gym, torch as th
import torch.nn as nn
from stable_baselines3 import SAC
from mirror_map import MirrorMap, patch_mirror_symmetry
from train_humanoid_sac import WalkRewardWrapper, MirrorEqSAC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

e = gym.make("Humanoid-v4", max_episode_steps=1500)
patch_mirror_symmetry(e.unwrapped.model)
e = WalkRewardWrapper(e)
mm = MirrorMap(e.unwrapped.model)

old = SAC.load(os.path.join(ROOT, "best_model/best_model.zip"), device="cpu")
print(f"warm-start num_timesteps={old.num_timesteps}")

# ---- 固定探针状态集（warm-start 策略的确定性轨迹） ----
obs, _ = e.reset(seed=7)
O, A = [], []
done = False
while not done and len(O) < 60:
    a, _ = old.predict(obs, deterministic=True)
    O.append(obs.copy()); A.append(a.copy())
    obs, r, term, trunc, _ = e.step(a)
    done = term or trunc
O, A = np.array(O), np.array(A)
tgt = np.array([mm.mirror_action(a) for a in A])
print(f"探针集: {len(O)} 态")

def gap_of(model, anchored=False):
    """anchored=True: π(M(s)) vs mirror(a_warm)（固定锚，策略漂移会误报）
    anchored=False: π(M(s)) vs mirror(π(s))（自洽，同一模型两侧，不受漂移影响）"""
    ds, act_stds = [], []
    mean_all = []
    for t in range(len(O)):
        ms = th.as_tensor(mm.mirror_obs(O[t])[None], dtype=th.float32, device=model.device)
        mean_ms, _, _ = model.actor.get_action_dist_params(ms)
        pi_ms = th.tanh(mean_ms)[0].detach().numpy()
        mean_s, _, _ = model.actor.get_action_dist_params(
            th.as_tensor(O[t][None], dtype=th.float32, device=model.device))
        pi_s = th.tanh(mean_s)[0].detach().numpy()
        mean_all.append(pi_s)
        if anchored:
            d = np.linalg.norm(pi_ms - tgt[t])
        else:
            d = np.linalg.norm(pi_ms - mm.mirror_action(pi_s))
        ds.append(d)
    # 动作多样性（防对称塌缩：所有状态输出同一动作时 std→0）
    diversity = float(np.std(np.array(mean_all)))
    return float(np.median(ds)), diversity

def build(lam):
    m = MirrorEqSAC('MlpPolicy', e, buffer_size=30_000, batch_size=256, train_freq=256,
                    gradient_steps=8, learning_starts=256, ent_coef=0.05, target_entropy=-8.5,
                    policy_kwargs={'net_arch': [512, 256], 'activation_fn': nn.Tanh},
                    seed=0, verbose=0, mirror_map=mm, eq_mix_lambda=lam)
    m.policy.load_state_dict(old.policy.state_dict())
    m.learning_starts = 256
    return m

g_w, gw_d = gap_of(old)
print(f"自洽缺口基线（warm 本体）: {g_w:.3f} (动作多样性 {gw_d:.3f})")

results = {}
for lam in (0.0, 0.2, 0.4):
    m = build(lam)
    m.learn(total_timesteps=4000, reset_num_timesteps=False, log_interval=None)
    eqv = m.logger.name_to_value.get("train/eq_reg_loss")
    g, div = gap_of(m)
    results[lam] = (g, div)
    print(f"λ={lam}: 自洽缺口 {g:.3f} (多样性 {div:.3f}) (eq_reg_loss 最新 {eqv if eqv is None else round(eqv,4)})")
    del m

_, s0 = results[0.0]
print(f"\n自洽缺口对照: λ=0 {s0:.3f} | λ=0.2 {results[0.2][0]:.3f} | λ=0.4 {results[0.4][0]:.3f}")
ok = results[0.4][0] < s0 * 0.85 and results[0.4][1] > results[0.0][1] * 0.5
print("✓ 自洽缺口随 λ 收敛且动作未塌缩" if ok else "✗ 自洽形式亦未见效")
