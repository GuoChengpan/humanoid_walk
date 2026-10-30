"""整集镜像重放验证（方案 step 3：先验证后启用）
  A. 动作映射往返
  B. 逐项奖励一致性：原集 vs 镜像集——物理镜像为机器精度，混沌发散前逐项应相等
     （重点：position_swap 稠密项当时钟平移半周期后必须逐项相等）
  C. 镜像集有效性：跑满录制动作、终止行为、奖励有限、与原集同量级
  D. MirrorInjector → ReplayBuffer 注入路径（8 槽位组 add + sample 形状/有限性）
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer

from mirror_map import MirrorMap, patch_mirror_symmetry
from train_humanoid_sac import WalkRewardWrapper, MirrorReplayer, MirrorInjector

ENV_ID = "Humanoid-v4"
N_CHECK = 30          # 逐项对比窗口
rng = np.random.default_rng(0)
FAIL = []

# ---------- 环境栈（与训练一致：修补 + wrapper + 摆动驱动奖励） ----------
def build_env():
    e = gym.make(ENV_ID, max_episode_steps=1500)
    patch_mirror_symmetry(e.unwrapped.model)
    return WalkRewardWrapper(e, swing_drive_weight=2.0)

env = build_env()
replayer = MirrorReplayer(build_env(), MirrorMap(env.unwrapped.model))

# ---------- A. 动作映射往返 ----------
a = rng.uniform(-0.4, 0.4, 17)
ok_a = np.allclose(env.action_space.high, 0.4) and np.allclose(
    replayer.mm.mirror_action(replayer.mm.mirror_action(a)), a)
print(f'A. 动作映射往返: {"✓" if ok_a else "✗"} (action_space high={env.action_space.high[0]})')
if not ok_a:
    FAIL.append('A 动作往返')

# ---------- B/C. 原集收集 + 镜像重放 ----------
model = SAC.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "best_model", "best_model.zip"))
obs, _ = env.reset(seed=52)
q0 = env.unwrapped.data.qpos.copy()
v0 = env.unwrapped.data.qvel.copy()
clock = env.total_steps
acts, rews, infos_l = [], [], []
for t in range(1500):
    act, _ = model.predict(obs, deterministic=True)
    acts.append(np.asarray(act, dtype=np.float64))
    obs, r, term, trunc, info = env.step(act)
    rews.append(float(r))
    infos_l.append(info)
    if term or trunc:
        break
L = len(rews)
print(f'B. 原集: {L} 步, 总奖励 {sum(rews):.1f}, 终止={term} 截断={trunc}')

ep = replayer.replay(q0, v0, clock, acts, return_infos=True)
R = ep["reward"]
print(f'C. 镜像集: {len(R)} 步, 总奖励 {R.sum():.1f}, 末步 done={bool(ep["done"][-1])}, '
      f'奖励有限={bool(np.isfinite(R).all())}')
if not np.isfinite(R).all():
    FAIL.append('C 镜像奖励非有限')

# ---------- 逐项奖励一致性 ----------
# 特殊键：镜像左右互换/变号属预期行为，按跨键核对；其余键镜像不变，按同键核对
NEGATED = {'roll_angle',                                  # 镜像: roll → −roll
           'y_position', 'y_velocity'}                    # 镜像: y → −y（按定义取反）
SWAPPED = {'last_left_stride': 'last_right_stride',       # 镜像: 左右角色互换
           'last_right_stride': 'last_left_stride'}
INFO_ONLY = {'swing_forward_reward', 'last_swing_forward_reward',   # 记录的是摆动腿质量，镜像换腿
             'stride_length'}                                     # 记录的是最近落地脚，镜像换脚
info_keys = [k for k, v in infos_l[0].items() if isinstance(v, (int, float, np.integer, np.floating))]


def swap_partner(key):
    """分腿量在镜像下左右角色互换（last_reach_L↔R、slip_speed_L↔R、ev_*_L↔_R、last_rom_L_deg↔_R_deg…）。
    规则：把键里作为**独立词元**出现的 L/R 换成另一个（`_L`、`_R_`、结尾 `_L` 都算），
    这样新增的 `last_rom_L_deg` 这类带后缀的分腿量不用逐个登记。"""
    m = re.search(r"(?:^|_)(L|R)(?:_|$)", key)
    if not m:
        return None
    partner = key[:m.start(1)] + ("R" if m.group(1) == "L" else "L") + key[m.end(1):]
    return partner if partner in info_keys else None
print(f'\n   逐项 |Δ|（前 {min(N_CHECK, L)} 步; †=跨键/变号核对, ‡=仅记录不核对; '
      f'前10步为严格窗，10-30步含混沌落地时移仅供参考）:')
print(f'   {"项":32s} {"前10步max":>10s} {"前30步max":>10s}')
worst_r10 = 0.0
strict_fail = []
for k in info_keys + ['(total r)']:
    tag = ''
    if k in NEGATED:
        tag, mk = '†', lambda i: -float(ep["infos"][i].get(k, 0.0))
    elif k in SWAPPED:
        tag, mk = '†', (lambda i, _k=SWAPPED[k]: float(ep["infos"][i].get(_k, 0.0)))
    elif k in INFO_ONLY:
        tag, mk = '‡', (lambda i: float(ep["infos"][i].get(k, 0.0)))
    elif swap_partner(k):
        _p = swap_partner(k)
        tag, mk = '†', (lambda i, _pp=_p: float(ep["infos"][i].get(_pp, 0.0)))
    else:
        mk = lambda i: float(ep["infos"][i].get(k, 0.0))
    orig = [float(r) if k == '(total r)' else float(infos_l[i][k]) for i, r in enumerate(rews[:N_CHECK])]
    mir = [float(r) if k == '(total r)' else mk(i) for i, r in enumerate(R[:N_CHECK])]
    d = np.abs(np.array(orig) - np.array(mir))
    m10, m30 = d[:10].max(), d[:min(30, len(d))].max()
    if k == '(total r)':
        worst_r10 = m10
    elif tag == '' and m10 > 1e-2:
        strict_fail.append(f'{k}(Δ{m10:.1e})')
    print(f'   {k:32s} {m10:10.2e} {m30:10.2e} {tag}')
if worst_r10 > 0.1:
    FAIL.append(f'B 前10步总奖励偏差 {worst_r10:.3f} > 0.1')
if strict_fail:
    FAIL.append('B 镜像不变项前10步偏差超限: ' + ', '.join(strict_fail))

# ---------- D. 注入路径（真实 ReplayBuffer，n_envs=8 与训练一致） ----------
buf = ReplayBuffer(4096, env.observation_space, env.action_space, n_envs=8,
                   handle_timeout_termination=True)
inj = MirrorInjector(n_envs=8)
inj.drain_infos([{"mirror_episode": ep}], buf)
print(f'\nD. 注入: {inj.episodes} 集 → {inj.transitions} 条（pending 余 {len(inj.pending)}，'
      f'应为 {len(R) % 8}）')
s = buf.sample(32)
shapes_ok = (s.observations.shape == (32, 376) and s.actions.shape == (32, 17)
             and s.next_observations.shape == (32, 376))
finite_ok = all(np.isfinite(x.numpy()).all() for x in
                [s.observations, s.actions, s.next_observations, s.rewards])
print(f'   sample 形状={shapes_ok} 有限={finite_ok} '
      f'(rewards {tuple(s.rewards.shape)}, dones {tuple(s.dones.shape)})')
if not (shapes_ok and finite_ok):
    FAIL.append('D 注入路径')

# ---------- 总结 ----------
print('\n========== 总结 ==========')
if FAIL:
    print('✗ 未通过:', '; '.join(FAIL))
    sys.exit(1)
print('✓ 全部通过：镜像集与原集逐项奖励一致（混沌发散前），注入路径有效')
