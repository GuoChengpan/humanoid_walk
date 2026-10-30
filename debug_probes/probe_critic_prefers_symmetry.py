"""判决性实验：**critic 到底知不知道"更对称/更用左腿"的动作更值钱？**

在**当前策略自己访问的状态**上（这是关键：actor 的梯度只在这些状态上求值），对每个状态比较四个动作的 Q：
  a0  策略当前动作（基线）
  am  纯镜像动作 mirror(a0)          —— 若策略"镜像对称"会产生的动作（两腿同相）
  ag  滑移对称动作 mirror(a0[t−L])   —— 走路真正需要的对称（左右反相半周期）
  a*  critic 的贪心动作（对 a 做梯度上升）—— critic 认为的最优
读法（三种结果对应三种完全不同的处置）：
  Q*−Q0 很大          → 值函数已经知道有改进，瓶颈在 **actor/探索**（镜像数据、改奖励都排不到前面）
  Q*−Q0 ≈ 0           → 已到学到的 Q 的真不动点 → 问题在**奖励/数据**，不是优化
  Q_g > Q0 而 Q_m ≤ Q0 → critic 偏好反相（走路式）对称、不偏好同相 → 我们增广用的是**错的对称**
只读：不写模型、不碰训练进程。Q 用 γ=0.99 ⇒ 有效视界≈100 步，1 单位 Q ≈ 15 分/满集(1500步)。
"""
import os
import sys

import numpy as np
import torch as th

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

from mirror_map import MirrorMap
from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

MODEL = os.environ.get("PROBE_MODEL", "/home/guo/workspace/humanoid_walk/best_model/best_model.zip")
N_ROLL = 400
N_PROBE = 120
GAIT_LAG = 13          # 半周期≈13 步（0.20s，实测腿周期 0.35-0.41s）
GD_STEPS = 120
GD_LR = 0.05


def main():
    raw = gym.make("Humanoid-v4", max_episode_steps=1500)
    m = raw.unwrapped.model
    patch_mirror_symmetry(m)
    env = WalkRewardWrapper(raw, swing_drive_weight=2.0)
    mm = MirrorMap(m)
    anames = [m.joint(m.actuator_trnid[a][0]).name for a in range(m.nu)]
    iLY, iRY = anames.index("left_hip_y"), anames.index("right_hip_y")
    iLK, iRK = anames.index("left_knee"), anames.index("right_knee")

    model = SAC.load(MODEL, env=env, device="cpu")
    obs, _ = env.reset(seed=123)

    states, acts = [], []
    hist = []
    for t in range(N_ROLL):
        a, _ = model.predict(obs, deterministic=True)
        a = np.asarray(a, dtype=np.float64)
        obs, r, term, trunc, info = env.step(a)
        hist.append(a.copy())
        if len(hist) > 200:
            hist.pop(0)
        if t > 60 and (states.__len__() < N_PROBE) and t % 3 == 0:
            states.append(obs.copy())
            # 滑移对称：取 LAG 步前的动作做镜像（把"右腿刚才做的事"交给左腿）
            src = hist[-(GAIT_LAG + 1)] if len(hist) > GAIT_LAG else a
            acts.append((a.copy(), mm.mirror_action(a).copy(), mm.mirror_action(src).copy()))
        if term or trunc:
            obs, _ = env.reset()
            hist = []
    print(f"模型: {os.path.basename(MODEL)}  访问状态数={len(states)}  "
          f"探针动作=基线/纯镜像(同相)/滑移镜像(反相 lag={GAIT_LAG})/critic贪心")

    S = th.from_numpy(np.stack(states)).float()      # 裸 env：obs 形状 (376,)

    def q_of(A):
        # SB3 2.9: critic(obs, actions) -> (qf0, qf1)；用 twin 的最小值（actor 就是照这个优化的）
        with th.no_grad():
            qa, qb = model.critic(S, th.from_numpy(A))
            return th.minimum(qa, qb).numpy().ravel()

    A0 = np.stack([a[0] for a in acts]).astype(np.float32)
    AM = np.stack([a[1] for a in acts]).astype(np.float32)
    AG = np.stack([a[2] for a in acts]).astype(np.float32)
    # 定向反事实：把左腿（髋y+膝）的命令**沿它当前方向放大一倍**，其它通道不动
    AA = A0.copy()
    AA[:, [iLY, iLK]] = np.clip(A0[:, [iLY, iLK]] * 2.0, -1.0, 1.0)
    q0, qm, qg, qa_ = q_of(A0), q_of(AM), q_of(AG), q_of(AA)

    # critic 贪心：从 a0 出发对动作做投影梯度上升
    A = th.from_numpy(A0).clone().float().requires_grad_(True)
    opt = th.optim.Adam([A], lr=GD_LR)
    for _ in range(GD_STEPS):
        opt.zero_grad()
        qa, qb = model.critic(S, A)
        loss = -th.minimum(qa, qb).mean()      # 梯度**上升**（此前写成下降，得到的其实是 Q 最低的动作）
        loss.backward()
        opt.step()
        with th.no_grad():
            A.clamp_(-1.0, 1.0)
    AST = A.detach().numpy()
    qs = q_of(AST)

    u = lambda d: (d.mean(), d.std() / max(np.sqrt(len(d)), 1))
    print(f"\n=== Q 差（单位：Q；×15 = 分/满集）===")
    for nm, qq in (("纯镜像(同相)", qm), ("滑移镜像(反相/走路式)", qg),
               ("左腿命令×2(定向)", qa_), ("critic 贪心(上升)", qs)):
        d = qq - q0
        U, udid = u(d)
        print(f"  {nm:24s} ΔQ={U:+.3f}±{udid:.3f}  = {U*15:+7.1f} 分/集   "
              f"认为更好的状态占 {100*np.mean(d > 0.02):.0f}%")

    print("\n=== 贪心动作把哪条腿改了？（Δa = a* − a0）===")
    dA = AST - A0
    for nm, i in (("左髋 y", iLY), ("右髋 y", iRY), ("左膝", iLK), ("右膝", iRK)):
        print(f"  {nm:8s} Δa 均={dA[:, i].mean():+.3f} |a0|均={np.abs(A0[:, i]).mean():.3f} "
              f"|a*|均={np.abs(AST[:, i]).mean():.3f}  |a*|>|a0| 占 {100*np.mean(np.abs(AST[:, i])>np.abs(A0[:, i])+0.02):.0f}%")
    print(f"  左腿命令幅度总和：a0={np.abs(A0[:, [iLY, iLK]].sum(1)).mean():.3f} "
          f"a*={np.abs(AST[:, [iLY, iLK]].sum(1)).mean():.3f} "
          f"（右腿：a0={np.abs(A0[:, [iRY, iRK]].sum(1)).mean():.3f} a*={np.abs(AST[:, [iRY, iRK]].sum(1)).mean():.3f}）")
    dl = np.abs(AST[:, iLY]) - np.abs(A0[:, iLY])
    dr = np.abs(AST[:, iRY]) - np.abs(A0[:, iRY])
    print(f"  Δ|左髋| 与 Δ|右髋| 的相关={np.corrcoef(dl, dr)[0,1]:+.2f}"
          f"（正=它想让两腿一起用力；负=它想把活从右腿挪到左腿）")
    print(f"  左髋命令被加大(Δ|a|>0.05)的状态占 {100*np.mean(dl > 0.05):.0f}%  "
          f"被减小(Δ|a|<-0.05)的占 {100*np.mean(dl < -0.05):.0f}%")
    print(f"\n  基线 Q0 均={q0.mean():.2f}   critic 贪心 Q* 均={qs.mean():.2f}")

    # 关键补充：**critic 自己对称吗？** 比较 Q(s,a) 与 Q(M s, M a)
    MS = th.from_numpy(np.stack([mm.mirror_obs(x) for x in states]).astype(np.float32))
    MA = np.stack([mm.mirror_action(a) for a in A0]).astype(np.float32)
    with th.no_grad():
        def qmin(X, Ac):
            qa, qb = model.critic(X, th.from_numpy(Ac))
            return th.minimum(qa, qb).numpy().ravel()
        qm2 = qmin(MS, MA)
    d = qm2 - q0
    print(f"\n=== critic 的等变性：Q(Ms,Ma) − Q(s,a) = {d.mean():+.3f}±{d.std()/np.sqrt(len(d)):.3f}"
          f"   （|相对|={abs(d.mean())/max(abs(q0.mean()),1e-9)*100:.1f}% of Q0={q0.mean():.1f}）")
    print(f"    镜像后值更高占 {100*np.mean(d>0.02):.0f}%、更低占 {100*np.mean(d<-0.02):.0f}%")
    print("    读法：|Δ| 很小 → 镜像数据**确实**把 critic 对称化了，那对称动作在 s 上被扣分就是真判断；")
    print("          |Δ| 很大 → 镜像数据没进到值函数里（增广白做），此时的 Q(Ms,Ma) 比较不可信")


if __name__ == "__main__":
    main()
