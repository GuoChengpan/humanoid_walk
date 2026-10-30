"""(甲) 的立项前实测：**左髋的探索到底缺不缺**，以及"定向加噪"能不能既甩开腿又不摔。

背景与判据来源（CONTEXT_HANDOFF §17/§24）：
 · det 左腿摆动行程 1.78°（右腿 58-61°），连续 5 轮奖励改动都没动它；
 · 开环探针证明它**不是执行器限制**（pin ctrl=−1 → 114°@10.5rad/s，且基线 |ctrl| p95=0.39、0% 饱和）；
 · ⑥ 把**全局** ent_coef 0.05→0.10 跑了 770K 步，det ROM 仍然 1.78°——所以"全局加熵"已被证伪。
本脚本要回答的是剩下的那个可能性：**SAC 的 std 是逐维可学的，左髋那一维可能已经塌成很小的值，
全局熵系数救不了它**（熵项对每维一视同仁，但梯度会把"现在没用的自由度"的 std 继续压小）。

三段输出：
 1) 逐维 std（`actor.log_std`）+ 关节名，看左髋在 17 维里排第几、和右髋差多少
 2) det rollout 里每维**实际下发的动作**分布与饱和比例（左髋是不是长期贴着同一个值）
 3) 干预约定：把左髋那一维 std 强制抬到若干档，跑若干条随机集，量
    「左腿摆动行程 p50/p90」和「存活 ep_len / 摔率」——**两者都要**：只甩腿但全摔，
    buffer 里就是崩盘数据（§3.1 的老死亡螺旋）；能活着甩开，定向探索才有意义。
只读快照（best_model_snapshots），不碰 best_model.zip、不改训练文件。
"""
import copy
import glob
import re
import sys

import mujoco
import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

HORIZON = 1500
# 训练停止后 best_model.zip 才是终版；跑着的时候必须改用 best_model_snapshots/ 里的历史快照，
# 否则会读到正在被覆写的文件（本文件的 SNAP 两种都支持，命令行可传路径）
SNAP = "/home/guo/workspace/humanoid_walk/best_model/best_model.zip"
SNAP_DIR = "/home/guo/workspace/humanoid_walk/best_model/best_model_snapshots/"


def build(path):
    raw = gym.make("Humanoid-v4", max_episode_steps=HORIZON)
    patch_mirror_symmetry(raw.unwrapped.model)
    env = WalkRewardWrapper(raw, swing_drive_weight=2.0)
    model = SAC.load(path, env=env, device="cpu")
    return model, env, raw


def actuator_names(raw):
    m = raw.unwrapped.model
    return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "act%d" % i
            for i in range(m.nu)]


def dump_std(model, names):
    ld = None
    for nm, p in model.policy.named_parameters():
        if "log_std" in nm and p.numel() == len(names):
            ld = p.detach().cpu().numpy().ravel()
    if ld is None:
        print("!! 没找到逐维 log_std（可能用了 use_sde 的 state-dependent std）")
        return None
    std = np.exp(ld)
    order = np.argsort(std)
    print("逐维动作 std（小→大）：")
    for i in order:
        print("   %-22s std=%.4f  %s" % (names[i], std[i],
              "★左髋前后" if "left_hip_y" in names[i] else
              ("  右髋前后(对照)" if "right_hip_y" in names[i] else "")))
    return std


def det_actions(model, env, names, seed=123):
    obs, _ = env.reset(seed=seed)
    acts, roms = [], []
    for _ in range(HORIZON):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        acts.append(a.copy())
        roms.append(float(info.get("last_rom_L_deg", 0.0)))
        if term or trunc:
            break
    A = np.array(acts)
    print("\ndet rollout（%d 步）每维下发动作：mean/std/max|a|（饱和=|a|>0.95 占比）" % len(A))
    for i, nm in enumerate(names):
        if "hip_y" not in nm and "knee" not in nm:
            continue
        print("   %-22s mean=%+.3f std=%.3f max|a|=%.3f 饱和%.0f%%" % (
            nm, A[:, i].mean(), A[:, i].std(), np.abs(A[:, i]).max(),
            100.0 * (np.abs(A[:, i]) > 0.95).mean()))
    print("   det 左腿摆动行程 p50/p90 = %.2f° / %.2f°" % (
        np.percentile(roms, 50), np.percentile(roms, 90)))
    return A


def forced(model, target_std_dim, value):
    """返回改过指定那一维 log_std 的**副本**（不碰原对象，更不碰磁盘权重）。"""
    m2 = copy.deepcopy(model)
    for nm, p in m2.policy.named_parameters():
        if "log_std" in nm and p.numel() > target_std_dim:
            with np.errstate():
                pass
            p.data[target_std_dim] = float(np.log(value))
    return m2


def stochastic_probe(model, env, names, idx, levels, n_ep=3, cur_std=None):
    print("\n定向抬 std 的干预实验（左髋前后 = %s，当前 std=%s）" % (
        names[idx], "%.4f" % cur_std if cur_std else "?"))
    for lv in levels:
        m2 = forced(model, idx, lv)
        obs, _ = env.reset(seed=7)
        eps, rom_all = [], []
        cur, ev, run = 0, 0.0, []
        steps = 0
        while len(eps) < n_ep and steps < n_ep * HORIZON + 10:
            a, _ = m2.predict(obs, deterministic=False)
            obs, r, term, trunc, info = env.step(a)
            cur += 1
            steps += 1
            if "ev_rom_L_mean" in info:
                ev = float(info["ev_rom_L_mean"])
            run.append(float(info.get("last_rom_L_deg", 0.0)))
            if term or trunc:
                eps.append((cur, bool(term), ev))
                rom_all.extend(run)
                cur, ev, run = 0, 0.0, []
                obs, _ = env.reset()
        lens = [e[0] for e in eps]
        fell = np.mean([1.0 if e[1] else 0.0 for e in eps])
        r = np.array(rom_all)
        print("  std=%.2f → ep_len %s 摔率 %.0f%% | 左腿摆动行程 p50=%.1f° p90=%.1f° max=%.1f°" % (
            lv, str(lens), 100 * fell, np.percentile(r, 50), np.percentile(r, 90), r.max()))
        del m2


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        snaps = sorted(glob.glob(SNAP_DIR + "best_model_*.zip"),
                       key=lambda p: int(re.search(r"_(\d+)\.zip", p).group(1)))
        path = snaps[-1] if snaps else SNAP      # 快照被归档走就用终版 best_model.zip
    print("权重：%s\n" % path.split("/")[-1])
    model, env, raw = build(path)
    names = actuator_names(raw)
    idx_L = [i for i, n in enumerate(names) if "left_hip_y" in n][0]
    std = dump_std(model, names)
    if std is not None:
        iR = [i for i, n in enumerate(names) if "right_hip_y" in n][0]
        print("\n左髋 std=%.4f 右髋 std=%.4f 比值=%.2f ｜ 左髋排名 %d/17（1=最小）" % (
            std[idx_L], std[iR], std[idx_L] / std[iR], int(np.argsort(np.argsort(std))[idx_L]) + 1))
    det_actions(model, env, names)
    levels = [float(std[idx_L]), 0.15, 0.30, 0.50] if std is not None else [0.05, 0.15, 0.30, 0.50]
    stochastic_probe(model, env, names, idx_L, levels, cur_std=float(std[idx_L]) if std is not None else None)


if __name__ == "__main__":
    main()
