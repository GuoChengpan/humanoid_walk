"""reach 该用哪个参照？——实测三种参照与"腰前倾"的耦合度，并给新 floor/top 定标。

现状：reward 里 reach = 落地帧 脚x − 同侧大腿(thigh)原点x。问题：躯干前倾会把大腿原点往后带、
      于是"倾得越狠 → reach 越好看"，pitch 本 run 已从 0.032 涨到 0.056（在涨）。
      §13 就记过泄漏量级 ≈0.89·sinθ（10°≈0.155 m）。

候选参照（都在落地帧取，脚用 left/right_foot body 的 x）：
  H = 同侧大腿原点 x      （现行）
  T = 躯干自由关节 x = qpos[0]
  P = 两侧大腿原点 x 的均值（≈髋中心，步态学里 step length 的标准参照）
判据：① |corr(reach, pitch)| 越小越好（不被前倾买通）；② 分布要能定出"够得着"的 floor/top；
      ③ 左右差异仍然真实（不能被参照抹平）。
只读：不改代码、不动模型。
"""
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

MODELS = [m for m in [
    os.environ.get("PROBE_MODEL", ""),
    "/home/guo/workspace/humanoid_walk/best_model/tmp_model_data/SAC_18/18_3_min(左右步幅)/best_model.zip",
] if m and os.path.exists(m)]
HORIZON = 1500
N_ROLL = 6000
SEEDS = (123, 2026, 7)


def collect(path):
    def _f():
        raw = gym.make("Humanoid-v4", max_episode_steps=HORIZON)
        patch_mirror_symmetry(raw.unwrapped.model)
        return WalkRewardWrapper(raw, swing_drive_weight=2.0)
    env = _f()
    model = SAC.load(path, env=env, device="cpu")
    m = env.unwrapped.model
    data = env.unwrapped.data
    foot = {s: m.body(f"{s}_foot").id for s in ("left", "right")}
    thigh = {s: m.body(f"{s}_thigh").id for s in ("left", "right")}
    out = {s: [] for s in ("L", "R")}
    pitches = []
    ep = 0
    obs, _ = env.reset(seed=SEEDS[0])
    prevL = prevR = True
    for t in range(N_ROLL):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        q = data.qpos[3:7]
        pitch = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")[1]
        lx, rx = float(data.xpos[foot["left"]][0]), float(data.xpos[foot["right"]][0])
        lhx, rhx = float(data.xpos[thigh["left"]][0]), float(data.xpos[thigh["right"]][0])
        torso_x = float(data.qpos[0])
        midhip = 0.5 * (lhx + rhx)
        # 用奖励自己的真接触判据，避免"看起来落地"的假阳性
        cl, cr, _, _ = env._foot_contact_state(m, data)
        if cl and not prevL:
            out["L"].append((lx - lhx, lx - torso_x, lx - midhip, pitch, ep))
        if cr and not prevR:
            out["R"].append((rx - rhx, rx - torso_x, rx - midhip, pitch, ep))
        prevL, prevR = cl, cr
        pitches.append(pitch)
        if term or trunc:
            ep += 1
            obs, _ = env.reset(seed=SEEDS[(ep) % len(SEEDS)])
    del model, env
    return {k: np.asarray(v) for k, v in out.items()}, np.asarray(pitches)


def depercol(x, y, ep):
    """按集去均值后再算相关，避免"不同集之间"的差异冒充集内耦合。"""
    num = sx2 = sy2 = 0.0
    for e in np.unique(ep):
        m = ep == e
        if m.sum() > 4:
            dx, dy = x[m] - x[m].mean(), y[m] - y[m].mean()
            num += float((dx * dy).sum())
            sx2 += float((dx * dx).sum())
            sy2 += float((dy * dy).sum())
    return num / max(np.sqrt(sx2 * sy2), 1e-12)


def slope(x, y, ep):
    num = den = 0.0
    for e in np.unique(ep):
        m = ep == e
        if m.sum() > 4:
            num += ((x[m] - x[m].mean()) * (y[m] - y[m].mean())).sum()
            den += ((y[m] - y[m].mean()) ** 2).sum()
    return num / max(den, 1e-12)


def main():
    for path in MODELS:
        rec, pit = collect(path)
        print(f"\n########## {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)} "
              f"##########")
        print(f"落地事件：左 {len(rec['L'])} 右 {len(rec['R'])}   pitch 范围 [{pit.min():+.3f},{pit.max():+.3f}] "
              f"均值 {pit.mean():+.4f}")
        print("%-22s %8s %8s %10s %10s   %s" % ("参照", "corrL", "corrR", "斜率L(分/rad)", "斜率R", "reach 分布 p10/p50/p90"))
        for i, nm in enumerate(("H 同侧大腿(现行)", "T 躯干 qpos[0]", "P 髋中心(两腿均值)")):
            row = []
            for s in "LR":
                A = rec[s]
                if len(A) < 6:
                    row = ["-"] * 5
                    break
                x, p, e = A[:, i], A[:, 3], A[:, 4]
                c = depercol(x, p, e)
                sl = slope(x, p, e)
                q10, q50, q90 = np.percentile(x, [10, 50, 90])
                row.append((c, sl, (q10, q50, q90)))
            if len(row) == 2:
                print("%-22s %8.2f %8.2f %10.3f %10.3f   L(%.3f/%.3f/%.3f) R(%.3f/%.3f/%.3f)" % (
                    nm, row[0][0], row[1][0], row[0][1], row[1][1],
                    row[0][2][0], row[0][2][1], row[0][2][2],
                    row[1][2][0], row[1][2][1], row[1][2][2]))
        A = rec["L"]
        if len(A) > 6:
            print("  现行 reach(H) 与 pitch 的关系：corr=%.2f  斜率=%.3f m/rad → 前倾 5°(0.087rad) 送 %.3f m reach"
                  % (depercol(A[:, 0], A[:, 3], A[:, 4]), slope(A[:, 0], A[:, 3], A[:, 4]),
                     slope(A[:, 0], A[:, 3], A[:, 4]) * 0.087))
            B = rec["R"]
            if len(A) > 10:
                k = max(2, int(.2 * len(A)))
                print("  前倾是否被奖励买通：reach(H) 最高20%%事件均值 − 最低20%%事件均值 = %.3f m"
                      % (np.sort(A[:, 0])[-k:].mean() - np.sort(A[:, 0])[:k].mean()))
            # 定标建议：用 T/P 参照后的分布给 floor/top
            for i, nm in ((1, "T"), (2, "P")):
                x = np.concatenate([A[:, i], B[:, i]]) if len(B) > 6 else A[:, i]
                p10, p50, p90 = np.percentile(x, [10, 50, 90])
                print(f"  若改用参照 {nm}: 全体落地 p10={p10:+.3f} p50={p50:+.3f} p90={p90:+.3f} "
                      f"→ floor 建议 {p10:.2f}、top 建议 {p50+2*(p90-p50):.2f}（让当前工作点在斜坡下 1/3 处）")


if __name__ == "__main__":
    main()
