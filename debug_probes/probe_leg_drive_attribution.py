"""腿到底有没有在"驱动"？把每一次落地拆成可量的量（只读，两份权重对照）。

用户的诊断：策略靠腾空/折叠把身体往前带、腿被拖着走 ⇒ 步幅来自身体位移而不是腿主动前摆。
这个说法可以被量化成四个可检验的量。对每一次落地（去抖 ≥5 帧）记录：
  · foot_behind_com = body_x − foot_x  @落地   （**正 = 脚落在躯干后面**⇒ 这条腿没在"迈步"，只是被身体带过去接着）
  · swing_foot_disp = foot_x @落地 − foot_x @该脚上次离地   （脚自己在空中前移了多少）
  · swing_body_disp = body_x @落地 − body_x @该脚上次离地   （这段时间身体前移了多少）
      ⇒ **drive_share = swing_foot_disp / max(swing_body_disp, ε)**：>1 才是"腿主动往前插"，≈0 是"腿被身体带着"
  · other_planted = 本次落地瞬间另一只脚是否在地上（=真单支撑里落脚，走路的必要条件；弹跳在飞行期落地 ⇒ False）
  · swing_frames  摆动时长
再给两档汇总：全部落地 / 只统计"另一只脚撑地"的落地（走路口径）。
"""
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

ROOT = "/home/guo/workspace/humanoid_walk/best_model/"
POL = [("⑧ 弹跳 672K", ROOT + "tmp_model_data/SAC_18/18_加重腾空惩罚/best_model_snapshots/best_model_672000.zip"),
       ("⑨ 碎步 1696K", ROOT + "tmp_model_data/SAC_18/18_8_从头训/best_model_snapshots/best_model_1696000.zip")]
HORIZON, SEEDS = 1500, (123, 2026)


def run(path, seed):
    raw = gym.make("Humanoid-v4", max_episode_steps=HORIZON)
    patch_mirror_symmetry(raw.unwrapped.model)
    env = WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=False)
    w, u = env, env.unwrapped
    model = SAC.load(path, env=env, device="cpu")
    obs, _ = env.reset(seed=seed)
    prev = {"L": True, "R": True}
    st = {s: dict(lift_fx=None, lift_bx=None, prev_land_fx=None) for s in "LR"}
    rows = []
    for _ in range(HORIZON):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        cur = {"L": bool(w.left_contact_prev), "R": bool(w.right_contact_prev)}
        bx = float(u.data.qpos[0])
        fx = {"L": float(u.data.body("left_foot").xpos[0]), "R": float(u.data.body("right_foot").xpos[0])}
        for s in "LR":
            if cur[s] and not prev[s]:                              # 本次落地
                o = "R" if s == "L" else "L"      # 修正：取**另一只**脚（上一版写反，取成了同侧）
                d = st[s]
                rec = dict(side=s, behind=bx - fx[s], foot_to_body=fx[s] - bx)
                if d["lift_fx"] is not None:
                    rec["foot_disp"] = fx[s] - d["lift_fx"]
                    rec["body_disp"] = bx - d["lift_bx"]
                else:
                    rec["foot_disp"] = rec["body_disp"] = float("nan")
                if d["prev_land_fx"] is not None:
                    rec["stride"] = fx[s] - d["prev_land_fx"]
                else:
                    rec["stride"] = float("nan")
                rec["other_planted"] = cur[o]
                rows.append(rec)
                d["prev_land_fx"] = fx[s]
            if prev[s] and not cur[s]:                              # 本次离地
                st[s]["lift_fx"], st[s]["lift_bx"] = fx[s], bx
            prev[s] = cur[s]
        if term or trunc:
            break
    del model, env, raw
    return rows


def stat(rows, tag, filt):
    rs = [r for r in rows if filt(r)]
    if not rs:
        print("   %-22s 无样本" % tag)
        return
    g = lambda k: np.array([r[k] for r in rs], float)
    fb, fd, bd, sr = g("foot_to_body"), g("foot_disp"), g("body_disp"), g("stride")
    ds = np.where(np.abs(bd) < 1e-6, np.nan, fd / np.where(np.abs(bd) < 1e-6, 1, bd))
    print("   %-22s n=%4d｜脚在躯干后的比例 %.0f%%（foot_to_body p50 %+.3f m）｜摆动中 脚/身 位移比 p50 %+.2f"
          "｜单步（同脚两次落地间距）p50 %+.3f m" % (
              tag, len(rs), 100.0 * float(np.mean(fb < 0)), float(np.median(fb)),
              float(np.nanmedian(ds)), float(np.nanmedian(sr))))


def main():
    for name, path in POL:
        rows = []
        for s in SEEDS:
            rows += run(path, s)
        print("\n=== %s ｜落地 %d 次（两种子）===" % (name, len(rows)))
        stat(rows, "全部落地", lambda r: True)
        stat(rows, "只算另一脚撑地的落地", lambda r: r["other_planted"])
        stat(rows, "只算腾空期落地(飞行中)", lambda r: not r["other_planted"])
        print("   落地瞬间另一只脚**撑地**的占比 %.0f%%（=走路口径；弹跳/跑步这个值应远低于 100%%）" % (
            100.0 * np.mean([r["other_planted"] for r in rows])))
        for side, tag in (("L", "左腿"), ("R", "右腿")):
            stat(rows, "  " + tag, lambda r, side=side: r["side"] == side)


if __name__ == "__main__":
    main()
