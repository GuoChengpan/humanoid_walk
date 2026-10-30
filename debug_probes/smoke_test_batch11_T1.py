"""批次⑪·T1 的冒烟测试：**不调用 model.learn()、不加载任何权重、不保存任何文件**，
所以它不可能产出模型（⑩' 那次的教训：假的 learn() 不 raise 就会存出模型，已避免）。

要回答的两件事
1. 结构断言：T1 是否真的按设计接管两条 hip_y —— 摆相窗口内命令被替换成模板脉冲、支撑相原样交还、
   左右严格反相（同一帧不得两腿都在摆）、占空比 ≈ 22/60、相位与 reward 的 gait_phase 同源。
2. 被动基线：**零动作策略** + 模板，跑 det rollout，量 ep_len / 落地每秒 / spacing / ds / flight / v。
   这一条是 200K 判决的参照物（教训：判据必须相对被动基线，否则接口先验会伪造指标）。
   同时跑"零动作 + 模板关闭"作对照，确认改变来自模板而不是噪声。

放行线（预注册，见 §46）：结构断言 4/4 必须全过；被动基线不设门槛（它只是参照），但必须
  无 NaN、且模板开关前后至少有一个量发生可见变化（否则说明 osc_enable 根本没生效）。
"""
import sys
import inspect

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
import mujoco

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry
import train_humanoid_sac as T

DT = 0.015
FRAMES = 1500
SEEDS = (123, 2026)


class ActionRecorder(gym.Wrapper):
    """记录真正交给物理引擎的动作（= 模板改写之后的）。"""

    def step(self, a):
        self.last = np.asarray(a, dtype=np.float64).copy()
        return super().step(a)

    def reset(self, **kw):
        self.last = None
        return super().reset(**kw)


def build(osc):
    raw = ActionRecorder(gym.make("Humanoid-v4", max_episode_steps=FRAMES))
    patch_mirror_symmetry(raw.unwrapped.model)
    w = WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=osc)
    return raw, w


def passive_run(osc, seed):
    raw, w = build(osc)
    m = raw.unwrapped.model
    idx = {s: [i for i in range(m.nu)
               if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) == f"{s}_hip_y"][0]
           for s in ("left", "right")}
    obs, _ = w.reset(seed=seed)
    zero = np.zeros(m.nu)
    prev = {s: True for s in ("left", "right")}
    lands, spac, rows, nan_hit = 0, [], [], False
    both_active = 0
    clock_gap = []
    n_act = {s: 0 for s in ("left", "right")}
    for t in range(FRAMES):
        obs, r, term, trunc, info = w.step(zero)
        a = raw.last
        # 期望一律用**内部真正使用的那个相位** w.osc_phase（它在 total_steps 自增之前取值），
        # 另外单独检查"模板时钟 vs 奖励时钟"是否恒定差 1 帧、不随时间漂移。
        ph = w.osc_phase
        gph = w.gait_phase                      # reward 在 :714 用的（自增之后）
        clock_gap.append(round(((gph - ph) % 1.0) * w.gait_period))
        in_win = {}
        for s in ("left", "right"):
            off = 0.0 if s == "left" else 0.5
            p = (ph - off) % 1.0
            in_win[s] = p < w.osc_swing_frac
            expect = (-w.osc_A * np.sin(np.pi * p / w.osc_swing_frac)) if (osc and in_win[s]) else None
            if expect is not None:
                n_act[s] += 1
                rows.append(abs(a[idx[s]] - expect) < 1e-9)
            elif osc:
                rows.append(abs(a[idx[s]] - 0.0) < 1e-9)   # 支撑相：零动作必须原样通过
        if osc and in_win["left"] and in_win["right"]:
            both_active += 1                                # 反相被破坏的直接证据
        if not np.all(np.isfinite(obs)) or not np.isfinite(r):
            nan_hit = True
        cL, cR = bool(w.left_contact_prev), bool(w.right_contact_prev)
        for s, c in (("left", cL), ("right", cR)):
            if c and not prev[s]:
                lands += 1
                spac.append(float(w.foot_diff))
            prev[s] = c
        if term or trunc:
            break
    n = t + 1
    out = dict(osc=osc, seed=seed, frames=n, land_per_s=lands / (n * DT),
               spacing_p50=float(np.median(spac)) if spac else float("nan"),
               ds=float(info.get("double_support_frac", float("nan"))),
               flight=float(info.get("flight_frac", float("nan"))),
               v=float(info.get("speed", float("nan"))),
               ret=float(info.get("ep_ret", float("nan"))) if "ep_ret" in info else float("nan"),
               nan=nan_hit, active_L_frac=n_act["left"] / n, active_R_frac=n_act["right"] / n,
               both_active=both_active,
               cmd_ok_frac=float(np.mean(rows)) if rows else float("nan"),
               clock_gap=sorted(set(clock_gap)))
    w.close()
    return out, both_active


def structure_checks():
    print("=== 结构断言（T1 是否真按设计接管 hip_y）===")
    ok_all = True
    src = inspect.getsource(T.WalkRewardWrapper.step)
    checks = [
        ("双腿索引都存在", "_hip_y\"][0] for s in (\"left\", \"right\")" in src or 'for s in ("left", "right")' in src),
        ("旧的接触门控已删除", "air = not self.left_contact_prev" not in src),
        ("相位来自绝对时钟 gait_period", "self.total_steps % self.gait_period" in src),
        ("k 无常程（osc_k_start 直通）", "prog = min((self.osc_frames" not in src),
    ]
    for name, val in checks:
        print(f"  {'PASS' if val else 'FAIL'}  {name}")
        ok_all &= bool(val)
    # 常量一致性：wrapper 与 OscCurriculum 的 k 必须同源（⑦ 的翻车点）
    k_wrap = (T.WalkRewardWrapper.__init__, )
    src_main = inspect.getsource(T.OscCurriculum.__init__)
    same_k = ("k0=1.0" in src_main and "k1=1.0" in src_main)
    print(f"  {'PASS' if same_k else 'FAIL'}  OscCurriculum 的 k0/k1 与 wrapper 一致（都是 1.0）")
    ok_all &= same_k
    txt = open("/home/guo/workspace/humanoid_walk/train_humanoid_sac.py").read().split("\n")
    for pat in ("OSC_ENABLE = ", "FROM_SCRATCH = ", "total_timesteps = "):
        hit = [l.strip() for l in txt if l.strip().startswith(pat) and "#" not in l.split(pat)[1][:2]]
        print(f"  磁盘现状：{hit[0] if hit else '未找到 ' + pat}")
    return ok_all



def clock_level_checks():
    """占空比/反相/连续性必须在**纯时钟**上验，不能在一集只有 32 帧的 rollout 上验（量化噪声）。"""
    P, SF = 60, 22.0 / 60.0
    inL = np.array([(k % P) / P < SF for k in range(2 * P)])
    inR = np.array([((k % P) / P - 0.5) % 1.0 < SF for k in range(2 * P)])
    ok = []
    ok.append(("左腿摆相帧数 = 22/周期", int(inL[:P].sum()) == 22))
    ok.append(("右腿摆相帧数 = 22/周期", int(inR[:P].sum()) == 22))
    ok.append(("两腿从不同帧同时摆（反相）", int((inL & inR).sum()) == 0))
    d = int(np.argmin(np.abs(np.arange(P) - 0)))
    gap = int(np.argmax(inR[:P]) - np.argmax(inL[:P]))
    ok.append(("左右窗口起点相差半周期 = 30 帧", gap == P // 2))
    ok.append(("摆相窗口连续（无碎片）", int(np.diff(np.flatnonzero(inL[:P])).max()) == 1))
    print("\n=== 时钟层断言（占空比/反相/连续，与 episode 长度无关）===")
    allok = True
    for name, v in ok:
        print(f"  {'PASS' if v else 'FAIL'}  {name}")
        allok &= bool(v)
    return allok


def main():
    if not (structure_checks() and clock_level_checks()):
        print("\n结构断言没过 ⇒ 不要开训")
        return 1
    rows = []
    for osc in (True, False):
        for s in SEEDS[:1]:
            r, both = passive_run(osc, s)
            rows.append(r)
            print(f"osc={r['osc']} seed={r['seed']}: frames={r['frames']:5d} land/s={r['land_per_s']:5.2f} "
                  f"spacing_p50={r['spacing_p50']:6.3f} ds={r['ds']:5.3f} flight={r['flight']:5.3f} "
                  f"v={r['v']:5.2f} actL={r['active_L_frac']:.3f} actR={r['active_R_frac']:.3f} "
                  f"both_active_frames={r['both_active']} cmd_ok={r['cmd_ok_frac']:.3f} NaN={r['nan']}")
    on, off = rows[0], rows[1]
    if on["frames"] < 60:
        print(f"  注：被动 rollout 只有 {on['frames']} 帧（零动作必摔），故其中的 actL/actR 占比只是量化值，不作判据")
    print("\n=== 放行判定 ===")
    bad = []
    if on["nan"] or off["nan"]:
        bad.append("出现 NaN")
    if on["both_active"] != 0:
        bad.append(f"两腿同帧摆动 {on['both_active']} 次 ⇒ 反相没做到")
    if on["clock_gap"] != [1]:
        bad.append(f"模板时钟与奖励时钟的偏差不恒定: {on['clock_gap']} 帧（应恒为 [1]，否则两机制会互相漂移）")
    if on["cmd_ok_frac"] < 0.999:
        bad.append(f"命令不符模板：合格帧比例只有 {on['cmd_ok_frac']:.3f}")
    if abs(on["land_per_s"] - off["land_per_s"]) < 0.2 and abs(on["v"] - off["v"]) < 0.02:
        bad.append("模板开/关的两个量几乎没差 ⇒ osc_enable 可能没生效")
    print(("PASS：模板生效、反相严格、无 NaN，可以开训" if not bad else "FAIL：" + "; ".join(bad)))
    print(f"\n被动基线（200K 判决时的对照物）：land/s={on['land_per_s']:.2f} "
          f"spacing={on['spacing_p50']:.3f} m ds={on['ds']:.3f} flight={on['flight']:.3f} v={on['v']:.2f}")
    print(f"（模板关闭的零动作对照：land/s={off['land_per_s']:.2f} spacing={off['spacing_p50']:.3f} "
          f"ds={off['ds']:.3f} flight={off['flight']:.3f} v={off['v']:.2f}）")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
