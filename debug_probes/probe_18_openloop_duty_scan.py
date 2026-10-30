"""⑰ 决策前的第 1 件测量：开环（零动作 + 模板）能不能造出双支撑。

为什么必须先量它：我提的 ⑰(i) 是把 `stride_bonus`/`position_swap` 的结算搬回"另一脚仍在支撑"
那一桶。人在走的时候 ds≈0.27、flight=0.000（CMU 9-clip 实测），所以 ds 是**唯一能把走路和
弹跳分开的观测量**。但 ⑯ 末段实测 ds=0.0037、flight=0.406、`ev_spacing_*_ds_n` 整列 NaN，
每腿 duty 只有 0.30 ⇒ 如果这个装置在模板驱动下 duty 根本上不去，那"按 ds 付款"就是把钱全部
归零、策略饿死，⑰(i) 直接不可采纳，该改的是装置而不是结算线。

变量只有两个，都是零风险的（不动权重、不动价格、只读）：
  · osc_A ∈ {0, 0.15, 0.25, 0.35}
  · 双腾空仲裁 ∈ {现状"谁都不驱动" (:1325) / 驱动腾空更久的那条腿}

判据在看到任何数字**之前**钉死（人参照：ds 0.27 / flight 0.000 / 2.13 landings/s）：
  PASS = ds_frac ≥ 0.10 且 flight_frac ≤ 0.10 且 存活 ≥ 500 帧 且 v ≥ 0.30
  有任何一格 PASS ⇒ ⑰(i) 可采纳；全灭 ⇒ ⑰(i) 不可采纳，先修装置。

接触口径一律用 wrapper 自己的 `_foot_contact_state`（memory 里记过：用"任一 foot geom 碰 world"
这套近邻判据会把 rom 从 11.2° 吹成 18.3°，两条判据不一致时以 wrapper 为准）。
"""
import importlib.util
import os
import sys

import numpy as np
import gymnasium as gym

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")

SRC = "train_humanoid_sac.py"
txt = open(SRC, encoding="utf-8").read()
# 变体 B：把 :1325 的"双腾空谁都不驱动"换成"驱动腾空更久的那条腿"。
# 只替换这一句，其余字节相同 ⇒ 任何差异都能归因到这条仲裁规则。
ANCHOR = 'want["left"] = want["right"] = False'
assert txt.count(ANCHOR) == 1, f"锚点出现 {txt.count(ANCHOR)} 次，先去核对代码再谈替换"
VAR_B = ('want["left"], want["right"] = '
         '(self._osc_air["left"] >= self._osc_air["right"]), '
         '(self._osc_air["right"] > self._osc_air["left"])')
VAR_B_PATH = "/tmp/train_b18_drive_both.py"
open(VAR_B_PATH, "w", encoding="utf-8").write(txt.replace(ANCHOR, VAR_B, 1))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


W_A = load(SRC, "b18_current")            # 现状：双腾空不驱动
W_B = load(VAR_B_PATH, "b18_drive_both")  # 变体：双腾空驱动腾空更久的那条
SEEDS = tuple(int(x) for x in os.environ.get("S18_SEEDS", "123,2026,7,31,55,77").split(","))
FR = int(os.environ.get("S18_FRAMES", "1500"))


def build(W, osc_A):
    raw = gym.make("Humanoid-v4", max_episode_steps=FR)
    W.patch_mirror_symmetry(raw.unwrapped.model)
    w = W.WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=(osc_A > 0))
    w.osc_A = osc_A
    return w, raw.unwrapped.model.opt.timestep * raw.unwrapped.frame_skip


def run(W, osc_A, seed):
    """零动作 + 模板开环：一遍循环同时取每腿 duty、ds/腾空占比、落地频率、净前进速度。"""
    w, dt = build(W, osc_A)
    w.reset(seed=seed)
    m = w.unwrapped.model
    hip_x = m.body("left_thigh").pos[0]      # 仅作前进参考，下面用 qpos[0]（根 x）
    n = ds = fl = sl = land = cL = cR = 0
    hit_prev = (False, False)
    x_prev = w.unwrapped.data.qpos[0]
    v_sum = 0.0
    for _ in range(FR):
        _, _, ter, tru, _ = w.step(np.zeros(w.action_space.shape, dtype=np.float32))
        if ter or tru:
            break                                  # 摔倒/截断：存活帧数本身就是读数
        hitL, hitR, _, _ = w._foot_contact_state(m, w.unwrapped.data)
        if hitL and hitR:
            ds += 1
        elif not hitL and not hitR:
            fl += 1
        else:
            sl += 1
        if not hit_prev[0] and hitL:
            land += 1
        if not hit_prev[1] and hitR:
            land += 1
        hit_prev = (hitL, hitR)
        cL += int(hitL)
        cR += int(hitR)
        n += 1
        v_sum += (w.unwrapped.data.qpos[0] - x_prev) / dt
        x_prev = w.unwrapped.data.qpos[0]
    return dict(n=n, duty_L=cL / max(n, 1), duty_R=cR / max(n, 1),
                ds_frac=ds / max(n, 1), flight_frac=fl / max(n, 1), single=sl / max(n, 1),
                land_s=land / max(n * dt, 1e-9), v=v_sum / max(n, 1), dt=dt)


probe = run(W_A, 0.15, SEEDS[0])
print(f"帧长 dt = {probe['dt']:.4f} s ⇒ 1 Hz 对应 {1/probe['dt']:.1f} 帧"
      f"；wrapper :1731 的步频折扣用的是 0.015 s/帧 ⇒ 落地频率被高估 "
      f"{0.015 / probe['dt']:.2f}×（这是条欠账，等下单独核）")

# ---------- 基线复现：先用同一个接触判据跑出 ⑯ 策略，看能否对上 eval jsonl ----------
import json
from stable_baselines3 import SAC
import gc
MODEL = os.environ.get("PROBE_MODEL", "best_model/best_model.zip")
model = SAC.load(MODEL)
# load 会把 9.14 GB 的 replay buffer 也读回来（本机只剩 9.13 GB）；探针只用 policy，立刻释放。
del model.replay_buffer
gc.collect()
rows = [json.loads(l) for l in open("eval_logs/eval_gait.jsonl", encoding="utf-8") if l.strip()]
rows = [r for r in rows if r.get("b16")]
tail = rows[-270:] if len(rows) >= 270 else rows
jmed = lambda k: float(np.nanmedian([r.get(k, np.nan) for r in tail]))


def run_policy(m_, seed):
    w, dt = build(W_A, 0.15)                       # 模板按现状开着（和训练/eval 同一套 k/A）
    w.reset(seed=seed)
    mm = w.unwrapped.model
    n = ds = fl = land = cL = cR = 0
    hit_prev = (False, False)
    for _ in range(FR):
        a, _ = m_.predict(w.unwrapped._get_obs(), deterministic=True)
        _, _, ter, tru, _ = w.step(a)
        if ter or tru:
            break
        hitL, hitR, _, _ = w._foot_contact_state(mm, w.unwrapped.data)
        ds += int(hitL and hitR)
        fl += int((not hitL) and (not hitR))
        land += int((not hit_prev[0] and hitL) + (not hit_prev[1] and hitR))
        hit_prev = (hitL, hitR)
        cL += int(hitL); cR += int(hitR); n += 1
    return dict(n=n, ds=ds / max(n, 1), fl=fl / max(n, 1),
                land_s=land / max(n * dt, 1e-9), duty=(cL / max(n, 1) + cR / max(n, 1)) / 2)


pr = [run_policy(model, s) for s in SEEDS[:4]]
print(f"\n=== 基线复现（同一接触判据跑 ⑯ 策略 det，{len(pr)} 个 seed 中位）vs eval jsonl 末 {len(tail)} 集 ===")
print(f"  ds      探针 {np.median([r['ds'] for r in pr]):.4f}   jsonl(末200帧窗) {jmed('double_support_frac'):.4f}")
print(f"  flight  探针 {np.median([r['fl'] for r in pr]):.4f}   jsonl(末200帧窗) {jmed('flight_frac'):.4f}")
print(f"  land/s  探针 {np.median([r['land_s'] for r in pr]):.2f}   jsonl          {jmed('land_per_s'):.2f}")
print(f"  duty    探针 {np.median([r['duty'] for r in pr]):.3f}   jsonl          "
      f"{(jmed('stance_frac_L') + jmed('stance_frac_R')) / 2:.3f}")
print(f"  存活帧  探针 {np.median([r['n'] for r in pr]):.0f}   jsonl ep_len     {jmed('ep_len'):.0f}")
print("  注：jsonl 的 ds/flight 是**末尾 200 帧滑窗**，探针是整集占比；稳态下两者应当同量级，"
      "对不上就是判据或帧长错了，先别读下面的表。")


print(f"\n=== 开环（action≡0）支撑结构扫描，{len(SEEDS)} 个 seed 的中位数 ===")
print(f"{'驱动定律':22s} {'A':>5s} {'存活帧':>7s} {'dutyL':>6s} {'dutyR':>6s} {'ds':>6s} "
      f"{'腾空':>6s} {'land/s':>7s} {'v':>6s} {'判定':>5s}")
for W, name in ((W_A, "现状:双腾空不驱动"), (W_B, "变体:双腾空驱更久腿")):
    for a in (0.0, 0.15, 0.25, 0.35):
        rs = [run(W, a, s) for s in SEEDS]
        med = lambda k: float(np.median([r[k] for r in rs]))
        ok = med("ds_frac") >= 0.10 and med("flight_frac") <= 0.10 and med("n") >= 500 and med("v") >= 0.30
        print(f"{name:22s} {a:5.2f} {med('n'):7.0f} {med('duty_L'):6.3f} {med('duty_R'):6.3f} "
              f"{med('ds_frac'):6.3f} {med('flight_frac'):6.3f} {med('land_s'):7.2f} "
              f"{med('v'):6.3f} {'PASS' if ok else 'fail':>5s}"
              f"   逐seed存活={[int(r['n']) for r in rs]}")
print("\n人参照（CMU 9-clip 稳态直线窗）：ds 0.27、flight 0.000、2.13 landings/s、单步 0.393 m")
