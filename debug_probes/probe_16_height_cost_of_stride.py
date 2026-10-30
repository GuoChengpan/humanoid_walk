"""用户的假设：**迈步 → 身高下降 → height 掉分 → 政策退回不降身高的原地踏步**。

停训后有内存了，直接在 ⑮ 权重上做 det rollout，逐帧取 qpos[2] 与两个步幅观测量，量三件事：
  V1 身高现在在帐篷的哪一侧（带下沿 1.245 还是带上沿 1.452）——决定它是"边际小量"还是"贴边"
  V2 **回归**：z 对瞬时步幅 `lo_held`(=min(左右落地间距)) 的斜率。这是假设的机制本体：
     如果分开两脚真的压身高，斜率应为负且量级 ≥ 几何预测 2.9cm/0.45m = 6.4 mm/米步幅
  V3 换算成分：slope × 5 分/米 × flat × ddf ⇒ 每米步幅被 height 收多少税；与 stride_bonus 的
     每米步幅买单（事前算 1.3 分/帧 @0.45m）并列，看谁赢。
同时报 z 在腾空相/双支撑相的分布（假设说"迈步导致降低"，最该出现在腾空期）。

⚠ 这不是轨迹因果实验：回归里"两脚分开的时刻"和"身体正在往下走"可以同源于第三件事。
   所以 V2 的负斜率只支持"机制在场"，不能单独判"它就是原地踏步的原因"——那要靠 V3 的量级对比。
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

DT, FR = 0.015, 1500
SEEDS = tuple(int(x) for x in os.environ.get("H_SEEDS", "123,2026,7,31,55,77,91,101,5,11").split(","))
MODEL = os.environ.get("PROBE_MODEL", "best_model/best_model.zip")
BAND_LO, BAND_HI, W = 1.05, 1.55, 1.0     # band = min((z-1.05)/0.20, (1.55-z)/0.10, 1)


def build():
    raw = gym.make("Humanoid-v4", max_episode_steps=FR)
    patch_mirror_symmetry(raw.unwrapped.model)
    return WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=True)


model = SAC.load(MODEL)
Z, LO, PH, FG, DD = [], [], [], [], []
z_air, z_ds, z_si = [], [], []
for sd in SEEDS:
    w = build()
    obs, _ = w.reset(seed=sd)
    while True:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = w.step(np.asarray(a, dtype=np.float64))
        d = w.unwrapped.data
        Z.append(float(d.qpos[2]))
        LO.append(float(info.get("lo_held", np.nan)))
        fl, ds = float(info.get("flight_frac", 0)), float(info.get("double_support_frac", 0))
        PH.append("air" if fl > 0.5 else ("ds" if ds > 0.5 else "single"))
        FG.append(float(info.get("flat_gate", 1.0)))
        DD.append(float(info.get("dense_ds_factor", 1.0)))
        (z_air if fl > 0.5 else z_ds if ds > 0.5 else z_si).append(float(d.qpos[2]))
        if term or trunc:
            break
Z, LO, FG, DD = map(np.asarray, (Z, LO, FG, DD))
ok = np.isfinite(LO)
print(f"{len(SEEDS)} 集 · {len(Z)} 帧  权重={MODEL}")

q = lambda a, p: float(np.percentile(a, p))
print(f"\nV1 身高分布：p10={q(Z,10):.3f} p50={q(Z,50):.3f} p90={q(Z,90):.3f} 最小={Z.min():.3f} 最大={Z.max():.3f}")
band = np.minimum((Z - BAND_LO) / 0.20, (BAND_HI - Z) / 0.10).clip(0, 1)
below = float(np.mean(Z < 1.25))
above = float(np.mean(Z > 1.45))
print(f"   band 均值={band.mean():.3f} ｜落在平顶区间 1.25–1.45 的帧={1 - below - above:.2f}"
      f"｜带下沿之下(z<1.25)={below:.2f}｜带上沿之上(z>1.45)={above:.2f}")
print(f"   ⇒ {'**贴带下沿：再蹲就立刻进斜率区，你的假设有利**' if below > 0.35 else ('贴带上沿：再高就立刻进斜率区' if above > 0.35 else '大平顶 inside ⇒ 身高变化暂时免费，假设不利')}")
print(f"   分相位身高：air p50={q(np.array(z_air),50) if z_air else float('nan'):.3f} "
      f"single p50={q(np.array(z_si),50) if z_si else float('nan'):.3f} "
      f"ds p50={q(np.array(z_ds),50) if z_ds else float('nan'):.3f}  "
      f"（假设说迈步压身高 ⇒ 应 air/single 明显低于 ds）")

x, y = LO[ok], Z[ok]
if len(x) > 200 and x.std() > 1e-6:
    sl, ic = np.polyfit(x, y, 1)
    resid = y - (sl * x + ic)
    r2 = 1 - resid.var() / y.var()
    print(f"\nV2 回归 z ~ min(左右落地间距)：斜率={sl*1000:+.1f} mm 身高 / 米步幅  R²={r2:.3f} n={len(x)}")
    print(f"   步幅分布：p10={q(x,10):.3f} p50={q(x,50):.3f} p90={q(x,90):.3f}")
    for b0, b1 in ((0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 9)):
        m = (x >= b0) & (x < b1)
        if m.sum() > 50:
            print(f"   间距∈[{b0:.2f},{b1:.2f}) 帧={m.sum():5d}  z 中位={q(y[m],50):.3f}  band 均值={q(band[ok][m],50):.3f}")
    print(f"   几何独立预测（定长摆 sqrt(L²−(d/2)²)，L≈0.9）：−6.4 mm/米 ⇒ 实测斜率与之同号且量级相当才算机制在场")

    v3 = sl * 5.0 * float(FG.mean()) * float(DD.mean())
    print(f"\nV3 换算成钱：每米步幅 height 收 {v3:+.4f} 分/帧 = {v3*45*DT*1000/1000:+.3f}"
          f"（步幅 0→0.45 m 的总代价 = {v3*0.45*1500:+.0f} 分/集）")
    print(f"   同一件事在 stride_bonus 上的买单（事前算，min=0.45 ⇒ 满额）= +1.3 分/帧 = +1950 分/集")
    print(f"   ⇒ 净激励 = {v3*1500*0.45:+.0f} + 1950 = **{1950 + v3*1500*0.45:+.0f} 分/集**"
          f"（为正 ⇒ 钱仍偏迈步，height 不是拦住它的那只手；为负 ⇒ 假设成立）")
