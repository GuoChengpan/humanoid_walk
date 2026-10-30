"""按功能量筛历史快照（只读，不训练、不写模型）。

背景：best_model_snapshots 里的快照是按 eval_reward 存的，而奖励存在套利空间，
reward 更高不等于"落地交替/前进"更好。本探针用确定性策略把每个快照重新跑一遍，
与同归档的 best_model.zip 比功能量。

预注册判据（跑前钉死）：
  · 主量 min_adv = min(ev_step_adv_L_mean, ev_step_adv_R_mean)
  · 辅量 swap_rate = position_swap_land_count / total_land_count
  · 保留条件：min_adv >= best_min_adv + 0.01 m 且 swap_rate >= best_swap - 0.05
  · 其余读数（ds/flight/land_per_s/speed/ep_len）仅作旁证
仪器：1500 步、种子 123/2026、按归档世代配环境
  round1/** 与 round2/17_*  → 裸环境
  round2/SAC_18/**、last_TMP → 只加 patch_mirror_symmetry
  round2/SAC_19/19_1..19_4、best_model/ → patch + 当前 T1 wrapper
  round2/SAC_19/19_5 → patch + 它归档源码里的 T2 wrapper

用法：
  python3 debug_probes/screen_snapshots.py --workers 6            # 全量
  python3 debug_probes/screen_snapshots.py --limit 8 --workers 2  # 冒烟
输出：debug_probes/snapshot_screen.csv（追加写，含 archive/file/kind/seed/各项读数）
"""
import os, sys, glob, csv, json, argparse, importlib.util
import multiprocessing as mp

ROOT = "/home/guo/workspace/humanoid_walk"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
MAX_STEPS = 1500
SEEDS = (123, 2026)


def era_of(path):
    p = path.replace("\\", "/")
    if "/round1/" in p or "/round2/17_" in p:
        return "bare"
    if "/round2/SAC_19/19_5" in p:
        return "t2"
    if "/round2/SAC_19/" in p or p.startswith("best_model/"):
        return "t1"
    if "/round2/SAC_18/" in p or "/round2/last_TMP" in p:
        return "patch"
    return "bare"


def archive_of(path):
    d = os.path.dirname(path)
    if os.path.basename(d) == "best_model_snapshots":
        d = os.path.dirname(d)
    return d


def _archive_module(archive):
    """加载归档自己的训练源码（含它的 mirror_map），19_3 无源码时用 19_4 代理。"""
    d = os.path.join(ROOT, archive)
    src = os.path.join(d, "train_humanoid_sac.py")
    if not os.path.exists(src):
        if "19_3_" in archive:
            d = os.path.join(ROOT, "best_model_saved/round2/SAC_19/19_4_调权重_结果还是原地踏步速度慢了")
            src = os.path.join(d, "train_humanoid_sac.py")
        if not os.path.exists(src):
            return None
    for m in ("mirror_map", "archmod"):
        sys.modules.pop(m, None)
    sys.path.insert(0, ROOT)
    sys.path.insert(0, d)
    try:
        spec = importlib.util.spec_from_file_location("archmod", src)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        try: sys.path.remove(d)
        except ValueError: pass


def build_env(era, model_path, patch_fn, wrapper_cls):
    import gymnasium as gym
    env = gym.make("Humanoid-v4", max_episode_steps=MAX_STEPS)
    if era == "bare":
        return wrapper_cls(env, swing_drive_weight=2.0, osc_enable=False)
    mod = _archive_module(archive_of(model_path))
    if mod is not None and era in ("t1", "t2"):
        mod.patch_mirror_symmetry(env.unwrapped.model)
        return mod.WalkRewardWrapper(env, swing_drive_weight=2.0, osc_enable=True)
    # patch-era（18_x / last_TMP）：动力学用归档 patch，指标用当前 wrapper
    (mod.patch_mirror_symmetry if mod is not None else patch_fn)(env.unwrapped.model)
    return wrapper_cls(env, swing_drive_weight=2.0, osc_enable=False)


def run_file(args):
    path, kind = args
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import SAC
    from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry
    era = era_of(path)
    rows = []
    try:
        model = SAC.load(path, device="cpu", buffer_size=1)
        for seed in SEEDS:
            env = build_env(era, path, patch_mirror_symmetry, WalkRewardWrapper)
            obs, _ = env.reset(seed=seed)
            ret = 0.0; speeds = []
            for _ in range(MAX_STEPS):
                act, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(act)
                ret += float(r)
                speeds.append(float(env.unwrapped.data.qvel[0]))
                if term or trunc:
                    break
            steps = max(len(speeds), 1)
            land = getattr(env, "total_land_count", 0)
            swap = land and getattr(env, "position_swap_land_count", 0) / land
            row = dict(archive=os.path.relpath(archive_of(path), ROOT),
                       file=os.path.relpath(path, ROOT), kind=kind, era=era, seed=seed,
                       ep_len=steps, ret=ret, speed=float(np.mean(speeds)),
                       swap_rate=float(swap or 0.0),
                       ds=float(getattr(env, "double_support_frac", 0.0)),
                       flight=float(getattr(env, "flight_frac", 0.0)),
                       land_per_s=float(land / max(steps * 0.015, 1e-6)),
                       adv_L=float(info.get("ev_step_adv_L_mean", float("nan"))),
                       adv_R=float(info.get("ev_step_adv_R_mean", float("nan"))))
            row["min_adv"] = min(row["adv_L"], row["adv_R"])
            rows.append(row)
            env.close()
    except Exception as e:
        rows.append(dict(archive=os.path.relpath(archive_of(path), ROOT),
                         file=os.path.relpath(path, ROOT), kind=kind, era=era,
                         seed=-1, error=f"{type(e).__name__}: {e}"))
    return rows


def collect():
    tasks = []
    for bm in sorted(glob.glob("best_model_saved/**/best_model.zip", recursive=True)):
        tasks.append((bm, "best"))
        d = os.path.join(os.path.dirname(bm), "best_model_snapshots")
        for s in sorted(glob.glob(os.path.join(d, "*.zip"))):
            tasks.append((s, "snapshot"))
    if os.path.exists("best_model/best_model.zip"):
        tasks.append(("best_model/best_model.zip", "best"))
        for s in sorted(glob.glob("best_model/best_model_snapshots/*.zip")):
            tasks.append((s, "snapshot"))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="debug_probes/snapshot_screen.csv")
    a = ap.parse_args()
    tasks = collect()
    if a.limit:
        tasks = tasks[:a.limit]
    print(f"tasks={len(tasks)} workers={a.workers}", flush=True)
    fields = ["archive", "file", "kind", "era", "seed", "ep_len", "ret", "speed",
              "swap_rate", "ds", "flight", "land_per_s", "adv_L", "adv_R", "min_adv", "error"]
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        with mp.Pool(a.workers) as pool:
            for i, rows in enumerate(pool.imap_unordered(run_file, tasks), 1):
                for r in rows:
                    w.writerow({k: r.get(k, "") for k in fields})
                f.flush()
                if i % 20 == 0:
                    print(f"  {i}/{len(tasks)}", flush=True)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
