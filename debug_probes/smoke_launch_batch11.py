"""⑪ 的开训前冒烟：**用真实入口跑完整启动路径**，但把 `SAC.learn` 换成"打印状态后立即 raise"。

为什么必须这样测（教训：⑦ 时代 smoke_test 的假 learn 没 raise ⇒ 脚本继续走到 model.save() 存出模型，
已把那条规矩写进项目记忆）：raise 保证不会训练一秒、不会写任何权重。

要确认的五件事：
 1) 启动路径不崩（envs/vec 子进程/回调/镜像补丁/缓冲区分配都在真实内存压力下走通）
 2) 模板真的开在**训练 env 和评测 env 两边**，且 k=1.0（⑦ 的翻车点是两边不一致）
 3) FROM_SCRATCH 生效：learning_starts=100000、不读 best_model.zip、buffer 起点为空
 4) total_timesteps=4_000_000 与回调数量
 5) 跑完之后盘上关键文件没被改（best_model.zip 仍不存在、npz/jsonl mtime 不变）
"""
import os
import runpy
import sys
import time

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
ROOT = "/home/guo/workspace/humanoid_walk"
WATCH = [os.path.join(ROOT, "eval_logs/evaluations.npz"),
         os.path.join(ROOT, "eval_logs/eval_gait.jsonl"),
         os.path.join(ROOT, "sac_humanoid_walk_final.zip"),
         os.path.join(ROOT, "best_model/best_model.zip")]
before = {p: (os.path.getmtime(p), os.path.getsize(p)) if os.path.exists(p) else None for p in WATCH}

from stable_baselines3 import SAC

REPORT = {}


def boom(self, *a, **kw):
    import numpy as np

    def probe(name, fn):
        try:
            REPORT[name] = fn()
        except Exception as e:
            REPORT[name] = f"读取失败 {type(e).__name__}: {e}"

    def attrs(key, cast):
        v = np.asarray(self.env.get_attr(key)).reshape(-1)
        return sorted(set(cast(x) for x in v.tolist()))

    cb = kw.get("callback")
    REPORT["total_timesteps"] = kw.get("total_timesteps", a[0] if a else None)
    REPORT["n_callbacks"] = len(cb) if hasattr(cb, "__len__") else type(cb).__name__
    probe("learning_starts", lambda: int(self.learning_starts))
    probe("buffer_size", lambda: int(self.replay_buffer.buffer_size))
    probe("buffer_pos(应为0)", lambda: int(self.replay_buffer.pos))
    probe("train osc_enable", lambda: attrs("osc_enable", bool))
    probe("train osc_A", lambda: attrs("osc_A", float))
    probe("train swing_frac", lambda: attrs("osc_swing_frac", lambda x: round(float(x), 5)))
    probe("train k_start", lambda: attrs("osc_k_start", float))
    probe("train k_forced", lambda: attrs("osc_k_forced", lambda x: None if x is None else round(float(x), 3)))
    probe("train gait_period", lambda: attrs("gait_period", int))
    probe("train action_dim", lambda: attrs("action_space", lambda x: int(np.asarray(x).shape[0]))[:1])
    probe("eval_env osc 一致", lambda: sorted(set(
        bool(np.asarray(e.unwrapped.get_attr("osc_enable")).reshape(-1)[0])
        for e in [self._eval_env] if e is not None)) if getattr(self, "_eval_env", None) else "无(回调里持有)")
    print("\n[SMOKE] 到达 learn()，训练 env 实况：", flush=True)
    for k, v in REPORT.items():
        print(f"   {k:24s} = {v}", flush=True)
    raise SystemExit(0)   # ← 必须 raise：否则会继续往下走 model.save() 存出模型


SAC.learn = boom

os.chdir(ROOT)
try:
    runpy.run_path(os.path.join(ROOT, "train_humanoid_sac.py"), run_name="__main__")
except SystemExit as e:
    print(f"[SMOKE] 按预期在 learn() 处退出（code={e.code}）", flush=True)

print("\n=== 盘上关键文件是否被动过 ===")
bad = []
for p in WATCH:
    now = (os.path.getmtime(p), os.path.getsize(p)) if os.path.exists(p) else None
    tag = "未变" if now == before[p] else "**变了**"
    if now != before[p]:
        bad.append(p)
    print(f"  {os.path.relpath(p, ROOT):38s} {tag}  现在={'不存在' if now is None else str(now[1]) + 'B'}")
fails = [k for k, v in REPORT.items() if isinstance(v, str) and v.startswith("读取失败")]
ok_osc = REPORT.get("train osc_enable") == [True] and REPORT.get("train k_start") == [1.0]
print("\n结论：" + ("启动路径干净、模板在训练 env 开着且 k=1.0、FROM_SCRATCH 生效 ⇒ **可以开训**"
      if (not bad and not fails and ok_osc and REPORT.get("buffer_pos(应为0)") == 0)
      else f"**先别开训** 未读数={fails} 模板开关={REPORT.get('train osc_enable')} k={REPORT.get('train k_start')}"))
