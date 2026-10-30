"""批次A 日志路径端到端测试：用真实 wrapper 的 info 跑 LegActivityCallback + logger.dump()。

record() 改成 record_mean() 后，任何非 float 的 info 值都会在 dump 时才炸；这里在不启动训练
的前提下把这条路径完整走一遍（含 ev_* 通用循环），并打印若干窗口均值确认语义。
"""
import io
import sys
import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

from stable_baselines3.common.logger import Logger, HumanOutputFormat
from probe_left_leg_chain import build_env, load_model
from train_humanoid_sac import LegActivityCallback


def main():
    vec = build_env()
    model = load_model(vec)
    obs = vec.reset()
    infos = []
    for _ in range(200):
        act, _ = model.predict(obs, deterministic=True)
        obs, rew, done, info = vec.step(act)
        infos.extend(info)          # locals["infos"] 是"每 env 一个 dict"的扁平列表
        if done[0]:
            break

    # 真实 logger：写进内存流，dump 时若类型不对会当场抛
    _sio = io.StringIO()
    logger = Logger(folder=None, output_formats=[HumanOutputFormat(_sio)])
    cb = LegActivityCallback()
    model._logger = logger          # SB3 2.9: logger 是只读 property，底层字段是 _logger
    cb.init_callback(model)
    cb.locals = {"infos": infos, "actions": np.zeros((len(infos), 17))}
    ok = cb._on_step()
    print(f"_on_step 返回 continue_training={ok}")
    # 模拟训练里的 obs/policy 访问路径（n_calls%100 分支已在上面命中或跳过）
    cb.n_calls = 100
    cb.locals = {"infos": infos, "actions": np.zeros((len(infos), 17))}
    try:
        cb._on_step()
        print("n_calls=100 分支（action_std/exploration_std 抽样）也通过")
    except Exception as e:
        print(f"✗ 100 步分支异常: {type(e).__name__}: {e}")

    snapshot = dict(logger.name_to_value)   # dump() 会清空 name_to_value，先取快照
    logger.dump(step=200)
    names = list(snapshot)
    gait = sorted(n for n in names if n.startswith("gait/"))
    ev = sorted(n for n in gait if n.startswith("gait/ev_"))
    print(f"\ndump 成功，共 {len(names)} 键，gait/ {len(gait)} 个，其中 ev_ 事件量 {len(ev)} 个")
    for k in ["gait/reach_reward", "gait/slip_penalty", "gait/flight_frac",
              "gait/double_support_frac", "gait/last_reach_L", "gait/last_reach_R",
              "gait/last_left_stride", "gait/forward_reward"]:
        if k in snapshot:
            v = snapshot[k]
            print(f"  {k:32s} = {v:.4f}" if isinstance(v, (int, float, np.floating)) else f"  {k} = {v}")
    bad = [k for k, v in snapshot.items()
           if isinstance(v, (list, tuple, dict, np.ndarray))]
    print(f"\n非标量残留(应为空): {bad}")
    print("record_mean 窗口均值样例: "
          f"ev_reach_L_mean={snapshot.get('gait/ev_reach_L_mean', '缺失')} "
          f"n={snapshot.get('gait/ev_reach_L_n', '缺失')}")
    # 【批次①+②】新键必须真的落到 logger 里（缺一个就是记账漏项）
    need = ["gait/walk_gate", "gait/single_frac", "gait/survival_bonus",
            "gait/ds_landing_rate", "gait/flat_gate", "gait/dense_ds_factor",
            "gait/ev_spacing_L_mean", "gait/ev_spacing_L_negfrac",
            "gait/ev_spacing_R_mean", "gait/ev_step_adv_L_mean", "gait/ev_reach_L_mean",
            "gait/swing_min_reward", "gait/last_rom_L_deg", "gait/last_left_stride",
            "gait/osc_active", "gait/place_operand"]
    miss = [k for k in need if k not in snapshot]
    print(f"批次①+②+⑤(丙) 新键 {len(need)-len(miss)}/{len(need)} 到位: "
          f"{'✓ 全部' if not miss else '✗ 缺 ' + str(miss)}")
    for k in ("gait/walk_gate", "gait/single_frac", "gait/survival_bonus", "gait/ds_landing_rate"):
        if k in snapshot:
            print(f"   {k:26s} = {snapshot[k]}")
    vec.close()


if __name__ == "__main__":
    main()
