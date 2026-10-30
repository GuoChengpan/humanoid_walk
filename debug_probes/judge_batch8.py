"""批次⑧ 的判据读数（明早直接跑这个）。

判据原文写在 CONTEXT_HANDOFF §34.4，2026-10-12 01:5x **观察前钉死**，本脚本只是把它机器化：
  首判 300K：K1+K2+K3+K4 全过才算这批成立；复判 500K 要求 K1/K2 **仍在**（出现过又消失 = 不算过）。
  K1 是主判据，且历史上从未发生过：eval_gait.jsonl 全库 315 集里 ds≥0.03 的 17 集**跑满率 0%**。
⚠ `eval_logs/eval_gait.jsonl` 是 **append-only**、且行内**没有 step 字段**，所以"本轮的行"用
  `--since <开跑时的行数>` 指定（开跑前是 315）。默认 315；若你中途又跑过别的轮次，改这个数。
"""
import argparse
import glob
import json
import os
import re

import numpy as np
from tensorboard.backend.event_processing import event_accumulator

ROOT = "/home/guo/workspace/humanoid_walk/"
JSONL = ROOT + "eval_logs/eval_gait.jsonl"
TB_GLOB = ROOT + "sac_humanoid_walk/*/events.out.tfevents.*"
PREV_RUN_EPOCH = 1791795721          # ⑩ 于 17:01 开跑（PID 244641）⇒ 比它新的 events 才是本轮；比它旧的就别读
# 批次⑨：预算 4M、随机初始化 ⇒ 没有'新基线分数'可用，K3 改用绝对 sanity 值 1200；首判点 1M。
NEW_BASELINE, NEW_TOTAL = 1500.0, 1_000_000
K1_EP, K1_DS, K1_ROM, K1_V = 1499.0, 0.03, 6.0, 0.7
K2_ROM, K3_EVAL, K4_LO, K4_HI = 8.0, NEW_BASELINE * 0.8, 0.60, 0.85


def latest_tb():
    cands = []
    for p in glob.glob(TB_GLOB):
        m = re.search(r"tfevents\.(\d+)\.", os.path.basename(p))
        if m and int(m.group(1)) >= PREV_RUN_EPOCH:   # >= 才包含本轮自己（epoch 相等）
            cands.append((int(m.group(1)), p))
    return max(cands)[1] if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=0, help="开跑时 eval_gait.jsonl 已存在的行数（⑨=新建文件，取 0）")
    ap.add_argument("--jsonl", default=JSONL, help="可指向归档副本做回归验证")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.jsonl) if l.strip()][args.since:]
    n_eval = len(rows) // 5
    print("本轮 eval 集：%d 行 = %d 次 eval（每次 5 集）｜开跑行数偏移 --since=%d" % (len(rows), n_eval, args.since))
    if n_eval == 0:
        print("还没有本轮的 eval 数据。")
        return
    def col(key):
        # 随机预热期的集只有十几步、一次落地都没有 ⇒ 事件类 ev_* 键根本不会写进 jsonl。
        # 缺失一律按 NaN 处理（NaN 参与 >= 比较得 False，正好让 K1 不会把"没数据"当成"达标"）。
        return np.array([float(r.get(key, float("nan"))) for r in rows])
    ep, ds = col("ep_len"), col("double_support_frac")
    rom, romR = col("ev_rom_L_mean"), col("ev_rom_R_mean")
    v, ret, flight = col("speed"), col("ret"), col("flight_frac")
    n_ev = int(np.isfinite(rom).sum())
    if n_ev < len(rows):
        print("   （其中 %d/%d 集缺 ev_* 事件统计 = 集内一次落地都没有，按不达标处理）" % (len(rows) - n_ev, len(rows)))

    q = (ep >= K1_EP) & (ds >= K1_DS) & (rom >= K1_ROM) & (v >= K1_V)
    print("\nK1 主判据（ep_len≥1499 且 ds≥0.03 且 rom_L≥6° 且 speed≥0.7，四个缺一不可）")
    print("   达标集数 = **%d**（需 ≥2）" % int(q.sum()))
    if q.sum():
        for i in np.where(q)[0][:8]:
            print("     第%d集：ep_len=%.0f ds=%.3f rom_L=%.2f° v=%.2f ret=%.0f flight=%.3f" % (
                i, ep[i], ds[i], rom[i], v[i], ret[i], flight[i]))
    # 部分达标（用来区分"匍匐"与"走路"）：只看 ds+存活
    p = (ep >= K1_EP) & (ds >= K1_DS)
    print("   仅 ds≥0.03 且跑满的集数 = %d（若这些集 rom_L<6° 或 v<0.7 ⇒ 是匍匐不是走路，见 §34.3）" % int(p.sum()))

    last15 = rom[-15:] if len(rom) >= 15 else rom
    k2 = float(np.nanmean(last15[-(3 * 5):])) if len(rom) >= 15 else float("nan")
    k3 = float(np.nanmean(ret[-5:]))
    print("\nK2 det rom_L 末 3 次 eval 均值 = %.2f°（需 ≥%.1f°，起点 1.78°）⇒ %s" % (
        k2, K2_ROM, "过" if k2 >= K2_ROM else "不过"))
    ratio = rom / np.maximum(romR, 1e-9)
    k2b = float(np.nanmean(ratio[-15:]))
    print("K2b det rom_L/rom_R 末 3 次 eval 均值 = %.3f（需 ≥0.35；弹跳实测 0.04-0.08，⑧ 末 0.076）⇒ %s" % (
        k2b, "过" if k2b >= 0.35 else "不过"))
    spL, spR = col("ev_spacing_L_p50"), col("ev_spacing_R_p50")
    nA, nB = col("ev_step_adv_L_n"), col("ev_step_adv_R_n")
    k5L, k5R = float(np.nanmean(spL[-15:])), float(np.nanmean(spR[-15:]))
    raw_cad = float(np.nanmean((nA[-15:] + nB[-15:]) / (0.02 * ep[-15:])))
    print("K5 det 两脚间距 p50：L=%.3f / R=%.3f m（需**都** ≥0.05；⑨ 末段 L=−0.002 ⇒ 这条就是反碎步）⇒ %s" % (
        k5L, k5R, "过" if (k5L >= 0.05 and k5R >= 0.05) else "不过"))
    print("K6(原始仪表) 步频 = %.1f 步/秒（去抖阈值 6.0 ⇔ 原始 6.6，⑩ 实测抖动灌水约 +8%%）⇒ %s" % (
        raw_cad, "过" if raw_cad <= 6.6 else "不过"))
    print("    v_net 不在这个仪表里，判定点请用 probe_stride_geometry.py 量净位移/存活时间（⑨ 末段 0.45 m/s，门槛 0.6）")
    print("K3 det eval 末次均值 = %.0f（需 ≥%.0f，绝对存活 sanity；⑧ 的 1183=新基线80%% 不适用于从头训）⇒ %s" % (
        k3, K3_EVAL, "过" if k3 >= K3_EVAL else "不过"))
    print("   末 5 次 eval 的 ret 均值序列：%s" % [int(x) for x in
          [np.mean(ret[i:i + 5]) for i in range(max(0, len(ret) - 25), len(ret) - 4, 5)]])

    k4, k4v, steps = None, float("nan"), None
    tb = latest_tb()
    if tb:
        ea = event_accumulator.EventAccumulator(tb, size_guidance={"scalars": 0})
        ea.Reload()
        tag = "gait/dense_ds_factor"
        if tag in ea.Tags()["scalars"]:
            s = ea.Scalars(tag)
            st = np.array([x.step for x in s], float)
            vv = np.array([x.value for x in s], float)
            m = st >= max(st.max() - 80_000, 0)
            k4v = float(np.mean(vv[m]))
            steps = int(st.max())
            print("\nTB 事件文件：%s｜最新 step=%dK" % (os.path.basename(tb), steps // 1000))
    else:
        print("\n⚠ 没找到比 ⑦（epoch %d）更新的 events 文件 ⇒ K4 无法读数。"
              "若你已把本轮 TB 目录移出 sac_humanoid_walk/，请把 PREV_RUN_EPOCH 调成本轮开跑 epoch。" % PREV_RUN_EPOCH)
    print("K4 开关 gait/dense_ds_factor 近 80K 窗均值 = %.3f（需落在 [%.2f, %.2f]）⇒ %s" % (
        k4v, K4_LO, K4_HI,
        "过（门控在发钱但双支撑还没出来）" if K4_LO <= k4v <= K4_HI else
        ("**离开上界 = 真的在双撑了**" if k4v > K4_HI else "低于下界，异常")))

    if steps is None or n_eval < 63:
        cur = (steps // 1000) if steps else 0
        print("\n>>> 未到首判点 1M（TB=%dK，本轮 eval %d 次；首判需 ≥63 次）：只报趋势，**不出判决**。"
              " 0-100K 是随机预热期（无梯度更新、集长十几步、ev_* 键缺失都属正常），红线 E 按 §37.4 延后到 200K 后才生效。"
              % (cur, n_eval))
        print("    早停保险：若 1M 时 det eval<1200 且 ep_len_mean<300 ⇒ 停，不必跑满 4M。")
    else:
        ok = [int(q.sum()) >= 2, k2 >= K2_ROM, k2b >= 0.35, (k5L >= 0.05 and k5R >= 0.05),
              raw_cad <= 6.6, k3 >= K3_EVAL, K4_LO <= k4v <= K4_HI or k4v > K4_HI]
        print("\n>>> 判决：K1=%s K2=%s K2b=%s K5=%s K6=%s K3=%s K4=%s ⇒ %s" % (
            ok[0], ok[1], ok[2], ok[3], ok[4], ok[5], ok[6],
            "**四条全过：这批成立，续到 500K 复判（必须仍在）**" if all(ok) else
            "不通过 ⇒ 按 §34.4 奖励线到此关闭，不再动任何奖励常数，转 A/B/C/D"))
    # 红线 E 按 §37.4 从 200K 之后才生效：从头训的 0-200K 本来就全是短集+负回报，早期触发是必然的、不算数。
    if steps is not None and steps < 200_000:
        print("（红线 E 现在**不生效**：§37.4 声明它从 200K 后才开始判，当前短集/负回报是随机预热期的正常态）")
    elif float(np.nanmean(ep[-5:])) < 200 and (ret[-15:] < 0).all():
        print("★红线 E 触发（≥200K 才生效）：ep_len 末5集<200 且末 15 集回报全<0 ⇒ 立刻停")


if __name__ == "__main__":
    main()
