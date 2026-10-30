# Humanoid 交替步态训练 — 交接文档

> 最后更新：2026-09-20 上午（恢复重启中）。新会话接手：先读本文件，再分析最新日志。
> 日志目录：用户复用/删除 run 目录，**看 `sac_humanoid_walk/` 下 mtime 最新的 events 文件**（可能是 SAC_17 或 SAC_18）。

## 1. 任务与目标

SB3 2.9.0 SAC + MuJoCo Humanoid-v4（`train_humanoid_sac.py`，`WalkRewardWrapper`）。
核心目标：**真交替步态**（左右脚轮流当领脚），不是只前进。
用户靠渲染目测 + 落地序列数据判断，不以总分为准。

## 2. 当前状态（2026-09-20 上午）

**动作：从 576K 原版恢复重启**（用户执行 kill + cp + relaunch）：

```bash
cp "best_model/tmp_model_data/17_6/17_6_2.3+门控0.5/best_model_snapshots/best_model_576000.zip" best_model/best_model.zip
```

- 576K 原版是**唯一存活的健康检查点**（md5 ef855907… 已验证；新奖励全套下 deterministic eval ≈460 = 397.6 + lead_credit 底 37）
- warm-start 模式：fresh 构造器（LR 1e-5 固定）+ `old_model` 提权重 + prefill 500K（约 6 分钟静默，批量 predict）
- 首次 eval 预期 **≈460**、ep_len 1500；显著低于此 = 检查点恢复或加载有问题，先查 best_model.zip 的 num_timesteps

### 判据（新 run 沿用）

| 信号 | 含义 |
|---|---|
| `gait/lead_credit` 从基线 ~37 上涨 | **主指标**。过 ~200 = 左脚领先 10cm、交换解锁（swap_rate 应联动）；~570 = 全交替 |
| `gait/position_swap_rate` | 每 +0.0087 = 门控棘轮拿回 20 分；与 lead_credit 交叉验证 |
| eval ep_len 塌出 1500 | 躺平/摔倒哨兵 |
| `gait/flight_phase_penalty` 均值（基线 ~0.04）持续上涨 | 躲双支撑哨兵 |
| `gait/height_reward` 从 ~592 涨向 750 | 站姿梯度被吃（加分项，非必需） |
| 崩溃签名：**ep_len 连续钉 16-25 + eval ≈ −5x** | 立即停，见 §2.1 |

### 2.1 检查点事故记录（2026-09-20 凌晨，已定责）

- **事故链**：Run C（lead_credit 首跑）崩于 312K → 死前 best_model.zip 被覆盖成其 272K（eval 465）→ **助手误判"未被污染"并建议不回 576K** → Run D 装入后 16K 步死亡，2.25M 步全程 ep_len 16-25 → Run D 的 new-best 链把 best_model.zip 和 best_model_snapshots/ 里所有前几轮同号快照全部覆盖
- **根因**：EvalCallback 进程内最优从 −∞ 起步——**任何崩掉的 run 停止后，其第一次 eval（哪怕 −59）必然覆盖 best_model.zip**
- **定责**：用户文件管理流程完备（历史版本单独管理）；错在助手建议
- **固定流程（新增，必须遵守）**：**每次重启前必须确认/恢复 best_model.zip 的内容**（`读 zip 内 data 的 num_timesteps` 10 秒钟的事）
- 用户已决定不改 EvalCallback 路径（保留现工作流）
- **损失**：gate run（eval 529）、Run C（eval 465）检查点不可恢复；但代码层收获全部保留（height 修复、lead_credit 可达性证据、崩溃指纹）

### 2.2 崩溃指纹（已两次复现）

"正常走↔腾空模式"振荡放大 → 死亡：flight 在 ~6%↔38-40% 摆动，振幅扩大，末期 flight 80%+、speed 转负（后仰摔）、ep_len 16-25、eval −5x。交替机制指标（lead_credit/swap/stride_bonus）全程 ≈0，非直接推手，定性 SAC 策略/价值不稳定。
→ 处置：杀进程 → 恢复 last-known-good best_model.zip → 重启；若从 576K 起的新 run 再现同款螺旋，第一杠杆 **ent_coef 0.1→0.05**（会同时压 lead_credit 探索，权衡使用）。

## 3. reward 结构全貌（当前生效）

1. **stride bonus 真交换硬门控**：落地完成 `left_ahead↔right_ahead` 直连转移（`last_landing_swapped`）才给分；旧符号翻转软门控已删（170 次右脚 stride 98 次被微抖动骗过）
2. **stall_penalty 已退役**：只留监控。`gait/stall_max` 最灵敏；`gait/stall_penalty` 日志是落地帧采样伪影，勿读
3. **短步幅墙已拆**：双支撑落地落在前面(0<stride<0.2)不罚；`negative_stride`（落在后面）保留
4. **height_reward 平顶带 + 速度门**：`band = min((z−1.05)/0.20, (1.55−z)/0.10, 1.0)` × healthy_weight 0.5 × (`advance/0.015>0.5`)。站立/摔倒 0 分。旧帐篷(1.2±0.1)曾把躯干从 1.393 压到 1.21 吃 538 分/集（蹲踞引力，已除）
5. **lead_credit**：落地时 `diff_signed<0 且 body_advance>0.05 → 10×min(−diff,1.5)/1.5`。落地帧结算。基线 ~37/集（左脚微抖动底，29-33% 落地小幅负 diff），梯度向上
6. `alternate_penalty`（product×2.3）未动，连续拉力
7. `n_eval_episodes=5`（新奖励下单集 std ~8），eval_freq=2000

## 4. 固化的机制事实（勿再推导，全实测）

- **步态病理**：右脚常驻领脚的 grounded gallop。R 落地 diff +0.73~+2.60，L 落地 −0.18~+1.30（并拢）；左脚领先>10cm 落地 ~0.4%；L 落地 98.8% 不满足双支撑（R 脚 27cm 半空）→ stride 结算对左脚从不触发
- **alternate_penalty 假阳性**：alt_r 0.59 全靠贴地小抖动；真交替（R+2.0/L−1.5）值 +6.9/落地 ≈ +773/集
- **时间**：predict 0.25ms/次；串行 0.88ms/步；eval 5 集 ≈6.6s；fps ~115；prefill 500K ≈6 分钟
- **账目**：576K 在新奖励下 ≈460；站立 ≈0；零动作 −49 且 40 步摔；门控地板 ~398（曾误写 650）
- **LR 陷阱**：zip 内存着 `3e-4*prog+1e-5*(1-prog)` lambda（progress_remaining 起点 1），只在 `SAC.load` 整体加载时生效；现行 fresh 构造器+提权重模式不受影响；`reset_num_timesteps` 被注释，现行模式下无影响
- **ent_coef**：0.1 固定（`ent_coef_tensor` 路径，target_entropy −8.5 固定分支不读）

## 5. 纠错记录（别让它们复活）

1. "position_swap 指标坏了" —— 错，现行版正确；V9 宽松版（`prev!='right_ahead'`）才是 bug
2. "解注释 918 行就行" —— 错，swap 奖励只有 1.2% 梯度
3. "门关后地板 650" —— 算术错，实为 ~398（因此撤 stall 从可选变必需）
4. "lead_credit 基线为 0" —— 错，微抖动底 ~37
5. predict 6.45ms —— 错 25 倍
6. target_entropy 是追踪目标不是下限；α 不会衰减到 0
7. **"best_model.zip 未被污染"（2026-09-20 凌晨）—— 错，漏掉 EvalCallback 进程内最优从 −∞ 起步；用户因这个建议没有恢复 576K，导致整晚作废。教训：涉及 best_model.zip 内容的判断必须直接读 zip 验证，不许推理**
8. 上下文压缩曾丢"alternate_penalty 小步幅惩罚不动"结论——用户记得对，代码可证（product ∝ diff²）

## 6. 工具脚本（工作目录下）

| 脚本 | 用途 |
|---|---|
| `check_lead.py` | lead_credit 验证（wrapper vs 独立重算，已逐分吻合） |
| `check_height.py` / `check_fusion.py` | height 平顶带 / 门控 验证 |
| `cmp2models.py` | 两模型 10 集 deterministic 对比 + 落地序列（R/L 分脚 diff） |
| `smoke_train.py` | 8 路 SubprocVecEnv + warm-start 冒烟（需 `__main__` 保护） |
| `parse_fusion_run.py` | EventAccumulator 窗口统计模板 |
| `probe_*.py` | 时序/身高/站立 一次性探针 |

解释器：`/home/guo/tools_envs/mujoco_base/bin/python3`（系统 python3 无 torch/gymnasium）。脚本别放 /tmp（会被清）。

## 7. 关键代码位置（行号会漂，按函数找）

- `compute_position_swap_reward`：swap 判定 + stall 计数 + lead_credit，返回三元组
- `compute_alternate_penalty` / `_check_cycle`：符号翻转 product（prev_diff 副作用，每次落地只能调一次；在步幅结算前调用）
- 步幅结算块：真交换门控在 `if left_stride > MIN_SPACING` 分支
- 奖励合成：`lead_credit +` 在 `contact_phase_reward +` 后；`# position_swap_reward +` 保持注释
- 训练块：`__main__` 内 构造器(lr=1e-5) → old_model 提权重 → prefill 500K → `learning_starts=0` → learn(5M)
- 监控 tag：`gait/lead_credit`、`gait/stall_max`、`gait/flight_phase_penalty`、`gait/height_reward`、`gait/position_swap_rate`、`gait/speed`、`eval/mean_reward`、`eval/mean_ep_length`

## 8. 用户工作规矩（持续有效）

- 一次只改一个变量（明确豁免的批次除外）；改代码前出方案确认；用户常自己改，改前先读最新代码
- 模型/best_model.zip/git/训练进程用户自己管，不代管备份，**绝不擅自杀训练进程**
- 中文交流；结论要机制级证据（源码/实测数据）
- 惩罚针对交替问题本身；每次加压先算"摔倒 −50 vs 带伤走路"的躺平账
- **重启前必须验证 best_model.zip 内容（读 zip 的 num_timesteps）**——见 §2.1 教训
- 历史坑：站立静止局部最优；warm-start dip 是常态（56-136K 谷底后恢复，已三次复现）

## 9. 后手阶梯（当前轮失败时依序）

1. lead_credit 从 576K 起仍无法让左脚越线 → 奖励塑形边界，讨论课程/示范
2. ent_coef 0.1→0.05（仅当再现崩溃螺旋；权衡探索损失）
3. 启用 swap 奖励（`# position_swap_reward +`，+5×step_factor）与门控互补
4. 重审 `_check_cycle` 的 product 形状（∝ diff²，小步幅惩罚≈0 的老问题）
