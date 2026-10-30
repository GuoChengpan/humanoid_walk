# history — 历史源码与早期日志

本目录是各阶段的原样快照，**不保证能在当前代码上直接运行**；当前实现见仓库根目录的
`train_humanoid_sac.py`。加入 git 是为了保留改奖励/改接口时的原始版本，便于事后归因。

| 文件 / 目录 | 说明 |
|---|---|
| `train_humanoid_ppo.py` | 早期 PPO 训练脚本（含最早的 `WalkRewardWrapper`） |
| `train_humanoid_sac_fromscratch.py` | 从头训练版本（不加载旧权重） |
| `train_humanoid_sac_1028.py` | 10 月的一版主脚本快照 |
| `train_humanoid_sac_backup_2M_slidewalk.py` | 2M 步 slide-walk 奖励版的备份 |
| `train_humanoid_sac_batch11_osc.py` | 引入摆腿时钟模板时的脚本 |
| `train_humanoid_sac_batch12_4M.py` | 4M 续训的一版原样备份 |
| `train_humanoid_sac_batch13_14_1_86M.py` | 1.86M 续训的一版原样备份 |
| `train_humanoid_sac_batch16_1_4M_T2.py` | 摆腿模板 T2 驱动时的原样备份 |
| `model_重构前9.5.py`、`model_重构后9.8.py`、`model_重构后9.8_副本.py` | 早期 wrapper 的两个版本；原来后缀误写成 `.txt`，内容一直是 Python |
| `test.py` | 旧版 wrapper 副本，没有 `main`，不是渲染脚本 |
| `README_backup_2026-10-20.md` | README 的历史版本 |
| `README_detailed_2026-10-20.md` | 长版 README（完整数据表与方法论） |
| `ppo_humanoid_walk/` | 2026-08-01 的 PPO 日志（`PPO_1`/`PPO_2`，另含转 SAC 的 `SAC_3`） |
| `pop_humanoid_walk/` | 同日的 PPO 批量尝试日志（`PPO_1…PPO_7`） |
| `humanoid_logs/` | 更早的 PPO 日志（`PPO_1…PPO_4`） |
| `sac_humanoid_logs/` | 第一次 SAC 尝试的日志（`SAC_1`） |
