# 运行证据索引

当前结论、准入差距及下一项见 [STATUS](../STATUS.md)。运行目录按问题和日期命名。进入目录先读 `README.md` 或 `RESULT.md`，核对机器可读证据时看 `validation.json`、`comparison.json`、`summary.json` 或 `failure_analysis.json`。原始轨迹、检查点、冻结清单和停止记录保留在各自目录。

## 当前工作

| 问题 | 证据 |
| --- | --- |
| 模型建议的小动作在 Newton 中有效吗？ | [两案实测](s3_gradient_action_second_case_20261118_verified/README.md) · [逐步误差分解](s3_gradient_action_feedback_20261118/README.md) |
| 普通轨迹增加后为何退步？ | [三个固定终点的误差及梯度](s3_continuous_route_fit_audit_20261119/README.md) · [6/12 回合拟合](s3_continuous_route_fit_comparison_20261119/README.md) · [训练数据归档](s3_continuous_routes_20261118/RESULT.md) |
| 完整路线的简单控制能保住球吗？ | [原计划与固定反馈](s3_closed_loop_baseline_20261118/README.md) |
| 当前开发评分用什么参照？ | [原 61 对固定终点](s3_world_increment_data61_fit_20261117/README.md) · [小幅 Newton 实测及有限扰动、局部导数](s3_small_action_response_capture_20261117/README.md) · [49/61 对响应分解](s3_world_increment_data61_response_verified_20261117/README.md) |

## 历史证据入口

| 阶段或专题 | 证据 |
| --- | --- |
| S0 资产与环境 | [资产核查](s0_20261111/README.md) · [依赖安装](s1_20261111/README.md) |
| S1 规定路线、离铲与在线控制 | [可达性](s1_reachability_20261111/README.md) · [物理汇总](s1_arm_20261112/physical_summary.json) · [在线回合](s1_episode_20261112/summary.json) |
| S2 数据、划分和模型 | [数据清单](../data/s2_pilot_20261112/manifest.json) · [数据验收](s2_pilot_20261112/validation.json) · [模型汇总](s2_world_model_20261112/summary.json) |
| S3 确定性快照与动作响应 | [原生重放](s3_native_repeatability_20261114_retry/validation.json) · [公开状态恢复](s3_deterministic_archive_import_20261115_resume/validation.json) · [原幅动作回放](s3_deterministic_action_response_20261115/validation.json) |
| S3 配对、分组与历史动作 | [配对扩充](s3_coverage_20261113/validation.json) · [归一化审计](s3_group_normalization_20261113/audit.json) · [历史采集](s3_history_coverage_20261114/capture_validation.json) · [时变采集](s3_training_timing_capture_20261115/validation.json) |
| S3 接触与速度残差 | [接触冲量](../diagnostics/resource_probe/runs/s3_xpbd_contact_probe_20261117_rebind/README.md) · [世界系速度残差](s3_world_increment_fit_20261117/README.md) · [全时段监督](s3_world_increment_extended_window_fit_20261117/README.md) · [俯仰逐步归因](s3_world_increment_newton_reference_20261117/README.md) |
| S3 数据扩充及损失对照 | [46/49 对](s3_fitability_p3_20261116/README.md) · [49/61 对](s3_world_increment_data61_fit_20261117/README.md) · [61/97 对](s3_broad_training_20261118/README.md) · [自转监督时段](s3_spin_supervision_window_fit_20261118_verified/README.md) |
| 方法复盘 | [动作响应卡点及路线比较](s3_retrospective_20261115/README.md) |
