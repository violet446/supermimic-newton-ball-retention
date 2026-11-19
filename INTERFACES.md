# 单球任务接口

物理状态使用 SI 单位及 xyzw 四元数。球与工具的位姿、线速度以各自原点为准，角速度以世界坐标表示。当前任务使用 60 Hz 控制和 960 Hz 物理积分，一个完整控制周期包含 16 个物理子步。[任务配置](configs/s1_online_episode.json) · [设计](DESIGN.md)

## 环境与转移

| 入口 | 输入与输出 |
| --- | --- |
| `EpisodeEnv(root, cfg, device)` | 构建规定路线、机械臂、工具和单刚体球。 |
| `reset(episode_id=None)` | 恢复配置初态、倾角滤波与离铲监测，返回 `State`。 |
| `step(action)` | 输入两轴归一化目标 `[roll, pitch]`，返回 `Transition`。 |
| `snapshot()` / `restore(saved)` | 保存及恢复物理缓冲、控制器、时钟、离铲监测与回合计数。 |

动作裁剪到 ±1 后乘以配置倾角上限。当前上限每轴 5°，滤波时间常数约 0.63246 秒。先绕世界 X 轴横滚，再绕世界 Y 轴俯仰。控制器先生成并检查整个周期的子步命令，再推进物理。归档的请求动作保留原值，下一状态的 `previous_action` 保留裁剪值。

`State` 记录球、工具、关节 q/qd、滤波倾角和速率、上次裁剪动作、参考相位、时钟及离铲持续时间。`Transition` 记录回合 ID、步索引、`state_t`、请求动作、实际执行的子步 q/qd、`next_state_before_reset`、参考、上下文和结束标签。工具系相对线速度包含旋转坐标系的角速度项，见 `task.observation_t`。

离铲触发后的最后一条转移仍保存重置前状态及 1–16 个实际子步。`terminated` 表示物理终止，优先于时限 `truncated`。完成规定路线、全程无离铲及末态承载与近接触均满足才标记 `task.success`。当前奖励为 null。`validate_window` 检查时序、状态衔接和版本，拒绝跨回合或越过结束的窗口。短末步可用于终止标签，常规动力学窗口只用完整周期。[环境入口](src/ball_retention/episode.py) · [记录契约](src/ball_retention/contracts.py)

## 数据与划分

| 数据 | 入口 |
| --- | --- |
| 普通回合 | `dataset.read_manifest(dataset_root, verify_splits=...)` 校验划分与哈希，`load_episode(path, dt, substeps, expected_id=...)` 加载回合。S2 模型入口为 `model_data.prepare_dataset`，离线拟合显式使用 `verify_test=False`。 |
| 普通轨迹的嵌套档位 | 6 与 12 回合共享相同六组初态，按完整回合选取。当前对照按 32 步完整窗口读取，仅使用训练档。[归档](runs/s3_continuous_routes_20261118/RESULT.md) · [训练入口](scripts/train_s3_continuous_routes.py) |
| 同快照动作配对 | `pair_data.prepare_pairs` 读取多幅度视图。时变配对用 `timed_pair_data.prepare_timed_pairs` 读取 `s3-timed-action-pairs-v1` 清单，校验起点、命令、时钟、标签和分支哈希。 |
| 训练、开发与组权重 | `pair_collection.prepare_collection` 合并清单。组、组内快照、快照内动作轴和幅度逐层分配权重。组、回合及同快照的不同动作幅度不能跨划分。 |

清单记录相对路径、哈希和来源身份。开发用于评分，测试轨迹封存。相同物理初态在不同历史动作下的快照仍属于同一组。[S2 清单](data/s2_pilot_20261112/manifest.json) · [配对集合](configs/s3_pair_collection.json) · [采集入口](scripts/collect_s2.py) · [时变采集](scripts/collect_s3_training_timing.py)

## 世界模型

统一状态为工具系位置、相对线速度和球自转，共 9 维。`model_data.known_rollout` 从起始时钟、滤波状态、动作序列和已知参考得到 20 维工具特征。使用检查点对应的 `known_names`、`known_mean`、`known_scale`。预测时起点取当前观测，以后的球状态逐步使用模型预测。未来实测球状态用于监督或离线评分，实测工具未来状态也不回填预测。

| 模型 | 加载与动作导数 |
| --- | --- |
| S2 工具系残差 | `ResidualMLP.from_state_dict`，输入 20 维已知特征，动作导数见 `action_grad`。 |
| 解析坐标残差 | `FrameResidualMLP.from_state_dict`，`frame_model.pack_features` 给 20 维已知特征附加 90 项解析变换，动作导数见 `frame_grad.frame_action_sensitivity`。 |
| 位置梯形积分 | `position_integral_model.load_position_model` 兼容原残差与 `trapezoid_position_residual_v1`。位置由步前、步后速度的均值推进。 |
| 世界系速度残差 | `WorldIncrementMLP.from_state_dict` 加载 `world_increment_trapezoid_v1`，用 `model.features` 附加确定性变换。动作导数用 `world_action_jvp`，对完整请求序列的给定方向求导。 |
| 局部面残差实验 | `LocalSurfaceResidualMLP.from_checkpoint` 使用同次运行的 `field.npz` 及其校验哈希，保存平滑面场的边界和系数。 |

NPZ 用 `np.load(..., allow_pickle=False)`。在模型自己的归一化坐标中训练和滚动，再还原到 SI 评分。`state_error_scale_si` 固定状态损失尺度，配对损失的 `absolute_scale_si` 只作用于绝对状态项。原历史归一化可用 `group_fold` 的 `s2_only_v1`，留组重算用 `retained_training_v2`。[模型入口](src/ball_retention/world_increment_model.py) · [S2 拟合](scripts/train_s2.py) · [历史实验索引](runs/README.md)

固定终点、训练侧结果、冻结开发响应和完整 Newton 控制应分开报告。有限扰动用实际正负动作幅度，局部导数用名义动作附近的变化率，旧指标保护由相应冻结配方定义。[原 61 对评分](runs/s3_world_increment_data61_fit_20261117/README.md)
