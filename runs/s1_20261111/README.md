# S1准备：接口、独立环境与静态预检

日期：2026-11-11。S1未完成；未导入Newton/Warp、编译内核或执行物理step，未采训练数据、未训练。因资源准入与初始IK候选失败，停止于静态预检。未关闭应用或修改相邻工程。

## 已完成与证据

- 独立`.venv`不继承system-site-packages；Python 3.13.0，Newton 1.6.0、Warp 1.17.0、numpy 2.2.4、scipy 1.15.2、trimesh 5.1.0。requirements.lock记录全部5个包及SHA256。安装约106秒，低于15分钟；安装日志（本地归档）、依赖校验（本地归档）均通过，不等于CUDA运行通过。
- 按S0哈希引入9个资产约7.01MiB，assets/manifest.json保留来源、commit、哈希、未知许可证、单位及装配；未引入旧装料/锅/数据/权重。
- [接口契约](../../INTERFACES.md)与contracts/task/geometry/robot/resources静态模块已实现，没有导入相邻代码。Newton环境step/reset/snapshot及在线动作执行器尚未实现；模型输入维度未冻结。
- tests.log（本地归档）：12项CPU测试通过，涵盖参考轨迹导数、旋转坐标系速度、COM换算、承载点保持、FK雅可比差分、已知FK位姿的IK回归、实际网格间隙、歧义承载面拒绝、转移/窗口边界、低内存拒绝。

## 静态结果与限制

[preflight.json](preflight.json)保存配置、资源及全部8个IK候选。seed=20261111，最多8个种子、每个max_nfev=180，耗时约1.02秒。

工具承载点`[0.32,0,0.02850011]m`，法向约`[-0.443421,0.086235,0.892155]`。该点只有一个向上表面，下方另有向下表面。球心沿法向偏移20.2mm，距全网格最小间隙+0.2mm；网格水密且绕序一致。这支持候选初态无穿透，不代表静置稳定、整片承载区已识别；单三角面法向调平是否适合曲面仍待核查。

原候选基座原点、承载点世界`[0.45,0,0.35]m`与调平姿态的8个IK候选均失败。最好位置误差约48.642mm、旋转误差0.05381rad，要求≤0.02mm和0.0002rad。已知FK位姿的IK回归通过，但尚未与Newton FK交叉验证。固定搜索失败不证明数学不可达。保留原配置，不移动基座、改目标或放宽容限来隐藏失败。整段路线、自碰撞与地面间隙尚未验证，admitted_for_physics=false。

预检可用RAM约0.77GiB，低于4GiB启动门槛；GPU/磁盘准入通过。环境逻辑大小约570MiB、缓存571MiB，均在预算内；uv可能使用硬链接，逻辑大小不等于新增物理占用。没有开始30分钟/23,040子步物理预算。

## 下一项与复核

先限定CPU静态搜索预算，解释初始位姿失败并版本化可达候选，再核查8cm路线、倾角范围、关节连续性/速度/加速度、自碰撞与地面间隙。资源达标后才实现并验证物理环境。S2正式训练采集未开始。

```powershell
.venv/Scripts/python.exe -B scripts/static_preflight.py --output runs/s1_20261111/preflight_recheck.json
.venv/Scripts/python.exe -B -m unittest discover -s tests -v
uv pip check --python .venv/Scripts/python.exe
```

本轮不新增核心主课。建议15分钟阅读[INTERFACES](../../INTERFACES.md)，再看task.py及旋转坐标系测试。自测：为何State不能直接充当快照？为何记录全部子步命令？为何IK搜索失败不能直接宣布不可达？
