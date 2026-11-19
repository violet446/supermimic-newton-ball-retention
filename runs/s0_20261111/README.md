# S0 静态核查：Newton / EC66 / 单刚体球

日期：2026-11-11。性质：本机元数据、源码和二进制网格静态核查；没有导入 Newton/Warp、编译内核、安装、下载、仿真或训练。预算：60分钟、增量≤200MB；本轮在规格与下一轮检查方案收束后停止。未读旧数据、权重、历史报告，未启动 WSL，未关闭应用。未复制资产或旧算法代码。

## 1. 证据与边界

- [machine.json](machine.json)：机器即时快照（不是可用资源保证）。Windows 11，i9-13980HX，24核32线程，约15.6GiB RAM，采集时可用0.90GiB；RTX 4060 Laptop 8188MiB，驱动581.80，占用516MiB；C盘约28GiB空闲。WSL Ubuntu-24.04 已安装但停止，内部环境未知。
- [static_inventory.json](static_inventory.json)：9个必要资产的绝对路径、大小、SHA256、关节定义、STL包围盒；7个参考源码/依赖文件及2个本机Newton API文件的哈希。选定文件相对HEAD无差异。源仓库另有无关未跟踪文件，未读取其内容。
- static_audit.py（本地归档）：可重跑的标准库核查脚本，仅写本目录JSON，不导入物理库；执行命令 `python -B runs/s0_20261111/static_audit.py` 已通过。网格解析不等于接触或水密性验证。

来源仓库：`C:/newton/New/Particle_Spreading`，HEAD `866019f3d00a3400e76bca9162f0c084483e0786`；资产最近提交 `484e981a76f68b4bb5ce7b334bfa92c581f4139f`。具体文件以SHA256为准。旧项目AGENTS中的旧任务运行/推送约定不迁入本任务；遵循用户限定的只读资产与控制核查范围。

## 2. 版本与运行位置

实际可访问解释器：`C:/python/python.exe`，Python 3.13.0；已安装 Newton 1.6.0、warp-lang 1.17.0、numpy 2.2.4、scipy 1.15.2、trimesh 5.1.0，python-fcl未安装。来自dist-info元数据，没有执行import或CUDA测试。Newton无direct_url.json，无法据此给出上游commit；保存了关键API源码哈希，不能把PyPI版本冒充源码commit。

旧模块requirements声明 Newton 1.6.1 / Warp 1.18.0 / numpy 2.3.1 / scipy 1.18.1 / trimesh 5.1.1 / python-fcl 0.7.0.11。这是声明，不是本机安装事实。源项目 `.venv/Scripts/python.exe` 与 `.venv/bin/python` 均不存在；其setup.sh要求Python 3.12，但本轮没有执行。根目录旧启动脚本也是资料，不是新任务入口。

下一轮优先评估Windows独立 `.venv`，Python 3.13.0 / Newton 1.6.0 / Warp 1.17.0作为基于本机证据的候选固定组合；不使用system-site-packages，不升级全局或相邻环境。安装前解析全部传递依赖并保存锁文件/包哈希；目前没有完整锁文件，也没有承诺这组版本已可运行。S1无需torch、PPO、G1或python-fcl；用本地网格几何检查替代对旧geometry模块的导入。若Windows具体阻塞，再单列WSL方案，不能凭其存在就宣称可用。

## 3. 最小资产及装配

资产根：`C:/newton/New/Particle_Spreading/experiments/release_distribution_ec66_v2/assets`。

| 必需文件 | 用途 | 解释/处理 |
|---|---|---|
| ec66_description.urdf | 固定基座、6个revolute关节及flan固定关节 | 保留关节原点/轴/限位；首版不认证惯量或力矩能力 |
| ec66/base_link.STL、link1.STL至link6.STL | 7个机械臂link网格 | URDF visual/collision使用相同完整三角网格，无scale，加载器按米解释；CAD原始单位未独立认证 |
| shovel.STL | 铲子visual与接触几何 | 原始坐标经旧加载器减偏置、毫米→米、轴交换，见下 |

共9文件、7,351,978字节（约7.01MiB）。当前只列清单，S1构建时才有必要复制这9个文件。排除未被URDF引用的80字节flan.STL、source_layout.json（旧装料数据）、锅、旧配置与整个工程。资产授权/许可证未知：所查根目录及资产未发现许可证文件；不能套用Newton的Apache-2.0到EC66/STL。仅本地研究参考，若以后发布资产须补证。

工具原始点 `(x,y,z)` 到工具坐标：`(z, x-138, y-30)*0.001`。网格范围约 x=[0,0.543879]、y=[-0.136085,0.134970]、z=[-0.029675,0.201064]米，21,376三角面。中心区域向上三角面高度跨度较大，不能用整个AABB当承载平面或掉落边界。

`robot.py:17`定义法兰到工具旋转 `Rx(90°) @ Rz(90°)`，平移为零；工具原点是flan原点，**不是球的承载中心**。URDF中link6到flan平移 `[0,-0.089,-0.00047661]`米，rpy=`[-1.5708,0,3.1416]`。旧世界基座来自protocol.py，位置`[-1.2069976,-0.70478,0.1441063]`、yaw=-151.25°；记录来源但新场景不继承，拟固定新基座在世界原点，重新做可达性及地面间隙核查。

关节1/2/4/5/6限位±6.2832rad，关节3±2.7925rad；URDF速度上限分别4.6199、4.6199、6.3161、8.5379、8.5379、8.5379rad/s。effort前三150、后三28（URDF字段，未认证）。旧代码另有项目自定速度/加速度上限，不能当厂家规格继承。flan有零尺寸box占位，旧runtime加载时删除；flan惯量含约1量级主惯量且质量仅0.10466kg，动态机械臂建模前须单独审核。

## 4. 控制与接触的关键结论

源码阅读顺序：`geometry.py:7`（单位变换）→`robot.py:15`（FK/IK及关节约束）→`runtime.py:8`（URDF、flags及装配）→`collect.py:45–62`（命令和step）。只引用，不复制这些模块，因为geometry还隐式依赖fcl和旧装料数据。

旧路径是规定关节运动：曲线采样q/qd → 写state.joint_q/joint_qd → eval_fk → collide → SolverXPBD.step → 复制关节命令到下一状态。机器人全部设置KINEMATIC，非动力学伺服控制。末端IK采用SciPy least_squares、位置与旋转残差，不能直接放进PyTorch想象链路并假设可导。路径连续性、奇异点与限位仍需本任务验证。

旧球状物是 `add_particle`，没有刚体姿态/自转；旧runtime清除COLLIDE_SHAPES、工具has_shape_collision=False，但保留particle接触。因此本任务必须新增动态刚体sphere（含质量和惯量）、打开球—工具/臂/地面形状接触，并明确排除相邻连杆及装配重叠对。不能把500改为1就称单刚体球复用成功。完整凹工具网格先保留，不能用全局凸包填平铲腔。网格窄相碰撞与运动学表面速度是否正确是S1待测项。

本机Newton `builder.py:4958,7464,7820` 有body、sphere、triangle mesh入口；`solver_xpbd.py:49`文档声明刚体支持、速度布局为 `(v_com_world, omega_world)`，且不支持自动执行joint effort/velocity limits。这是安装源码说明，不是本机物理验证。执行器约束需任务层显式检查；工具原点速度需从COM速度加 `omega × (origin-COM)` 换算，不能直接把body_qd前三维当工具原点速度。

| 候选 | 判断 |
|---|---|
| 六维关节位置/速度修正 | 与现有FK入口相容，易写可微FK；会扰动规定平移，须另约束轨迹，不作为默认学习接口 |
| 两维工具倾角＋规定平移 | 暂定S1控制候选。绕承载中心倾斜，工具原点随之补偿；受限微分IK/连续关节插值实现，必须先验证可达性与运动一致性 |
| 力矩/动态伺服 | 目前没有已核实的旧接口与可靠惯量，首轮不选，不声称真机可执行 |

拟采用运动学机械臂＋动态球作为首个物理系统，保留真实臂几何、有限速连续关节运动与碰撞检查；它不包含机械臂受球反作用的动态响应。两维接口通过S1后才冻结；若失败先报告原因，不自动扩大自由度。想象中的控制器需用可微阻尼最小二乘/FK或经验证的控制残差模型；不能将未来实际工具姿态喂回预测。

## 5. 下一轮入口与学习安排

具体参数、命令草案、判据及预算见 [S1_CHECK_PLAN.md](S1_CHECK_PLAN.md)。本轮止于S0；首次物理运行未执行，接触稳定性、动作可控性、可达性、性能均未知。当前低内存不满足拟定运行准入条件。

本轮是环境/资产核查，不新增核心主课。建议15–20分钟读本报告第3–4节，再读 [任务设计](../../DESIGN.md) 第2–4节；自测：为何particle不能直接代表有自转的刚体球？为何规定关节运动不等于力矩控制？为何未来实际工具姿态不能进入想象滚动？原理学习继续使用 既有核心学习入口（本地归档），无需先读旧工程排错附录。
