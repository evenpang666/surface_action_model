# 单臂 UR7e + Pika

在 `surface-runtime` 环境、仓库根目录执行。配置：`configs/ur7e_single_pika.yaml`。
此入口提供单臂感知、标定与查看，不使用双臂训练/执行器。

| 设备 / 路径 | 默认 |
| --- | --- |
| UR7e | `192.168.124.73`，只读 RTDE |
| frontcam | `405622074939`，唯一重建相机 |
| wristcam | `260522276066`，不参与重建 |
| 页面 | <http://127.0.0.1:8767> |
| 标定 | `configs/hardware/single_pika/` |
| 标定过程 | `outputs/single_pika/calibration/` |

## 环境

安装[运行依赖与 OpenCV contrib](IMPLEMENTATION_GUIDE.md#环境)，另装 `python -m pip install pyserial`。
Pika 默认使用 YAML 中的稳定 by-id 串口；更换设备后先检查：

```bash
python -m real_scripts.pika_serial_discovery
```

可在 YAML 设置 `gripper_port: auto`，或启动时传 `--pika-gripper-port /dev/serial/by-id/...`。
设备识别失败时核对 USB、串口权限和占用进程。

## 标定

先在 PolyScope 校准活动探针/工具 TCP；下面的 tcp 命令只读取当前配置。

```bash
python real_scripts/calibrate_single_ur7e_pika.py tcp
python real_scripts/calibrate_single_ur7e_pika.py camera
```

1. 固定 frontcam 与板，移开机械臂。板为 **11×8、70 mm 格、52 mm marker、DICT_5X5_100、legacy pattern**。
2. RGB 检测稳定后按空格锁板，等待自动采样；`q` 取消。
3. 按 `selected_points.png` 与终端提示触碰指定内角，回车记录；不可达用 `id=N` 换点。
4. 移开机械臂，保持板不动，回车复核位姿；质量通过后安装相机标定。

默认颜色流 1280×720 @15 fps；不支持时自动尝试实际档位。无图形界面加 `--headless`。

板仍固定时续做或补测：

```bash
python real_scripts/calibrate_single_ur7e_pika.py camera --resume
python real_scripts/calibrate_single_ur7e_pika.py camera --resume --retouch-corner-ids 19
```

板已移动则开始新会话。`--output-dir PATH` 更换过程目录。
探点 RMS 需 ≤5 mm、重投影 RMS ≤0.8 px；失败时检查实际板格距、角点与 TCP，勿手改通过标志。
重新生成 `active_tcp.json`、移动相机或更换工具后，重新执行 camera。

## 启动和查看

```bash
python real_scripts/live_obstacle_detection.py --scene single-pika
```

默认打开 <http://127.0.0.1:8767>。无机器人重建加 `--no-robot`；
Voxblox 加 `--occupancy-backend voxblox`；不开网页加 `--no-browser`。

| 需求 | 参数 |
| --- | --- |
| 剔除机器人深度 | `--robot-mask`，默认关闭 |
| 修改工作区 | `--workspace-bounds XMIN XMAX YMIN YMAX ZMIN ZMAX` |
| 给定桌面 | `--table-plane NX NY NZ OFFSET`，在 ur_base 下 |
| 采集确认窗口后退出 | `--once --no-browser` |

单视角不能启用 LingBot。Pika 开度为 0=闭合、1=全开（95 mm），不同于 Robotiq 闭合比例。

## 人工筛选

打开 <http://127.0.0.1:8767/manual-obstacles>：捕获 front → 画框/旋转 → 点击筛选。
冻结预览按 B 查看选中体素；主页面 B 默认关闭，O 显示当前范围的障碍椭球。
输出为 `outputs/single_pika/manual_obstacles/捕获时间/`，可用 `--manual-output-dir` 更换。
单臂标注和快照不能交给双臂回放入口。

## 工具几何与排错

当前 `pika_geometry.json` 使用 `legacy_inherited` 安装几何。安装方式改变时，
从 `pika_geometry.example.json` 另存实测文件，填写 `flange_T_pika_step`、网格参考开度，
核验后更新 YAML 的 `robot.pika_geometry`。

| 问题 | 处理 |
| --- | --- |
| 没有串口 / 多个夹爪 | 运行发现命令，指定稳定 by-id 路径 |
| 开度过期 | 检查串口遥测；默认新鲜度 0.25 s |
| TCP 不匹配 | 核对活动 TCP，重新执行 tcp / camera |
| 标定恢复被拒绝 | 板已移动或文件摘要变化，重新开始 camera |
| 网格不贴合夹爪 | 核对实测安装矩阵、手指轴向与参考开度 |
