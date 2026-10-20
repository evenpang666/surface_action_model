# 实时感知与可视化

在仓库根目录、`surface-runtime` 环境运行。先完成[环境安装](../docs/IMPLEMENTATION_GUIDE.md#环境)和[标定](../docs/CALIBRATION.md)。

## 启动

```bash
python real_scripts/live_obstacle_detection.py
```

默认双臂 front/side，自动打开 <http://127.0.0.1:8766>；只读机器人和夹爪状态。

| 需求 | 命令参数 |
| --- | --- |
| 不自动打开网页 | `--no-browser`；之后运行 `python real_scripts/live_obstacle_detection_viewer.py` |
| 只重建，不连接机器人 | `--no-robot` |
| 剔除机器人深度、提取实测表面点 | `--robot-mask`；visual-point 推理需要开启 |
| 单臂 UR7e + Pika | `--scene single-pika`，端口 8767，见[单臂说明](../docs/SINGLE_UR7E_PIKA.md) |
| 固定桌面平面 | `--table-plane NX NY NZ OFFSET`，单位 m |
| 改分辨率 | `--width 640 --height 480`；默认 640×360 @15 fps |
| 采集确认窗口后退出 | `--once --no-browser` |

默认 NumPy 体素地图；默认关闭 robot mask 和 LingBot。`--no-robot` 不能与 `--robot-mask` 同用。
使用 `--workspace-bounds XMIN XMAX YMIN YMAX ZMIN ZMAX` 设置每臂基座下的范围，
`--right-workspace-bounds` 可单独指定右臂。启动时需露出桌面，或显式给定桌面平面。

## 页面操作

| 按键 | 功能 |
| --- | --- |
| B | 体素筛选，默认关闭 |
| O / L | 障碍椭球 / 连杆椭球，默认关闭 |
| V / R | 体素框 / 机械臂点云显示 |
| 空格 | 暂停 / 继续页面更新 |

B 关闭时，障碍椭球从全部当前占据体素建立；B 开启时只用筛选体素。
双臂启动时读取标注根目录下最新 `annotations.json`，供筛选使用；
`--manual-output-dir PATH` 更换根目录，默认 `outputs/manual_obstacles/`。
标注不可用时页面显示原因。筛选及椭球显示不修改原始检测地图。
冻结预览同样默认关闭筛选，B 开启后仅使用冻结筛选体素。

### 双视角人工 OBB / 体素验证

直接使用已有 RGB 标注，或独立捕获：

```bash
python real_scripts/manual_obstacle_annotation.py
```

在 <http://127.0.0.1:8770> 捕获、画框并保存，详见[RGB 标注](../docs/BATCH_OBSTACLE_ANNOTATION.md)。
独立标注捕获时需要退出检测进程；保存后重启检测即可加载最新标注。

需要旧式 `selection.json` 时，在检测页面点击「人工 OBB / 体素验证」，
或打开 <http://127.0.0.1:8766/manual-obstacles>：

1. 捕获冻结 RGB-D 帧。
2. 在 front/side 为同一物体使用同编号框，调整旋转。
3. 点击「筛选并显示体素」，检查冻结预览；需要只看选中体素时按 B。
4. 从 `outputs/manual_obstacles/捕获时间/` 取得 `selection.json`。

单臂仅标 front，输出在 `outputs/single_pika/manual_obstacles/`；不能供双臂回放使用。

### Voxblox TSDF 后端（可选）

首次在当前 Python 环境编译：

```bash
sudo apt-get install build-essential cmake git libeigen3-dev libgoogle-glog-dev libprotobuf-dev protobuf-compiler
python scripts/build_voxblox.py
python real_scripts/live_obstacle_detection.py --occupancy-backend voxblox
```

默认生成 ESDF；网页可切换 TSDF/ESDF 和表面/距离场。
保存地图加 `--voxblox-output outputs/voxblox/map.npz`；换 Python ABI 后重新构建。

### OctoMap 体素后端（可选）

```bash
python -m pip install 'octomap-python==1.10.0.0'
python real_scripts/live_obstacle_detection.py --occupancy-backend octomap \
  --octomap-output outputs/octomap/map.ot
```

支持 `.ot` / `.bt`；退出时导出，并保存坐标放置元数据。
关闭体素地图改用聚类 OBB：`--no-occupancy-map`；该模式不能进行体素捕获/筛选。

## 推理与回放

- [模型推理](../docs/IMPLEMENTATION_GUIDE.md#在线运行)：检测端需 `--robot-mask`；执行器默认 dry-run。
- [LeRobot 回放](../docs/MODEL_FREE_OBSTACLE_VALIDATION.md)：单进程 `lerobot_replay replay` 默认实机执行。

## 排错

快照：<http://127.0.0.1:8766/snapshot.json?compact=1>。先看 `status` 和 `diagnostics`。

| 问题 | 检查 / 处理 |
| --- | --- |
| 相机启动失败 | 退出其他相机进程，检查 USB/序列号和日志中的实际流配置 |
| `Active TCP differs` | 核对活动 TCP 与配置；修正后重启检测 |
| 修改 TCP 后仍显示同一错误 | `acquisition_worker_state=failed` 表示采集已退出，需要重启 |
| 处理慢、快照过期 | 查看 `capture_to_publish_seconds`、`frame_interval_seconds` 及各阶段耗时 |
| 机器人被识别成障碍 | 开启 `--robot-mask`；默认关闭 |
| visual-point 推理没有表面点 | 开启 `--robot-mask`，检查 `current_surface` |
| 关闭 B 后仍看不到全部椭球 | 重启检测并刷新页面，确认请求不带 `manual_filter=1` |
| Voxblox 导入失败 | 在正在使用的 Python 环境重新运行构建脚本 |

噪声排查看 `depth_evidence`；地图只把当前 occupied 体素作为障碍，快照时效检查仍保留。
