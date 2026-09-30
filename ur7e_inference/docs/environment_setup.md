# 环境安装

> 本文属于历史 Pika/Sci-VLA 工具。当前双臂 Surface Action Model 请使用
> [仓库根环境说明](../../README.md#环境)，不要混用下面的版本与安装路径。

服务端与机器人端使用独立 Python 环境：服务端运行 Sci-VLA/OpenPI 和 GPU 推理；机器人端只运行本项目、相机、Pika 与 `openpi-client`。

## pi0.5 服务端

服务端需要 Linux、NVIDIA GPU、Python 3.11 和 `uv`。在 Sci-VLA 的 OpenPI 目录执行：

```bash
cd /path/to/Sci-VLA/third_party/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

若仓库不是递归克隆，先执行 `git submodule update --init --recursive`。确认 checkpoint 路径可读，并用 `nvidia-smi` 检查 GPU、显存和占用进程。

启动 `mani_real_pi05`：

```bash
cd /path/to/Sci-VLA/third_party/openpi
POLICY_DIR=/path/to/mani_real_pi05/checkpoint \
  /path/to/ur7e_inference/scripts/start_pi05_server.sh
```

服务默认监听 `0.0.0.0:8000`。确认防火墙允许机器人侧访问该端口。

## 机器人端

项目要求 Python 3.9 或更高。包含推理、录像和示教采集的安装：

```powershell
conda create -n ur7e_collector python=3.10 -y
conda activate ur7e_collector
python -m pip install --upgrade pip

cd C:\Users\15261\Documents\projects\Sci-VLA\third_party\openpi
python -m pip install -e packages\openpi-client

cd C:\Users\15261\Documents\projects\ur7e_inference
python -m pip install -e ".[web-demo]"
Copy-Item config.example.yaml config.yaml
```

只运行实时推理时可使用较小安装：

```bash
pip install -e .
pip install agx-pypika
cp config.example.yaml config.yaml
```

当前示教采集直接保存点云预处理所需的 raw RGB-D NPY/JSON 文件，不依赖 LeRobot。`.[web-demo]` 安装 Gradio Web 页面；Pika Sensor 的 Lighthouse 追踪还需要项目内 Pika SDK 所用的 `libsurvive/pysurvive` 后端。RealSense 驱动、D435i/D405 固件及 USB 权限需按设备厂商要求配置。

## 硬件与网络检查

```bash
ur7e-vla --help
ur7e-vla list-cameras
python scripts/probe_policy.py --host 192.168.125.59 --port 8000
```

## 启动交互式 VLA 推理与示教采集

策略服务通过探测后，可在机器人端启动图形界面：

```powershell
ur7e-vla vla-gui --config config.yaml
```

此命令是 dry-run。只有在确认急停可达、工作空间清空后，才添加 `--execute`：

```powershell
ur7e-vla vla-gui --config config.yaml --execute
```

`vla-gui` 使用 Python 自带的 Tk 图形组件；若启动时报 `No module named tkinter`，请为当前
Python/conda 环境安装或启用 Tk。MP4 录制使用项目的 OpenCV 依赖，并写入
`runtime.recording_dir`（默认 `recordings`）。

示教采集使用 Gradio Web 页面，不需要 `$DISPLAY` 或 Tk：

```bash
ur7e-vla collect-demo --config config.yaml --task "pick cube" --execute
```

默认访问 `http://127.0.0.1:7860`。远程访问时显式监听局域网：

```bash
ur7e-vla collect-demo --config config.yaml --task "pick cube" --execute \
  --server-name 0.0.0.0 --server-port 7860
```

页面会一直运行；**Start Teleoperation** 启动独立遥操进程但尚不录制，状态 READY 后点击 **Start Recording** 才开始写入数据，最后通过 **Stop Teleoperation & Save** 正常结束并把 episode 保存到 `demo.output_dir`。采集进程异常退出后可直接再次点击 Start。

在 `config.yaml` 配置 Pika 串口、两路相机和 UR 地址。默认的 `auto` 会被动识别 PiKA Sense、PiKA Gripper 和相应腕部相机；Linux 用户还必须有串口设备访问权限，通常执行 `sudo usermod -aG dialout "$USER"` 后重新登录。机器人主机必须能同时访问 UR7e 的 `169.254.175.10` 与推理主机的 `192.168.125.59`；通常需要两张网卡或正确的静态路由。
