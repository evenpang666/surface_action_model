# UR7e + Pika + D435i 远程 pi0.5 推理

> 历史独立采集/推理工具说明，不是当前双臂 Surface Action Model 部署主线。
> 当前环境、地址及安全前置条件以[根 README](../README.md)为准。

文档：

- [环境安装](docs/environment_setup.md)
- [Pika Sensor 数据采集](docs/demo_collection.md)
- [VLA 推理](docs/vla_inference.md)

## 快捷自查

```bash
# 检查命令行入口和本地相机编号
ur7e-vla --help
ur7e-vla list-cameras

# 检查策略服务连通性和 action 形状；不会连接机械臂
python scripts/probe_policy.py --host 192.168.125.59 --port 8000
```

```bash
# 推理 dry-run；不会向 UR7e 或夹爪下发命令
ur7e-vla run --config config.yaml --task "pick cube" --duration 30

# 实时 Sensor 遥控采集；默认仅本机 Web UI，访问 http://127.0.0.1:7860
ur7e-vla collect-demo --config config.yaml --task "pick cube" --execute

# 局域网访问：另一台主机打开 http://<采集主机局域网-IP>:7860
ur7e-vla collect-demo --config config.yaml --task "pick cube" --execute \
  --server-name 0.0.0.0 --server-port 7860
```

采集页面会常驻：先点击 **Start Teleoperation** 连接设备并恢复状态，再点击 **Start Recording** 才开始写入原有 PNG/NPY episode，并额外保存对齐深度和夹爪开口供 safety 点云预处理，最后点击 **Stop Teleoperation & Save** 正常完成保存。采集进程异常退出后，修复硬件问题即可在同一页面再次点击 Start。详见 [Pika Sensor 数据采集](docs/demo_collection.md)。

只有传入 `--execute` 才会向 UR7e 和夹爪下发命令。真机前确认急停可达、工作区无障碍物，并先完成 dry-run。
