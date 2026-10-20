# RGB 障碍标注

在仓库根目录、`surface-runtime` 环境运行。页面默认 <http://127.0.0.1:8770>。

## 捕获当前相机

先退出占用相机的检测进程：

```bash
python real_scripts/manual_obstacle_annotation.py
```

1. 点击「捕获当前相机帧」。采集完成后相机自动释放。
2. 在 front/side RGB 图中画框、旋转，同一物体使用相同 `object_id`。
3. 点击保存；新增捕获用再次捕获操作，不复用移动前的框。

输出：`outputs/manual_obstacles/YYYYMMDD_HHMMSS_微秒±时区/`，
包含 RGB、`capture.json`，保存后生成 `annotations.json`。
默认相机序列号来自项目硬件配置；覆盖用 `--front-serial SERIAL --side-serial SERIAL`。
`--output-dir PATH` 更换输出根目录，`--no-browser` 关闭自动打开页面。

保存后可直接[回放避障](MODEL_FREE_OBSTACLE_VALIDATION.md)；检测可视化需重启后加载最新文件，按 B 开启筛选。

## 标注 HDF5

单个文件或目录均可：

```bash
python real_scripts/manual_obstacle_annotation.py --hdf5 /path/to/episode.hdf5
```

页面内选择文件和帧，画框并保存；无障碍帧点击「标记无障碍」。
支持 front-only 或 front/side；HDF5 模式不打开相机。
保存写回原 HDF5 的 `annotations/obstacle_obbs`，按原帧索引与时间戳对齐，可续标。

任务目录的双视角批量入口：

```bash
python real_scripts/batch_obstacle_annotation.py --task-dir /path/to/task --inspect
python real_scripts/batch_obstacle_annotation.py --task-dir /path/to/task
```

## 数据检查

- 双视角输入需 `timestamps_ns` 和 `observations/images/{front,side}/rgb`。
- 在录制 RGB 上画框；仅做二维标注时不要求深度。
- 双视角格式为 `dual_rgb_obb_annotation_v2`，单视角为 `single_rgb_obb_annotation_v1`。
- `reviewed=false` 表示未标注；`reviewed=true` 且 `status=no_obstacles` 才表示确认无障碍。
- 旧深度像素框 `dual_depth_obb_annotation_v1` 不能当作 RGB 标注续写，需另存数据重新标注。

[训练数据导出](QUEST3_DEPTH_ALIGNMENT.md#四视角-lerobot-训练数据导出) ·
[生成旧式 selection.json](../real_scripts/README.md#双视角人工-obb--体素验证)。
