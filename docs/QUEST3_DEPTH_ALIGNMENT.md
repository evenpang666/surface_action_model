# Quest3 深度与数据导出

## 预处理输入检查

front 深度必须是已对齐 RGB 的 `uint16 [N,H,W]`，尺寸与 RGB 一致，且满足以下任一声明：

- `config_yaml.cameras.front.depth_topic` 包含 `aligned_depth_to_color`。
- depth dataset 的 `depth_alignment` / `coordinate_system` / `annotation_image` 明确声明 RGB 对齐。

相同尺寸不能代替对齐声明。原生 `depth/image_rect_raw`、`depth/image_raw` 不能直接用于此流程；
应在采集端保存 RGB 对齐深度及其元数据。预处理不会再次对齐或修改原深度。

深度单位依次取 `--depth-unit-m`、dataset 的 `depth_scale_m` / `depth_unit_m`；
历史对齐 Z16 缺省为 0.001 m/单位。比例不同时显式传真实值。

## 提取 CoTracker 三维点

先按[预处理步骤](IMPLEMENTATION_GUIDE.md#数据预处理)生成 geometry shard，再运行：

```bash
python scripts/preprocess_cotracker_surface.py \
  --episode /path/to/episode.hdf5 \
  --geometry-shard outputs/gates/episode_00000.shard \
  --calibration configs/hardware/calibration/left_base_to_front_camera.yaml \
  --output outputs/tracks/episode_00000.npz \
  --preview outputs/tracks/episode_00000.png \
  --visualization outputs/tracks/episode_00000.html
```

`--calibration` 使用录制时固定 front RGB 相机到基座的外参；不能传 depth→RGB 或腕部相机外参。
输出查看 PNG/HTML；NPZ 的 `depth_alignment_metadata` 记录输入方式与单位。

## 四视角 LeRobot 训练数据导出

先生成带实测点的 shard，按需完成[RGB 障碍标注](BATCH_OBSTACLE_ANNOTATION.md)：

```bash
python scripts/export_quest3_training_lerobot.py \
  --shard outputs/surface_action/episode_00000.shard \
  --episode /path/to/episode.hdf5 \
  --output outputs/quest3_lerobot/episode_00000 \
  --repo-id local/quest3_dual_surface_training --require-reviewed
```

输入需 front、side、left_wrist、right_wrist 四路 RGB。省略 `--require-reviewed` 可在标注前导出预览；
未审核的空框不能作无障碍标签。标注更新后重新导出。

此导出用于数据交换；当前 PI0.5 训练入口仍读取 `.shard`，默认只输入 front RGB。
回放用的绝对关节 LeRobot 数据集另用[回放转换命令](MODEL_FREE_OBSTACLE_VALIDATION.md#准备数据与标注)，两者不要混用。
