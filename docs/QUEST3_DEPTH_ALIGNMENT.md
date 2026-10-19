# Quest3 已对齐深度的预处理契约

当前 Quest3 HDF5 的 front 深度已经是 RGB 对齐的 Z16 图。
`scripts/preprocess_cotracker_surface.py` 直接读取录制的 `observations/images/front/depth`，
按深度单位换算为米，在同一像素网格中做 FK 种子筛选与 CoTracker 点的三维反投影。
**预处理不再次对齐、重采样、补洞或覆盖原 HDF5 深度。** 它仍需要录制时有效的
front RGB 相机内参及其到共同 `left_base` 坐标系的外参。

输入的 depth 必须为 `uint16 [N,H,W]`，`H,W` 与录制 RGB 相同，且 HDF5 中的
`config_yaml.cameras.front.depth_topic` 含 `aligned_depth_to_color`，或深度 dataset
的 `depth_alignment` / `coordinate_system` / `annotation_image` 属性声明为 RGB 对齐。
仅凭相同尺寸不能证明对齐。标为 `/depth/image_rect_raw`、`/depth/image_raw` 或
原生深度的输入会被拒绝，避免错误的三维点监督。

深度单位优先使用 `--depth-unit-m`，其次是 dataset 的 `depth_scale_m` / `depth_unit_m`；
历史对齐 Z16 缺少该属性时沿用 0.001 m/单位。若采集端使用了不同单位，应显式记录或
传入实际比例。输出 NPZ 的 `depth_alignment_metadata` 会标明
`input_mode=recorded_aligned`、`rasterization=none` 和使用的单位。

```bash
python scripts/preprocess_cotracker_surface.py \
  --episode ../quest3_collect/data/dual_vr_episodes/任务/episode_00000.hdf5 \
  --geometry-shard outputs/gates/episode_00000.shard \
  --calibration configs/hardware/calibration/left_base_to_front_camera.yaml \
  --output outputs/tracks/episode_00000.npz \
  --preview outputs/tracks/episode_00000.png \
  --visualization outputs/tracks/episode_00000.html
```

`--calibration` 描述固定 front **color optical frame** 到基座的变换，不是深度到
RGB 的外参。front/side 是固定 D455；left/right wrist 当前只有 RGB。项目统一以
`left_base` 存储两臂三维点，右臂通过基座间变换进入该坐标系。腕部相机若以后增加
三维监督，需要逐帧 FK 与自己的手眼标定，不能套用 front 外参。

本次检查中，`../quest3_collect/data` 现有 15 个 HDF5 的 front 深度 topic 为
`aligned_depth_to_color/image_raw` 且与 RGB 同尺寸；但相邻采集项目当前
`config/dual_arm.yaml` 仍配置 `depth/image_rect_raw`。后续录制若沿用该配置，
预处理会拒绝，采集端需先恢复对齐 topic 及对应元数据。

## 四视角 LeRobot 训练数据导出

先用已对齐深度提取 CoTracker 点，把 `--measured-flow-npz` 交给
`scripts/preprocess_quest3_hdf5.py` 生成测量点 shard；可随后在原 HDF5 中完成
[front/side RGB 障碍物标注](BATCH_OBSTACLE_ANNOTATION.md)。四路 RGB、任务文字、
关节状态、动作、点云及可见掩码、逐帧 RGB OBB 可导出到 LeRobot：

```bash
conda run -n safety python scripts/export_quest3_training_lerobot.py \
  --shard outputs/surface_action/episode_00000.shard \
  --episode ../quest3_collect/data/dual_vr_episodes/任务/episode_00000.hdf5 \
  --output outputs/quest3_lerobot/episode_00000 \
  --repo-id local/quest3_dual_surface_training --require-reviewed
```

省略 `--require-reviewed` 可在标注前导出预览；`annotation.reviewed=0` 的空框只是
占位，不能当作无障碍。标注更新后需重新导出。LeRobot 保存原时间戳特征；其 `fps`
是均匀索引，录制间隔不规则时训练前仍需按时间戳处理。

当前 `scripts/train_pi05_ur7e_surface_pytorch.py` 直接读取项目 shard，默认只把
front 映射到 `base_0_rgb`。四视角 LeRobot 是数据交换产物，尚未成为该脚本的输入；
当前 PI0.5 图像接口仅有三个槽位。
