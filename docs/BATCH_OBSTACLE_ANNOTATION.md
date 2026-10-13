# Quest3 双视角 RGB 障碍物批量标注

`real_scripts/batch_obstacle_annotation.py` 离线逐帧读取任务目录中的 HDF5，人工只在
front、side **录制的 RGB 图**上画二维旋转框（OBB）。原生深度无需人工标注，也不要求
与 RGB 同分辨率；后续把框映射到三维时，才使用标定后的 RGB 对齐深度。标注坐标是各自
RGB 的像素坐标，不是深度像素坐标，也不是直接可用于碰撞计算的 3D OBB。

输入必须有 `timestamps_ns [N]` 和两路 `observations/images/{front,side}/rgb`，
每路 RGB 可以是 `[N,H,W,3] uint8` 或逐帧 JPEG。深度可为原生、已对齐或缺失；
脚本仅记录其来源以供审计。CoTracker 三维点提取使用已经 RGB 对齐的深度，
仍需录制时的 front RGB 到基座标定，见 [已对齐深度契约](QUEST3_DEPTH_ALIGNMENT.md)。

先检查任务目录，再启动本地页面：

```bash
conda run -n safety python real_scripts/batch_obstacle_annotation.py \
  --task-dir ../quest3_collect/data/dual_vr_episodes/你的任务目录 --inspect
conda run -n safety python real_scripts/batch_obstacle_annotation.py \
  --task-dir ../quest3_collect/data/dual_vr_episodes/你的任务目录
```

浏览器打开 `http://127.0.0.1:8770`。每帧分别在 front / side RGB 上画框、旋转，
同一物体在两路使用相同 `object_id`；无障碍时按“标记无障碍”。可逐帧导航、复制上一帧框、
断点续标。保存才写回原 HDF5，原图、深度、状态、动作和根属性不变。

标注保存到 `annotations/obstacle_obbs`，schema 为 `dual_rgb_obb_annotation_v2`，
`annotation_image=recorded_rgb`。其中 `frames_json[i]` 含两路框、原帧索引及时间戳；
`reviewed[i]=False` 表示尚未人工确认，不能当作无障碍负样本。
`status=no_obstacles` 且 `reviewed[i]=True` 才是明确的空场景。图像坐标下 `angle_deg`
顺时针为正。多窗口编辑使用 `revision` 防止旧版本覆盖新标注。

旧版 `dual_depth_obb_annotation_v1` 的框是深度像素坐标，不能静默改称 RGB 框。
脚本会拒绝在同一 group 上续标；应保留旧文件，另建新数据副本或显式迁移后重新标注。

训练 LeRobot 导出见 [Quest3 深度与数据集流程](QUEST3_DEPTH_ALIGNMENT.md#四视角-lerobot-训练数据导出)。
