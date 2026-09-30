# 当前工作站重新标定

用户确认 front 相机已移动且 side 尚未标定。2026-09-10 等历史外参不可用于当前桌面。
配置保持 camera_calibration_valid=false，直到新的三份外参通过质量检查。

需要一块固定的 ChArUco 板、其横纵格数/格边长/marker 边长/字典，以及已校准的 TCP 探针。
板必须在两相机中可见，整个左右臂触点过程不得移动板或相机。操作者手动移动机械臂；
脚本只读取 RTDE receive，不建立 RTDE control 或夹爪连接。

1. 以左臂 192.168.124.61、front=260922305390、side=260922303231 运行
   `real_scripts/calibrate_ur7e_realsense_integrated.py`，将输出放入新的会话目录。
2. 使用同一固定板，以右臂 192.168.124.60、front=260922305390 再运行一次。
3. 每个会话执行时需要传 `--serials`、`--robot-ip`、`--output-dir`、`--squares-x`、
   `--squares-y`、`--square-length-m`、`--marker-length-m`、`--dictionary`。
   不提供虚构尺寸示例；尺寸必须和实际打印板一致。
4. 确认触点拟合 RMS <=3mm、相机重投影 RMS <=0.8px，且融合结果正确。
5. 使用 `python scripts/install_surface_calibration.py --left-session <新左会话/camera_calibration.json> --right-session <新右会话/camera_calibration.json>`。
   它校验实测结果，生成 front_current.yaml、side_current.yaml、right_front_current.yaml，
   更新 hardware 标定引用及有效标志；不会覆盖已有同名标定文件。
6. 重跑只读 TCP 偏移核验、双视角点云重建，并核对 mesh hash 与 checkpoint。
   相对 base/TCP 几何变化后应重建 shard/训练，不能放宽 hash 校验复用不匹配标签。

标定脚本需要 cv2.aruco；新 runtime 环境应使用匹配版本的 opencv-contrib-python 替代 opencv-python。
本轮未取得新标定，也没有触发物理运动。
