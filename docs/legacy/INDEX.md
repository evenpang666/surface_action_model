# 历史入口及保留理由

这些文件不属于 Surface Action Model 默认链路；路径暂时保持兼容，避免破坏相互引用和用户现有实验命令。

| 分组 | 入口/代码 | 状态 |
| --- | --- | --- |
| 旧位移回归 | PI05SafetyPytorch、旧 checkpoint | 服务可读、执行器拒绝物理执行，保留回放 |
| 独立点模型 | safety_module/point_decoder.py、safety_flow_point_model.py、serve_pi05_prefix_policy.py、train_pi05_safety_flow_point_model.py | 历史实验；新联合模型在 OpenPI models_pytorch/surface_action_pytorch.py |
| 仿真 | scripts 中 libero / robosuite / upright_blocks / SDF / safe-space 脚本 | 基准与实验，非全局死代码 |
| 旧 runtime | ur7e_inference/、run_legacy_pi05_left_policy.py | 独立历史接口，不是安全部署入口 |
| 硬件维护 | 标定、重建、demo、collector、Pika SDK/libsurvive | 按运维用途保留，不能整体删除 |

已清理的确定冗余：训练/服务 tokenizer 合并；主感知未使用 tool_proxy_points/box_surface 和相关未用导入移除；失效工具 URDF/margin 参数移除；重复 workspace 筛选移除。

原 README 快照为 README_2026-09-22.md，其中命令/工具类型可能已经过时，不用作当前部署说明。
