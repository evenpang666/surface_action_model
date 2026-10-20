# 实验操作与结果记录

主线使用 [训练与推理入口](IMPLEMENTATION_GUIDE.md)；几何避障先做[无模型回放](MODEL_FREE_OBSTACLE_VALIDATION.md)。
完整论文对照、文献与大表模板保留在[实验设计原稿](legacy/EXPERIMENT_TABLE_DESIGN_2026-10-18.md)。
本页只列执行顺序与记录口径，未运行的结果填「待测」。

## 执行顺序

1. 固定训练/验证/测试划分、任务、布局、随机种子、时限和碰撞判据。
2. 训练普通动作 baseline 与 action + point 模型，记录权重和配置。
3. 在相同任务、起始状态和障碍布局下运行全部方法。
4. 保存逐 episode 结果、轨迹和耗时，再汇总；仿真与实机分表。
5. 核心对照完成后，再扩展障碍数量、动态障碍及新任务。

## 核心对照

| 方法 | 设置 |
| --- | --- |
| 动作 baseline | 仅策略动作，无新增安全模块 |
| Current-body CBF | 当前机器人几何 + CBF |
| FK-lookahead CBF | 名义 action chunk 的 FK 前视 + CBF |
| 联合模型，无安全层 | action + point checkpoint，关闭安全修正 |
| 联合模型，当前几何 | 同一 checkpoint + Current-body CBF |
| 完整模型 | 预测表面点 + CBF + FK 复核 |

比较控制层时共享感知、几何、安全距离、动作限幅与算力预算。
近期论文适配方法另行实现并标注改动；仓库不保证已有所有对照。

## 记录结果

| 指标 | 统计方法 |
| --- | --- |
| SR | 完成任务的 episode / 全部完整 episode |
| SSR | 完成任务且无禁止碰撞的 episode / 全部完整 episode |
| CR-env / CR-self / CR-held | 环境、机器人自身、抓持物禁止碰撞的 episode 比例，分别记录 |
| d05 | 各 episode 最小有符号间距的第 5 百分位，mm |
| Timeout | 超时比例，另记感知、求解、停机等失败原因 |
| 完成时间 | 成功 episode 的时间中位数/IQR，同时报告 SR |
| QP-fail | 失败 QP / 实际 QP 调用；没有调用时为 N/A |
| L95 | 从观测可用到动作获准发送的端到端时延 P95，ms |
| Point-ADE / FDE | 有效对应点的平均/末时刻误差，mm；同时报告覆盖率 |

真实碰撞依据接触记录或独立标注，不能用预测点进入感知框替代。
回放抵达终点只表示轨迹完成，不能当作语义任务成功。

逐 episode 至少保存：

```text
method_id, checkpoint, train_seed, eval_seed, task_id, layout_id,
obstacle_count, success, collision_env, collision_self, collision_held,
min_distance_mm, episode_time_s, timeout, termination_reason,
intervention_ticks, control_ticks, qp_calls, qp_failures,
latency_samples_ms, complete_episode
```

结果表：

| 方法 | 任务 | 障碍数 | episodes | SR | SSR | CR-env | CR-self | d05 mm | Timeout | L95 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 待测 | 待测 | 0 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

## 当前入口的限制

- `scripts/evaluate_pi05_safety_decoder_on_libero.py` 是历史独立点模型评估入口，不能直接当作主线双臂 checkpoint 评估器。
- 该入口的 `--scene-obstacle` 只支持 `none` / `wine_bottle`；没有 `--obstacle-count`，多障碍实验需补布局生成。
- 正式统计需补全 episode 成功检查、全部禁止接触对及物理子步接触累计。
- 不要把 `--max-samples` 提前截断当作完整 episode；缺少数据的结果保留「待测」。
