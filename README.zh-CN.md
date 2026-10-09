# cv-tracker

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Tests](https://img.shields.io/badge/tests-18%20passing-brightgreen.svg)](#测试)
[![GitHub stars](https://img.shields.io/github/stars/closing2005/closing2005-cv-tracker.svg)](https://github.com/closing2005/closing2005-cv-tracker/stargazers)

[English](README.md) | 中文 | [术语表 Glossary](GLOSSARY.md)

</div>

面向静态摄像头的实时多目标跟踪：检测 → 卡尔曼滤波跟踪 → 区域/绊线计数 → 行为事件告警，一条命令完成。**无需下载模型权重，无需安装 SciPy**。

```
视频帧 → [检测器] → [多目标跟踪器] → [区域 / 绊线] → [行为事件] → CSV + 统计报告
            MOG2         ByteTrack 式         进出统计      徘徊告警
            运动检测     两阶段关联                       超速告警
                         + 卡尔曼滤波                     逆行告警
                         + 匈牙利算法                     聚集告警
```

## 核心优势

- **两阶段关联（借鉴 ByteTrack）。** 第一轮以高置信度检测框进行匹配；未匹配的轨迹在第二轮中以*低*置信度检测框再次尝试匹配。被短暂遮挡后重新出现的目标会被原有轨迹认领，而非分配新 ID——ID 跳变是朴素跟踪器的主要缺陷，在此得到了有效抑制。
- **卡尔曼滤波（等速模型）**，过程噪声随目标尺寸自适应调整。大目标获得更宽的运动容差，小目标则更为严格，无需针对每个场景单独调参。
- **匈牙利算法手写实现**（`src/hungarian.py`）：O(n³) 的 Kuhn-Munkres 算法，支持矩形代价矩阵，不依赖 SciPy，无黑盒。
- **完整的轨迹生命周期管理**：`候选 → 确认 → 丢失 → 删除`。瞬时闪烁产生的伪影无法转正，只有经确认的轨迹才会被计数与分析。
- **行为事件理解，而不仅是画框。** 徘徊、超速、逆行、聚集——全部基于轨迹历史计算，无需额外模型。每个事件均附带证据（轨迹编号、帧号、测量值），结果可审计。
- **内置质量自检。** 会话摘要报告轨迹碎片率与质量结论，便于判断统计结果的可信度。
- **检测器可插拔。** 内置免权重的 MOG2 运动检测器（摄像头静止即可运行）；如需 YOLO 等深度检测器，继承 `Detector` 类即可接入，跟踪器无需改动。

## 快速上手

```bash
pip install -r requirements.txt

# 接入摄像头，实时可视化
python -m src.main --source 0 --show

# 处理视频文件，无界面运行，结果落盘
python -m src.main --source traffic.mp4 --no-show --out runs/run1 --max-frames 900

# 运行测试（无需摄像头）
python -m pytest tests/ -q
```

运行结束后在 `--out/` 中获取：

| 文件 | 内容 |
|---|---|
| `trajectories.csv` | 每条已确认轨迹逐帧的 `frame, track_id, x1, y1, x2, y2, cx, cy, vx, vy` |
| `summary.yaml` | 目标数量、平均速度、事件统计、轨迹质量结论 |

## 一键演示

无摄像头也可体验。以下脚本生成一段合成视频（两个目标，其中一个被遮挡 15 帧），执行完整跟踪流程，并输出带标注的结果视频：

```bash
python examples/synthetic_demo.py --out demo_out
# demo_out/input.mp4    - 合成输入视频
# demo_out/tracked.mp4  - 含检测框/编号/轨迹/绊线计数的输出视频
# demo_out/summary.yaml - 会话摘要
```

## 运行示例

```
frame 60: 2 tracks, detect 5.7ms, track 0.4ms
frame 120: 2 tracks, detect 5.1ms, track 0.3ms
---- session summary ----
tracks seen          : 4
avg lifetime (frames): 40.8
avg speed (px/s)     : 68.6
fragmented tracks    : 1
track quality        : good
events               : none
saved trajectories + summary to runs/run1
```

每 60 帧输出各阶段耗时，便于定位性能瓶颈。

## 性能

640×480 @ 30fps 合成视频，纯 CPU（无 GPU、无模型权重）实测：

| 阶段 | 每帧耗时 |
|---|---|
| 检测（MOG2 + 形态学 + 轮廓提取） | ~5.5 ms |
| 跟踪（卡尔曼预测 + 两阶段匈牙利匹配） | ~0.4 ms |
| 区域与事件 | <0.1 ms |

满足实时性要求；检测阶段耗时占比最高，如需进一步优化应优先替换为 DNN 检测器。

## 工作原理

**检测。** `MotionDetector` 采用 MOG2 背景减除，滤除阴影后经开闭运算去噪，按面积筛选轮廓。评分综合目标尺寸与填充率（实心目标得分高于丝状噪声），按分数降序排列，以利于匹配阶段优先处理高置信度目标。

**跟踪。** 每帧首先对全部存活轨迹执行卡尔曼预测，随后：

1. *第一轮*——轨迹与高置信度检测框按 IoU 代价进行全局最优匹配（匈牙利算法，`max_iou_dist` 剔除不合理匹配）。
2. *第二轮*——未匹配的轨迹以低置信度检测框再次尝试匹配。这是 ByteTrack 的核心思想：遮挡目标的微弱观测仍优于分配新 ID。
3. 仍未匹配的高置信度检测框创建新的候选轨迹，连续 `n_init` 次命中后转为确认状态。

丢失的轨迹依靠运动预测维持 `max_lost` 帧，短时遮挡不会导致身份碎裂。

**区域与绊线。** 多边形区域统计进入/离开数量及各轨迹停留时长。有向绊线报告穿越*方向*（以 p1 为起点面向 p2：自左向右穿越记为 "forward"）。

**行为事件。** 四个检测器仅基于轨迹历史工作：

| 事件 | 触发条件 | 关键参数 |
|---|---|---|
| 徘徊 | 轨迹在较小半径内停留过久 | `radius_px`、`min_frames` |
| 超速 | 平滑速度超过阈值 | `max_px_per_sec` |
| 逆行 | 运动方向与设定流向相反 | `flow`、`cos_thresh` |
| 聚集 | 单个区域内轨迹数超限 | `max_tracks`、`min_frames` |

单个轨迹的每种事件仅触发一次，避免重复告警；所有事件均附带证据。

## 配置

全部参数集中于 `config.yaml`，调整无需修改代码：

```yaml
detector:
  min_area: 500        # 过滤小于此面积(px^2)的连通块
  history: 500         # MOG2 背景历史帧数
  var_threshold: 16    # MOG2 灵敏度，数值越小越灵敏

tracker:
  high_thresh: 0.5     # 第一轮关联阈值
  low_thresh: 0.15     # 第二轮抢救阈值
  max_iou_dist: 0.7    # 代价高于此值(1 - IoU)的匹配将被拒绝
  n_init: 3            # 候选转为确认所需的连续命中数
  max_age: 30          # 候选轨迹删除前的最大帧数
  max_lost: 60         # 丢失轨迹删除前的最大帧数

zones:
  - name: "door"
    polygon: [[40, 200], [280, 200], [280, 440], [40, 440]]

tripwires:
  - name: "gate"
    p1: [160, 60]
    p2: [160, 420]

events:
  loitering: { enabled: true, radius_px: 40, min_frames: 150 }
  speeding:  { enabled: true, max_px_per_sec: 300 }
  wrong_way: { enabled: true, flow: [1, 0] }   # 预期运动方向为向右
  crowding:  { enabled: false, max_tracks: 5 }
```

## 测试

18 个测试，全部基于合成数据，无需摄像头：

```bash
python -m pytest tests/ -q
```

- `test_hungarian.py` —— 方阵/矩形矩阵的最优性、平局、空矩阵情形
- `test_tracker.py` —— 双目标 60 帧跟踪；5 帧遮挡经弱检测抢救后 **ID 无跳变**；闪烁伪影无法转正；丢失轨迹最终被删除
- `test_zones.py` —— 区域进入/离开与停留帧数；绊线方向判定；消失轨迹的状态清理
- `test_events.py` —— 各类行为事件精确触发一次，正常目标无误报

## 项目结构

```
src/
  detector.py   运动检测器(MOG2)，及供 DNN 接入的 Detector 接口
  kalman.py     单目标卡尔曼滤波，过程噪声自适应
  hungarian.py  手写 Kuhn-Munkres 实现，不依赖 SciPy
  tracker.py    多目标跟踪器：两阶段关联 + 轨迹状态机
  zones.py      多边形区域(进出/停留统计) + 有向绊线
  events.py     徘徊 / 超速 / 逆行 / 聚集检测器
  analytics.py  轨迹 CSV 落盘 + 会话摘要 + 质量自检
  main.py       命令行入口、各阶段计时、实时可视化
tests/          18 个合成测试，无需摄像头
config.yaml     阈值、区域、绊线、事件参数
```

## 应用场景

- 零售门店客流计数与停留时长分析
- 出入口、闸机的人员进出计数
- 无人值守区域的徘徊告警
- 扶梯、走廊、单行道的逆行检测
- 固定俯视场景的计数（停车场、仓库、工地）

## 已知局限

- 内置检测器基于运动检测：**要求摄像头静止**，难以处理伪装色目标或移动极缓慢的目标。通用场景建议接入 DNN 检测器，接口已预留。
- 未实现外观重识别——身份依靠运动预测度过短时遮挡，无法应对长时遮挡。这是为保持零权重、纯 CPU 实时性而做的权衡。
- 速度与距离均为像素单位，如需真实世界单位请自行标定。

## 后续规划

- [ ] YOLOv8 检测器插件（`cv2.dnn`，除模型权重外不引入新依赖）
- [ ] 外观 embedding，支持长时遮挡后的重识别
- [ ] 多摄像头轨迹接力
- [ ] 实时计数与事件流的 Web 面板

## 参与贡献

欢迎提交 Issue 与 PR，请使用 Issue 模板以便复现。完整规范见 [CONTRIBUTING.md](CONTRIBUTING.md)（代码风格、测试要求、双语文档规则）。

## 协议

MIT —— 见 [LICENSE](LICENSE)。
