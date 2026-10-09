# cv-tracker

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Tests](https://img.shields.io/badge/tests-18%20passing-brightgreen.svg)](#测试)

[English](README.md) | 中文 | [术语表 Glossary](GLOSSARY.md)

</div>

面向静态摄像头的实时多目标跟踪。检测 → 卡尔曼滤波跟踪 → 区域与绊线 → 行为事件——一个 CLI 全搞定，**零权重文件**，**零 SciPy**。

```
frame → [检测器] → [多目标跟踪器] → [区域 / 绊线] → [事件] → CSV + 统计摘要
              MOG2         ByteTrack 式         进出计数      徘徊
              运动检测     两阶段关联                       超速
                           + 卡尔曼                         逆行
                           + 匈牙利算法                     聚集
```

## 为什么这不是又一个玩具跟踪器

- **两阶段关联（ByteTrack 思想）。** 高置信度检测先匹配；剩下的轨迹再用*低*置信度检测抢救一次。被短暂遮挡的目标会被救回来，而不是顶着新 ID 重生——这是朴素跟踪器 ID 跳变的最大来源，这里处理掉了。
- **卡尔曼滤波（等速模型）**，过程噪声按目标尺寸自适应。大而快的框获得更宽的容差，小而远的不用——不用按场景调参。
- **匈牙利算法手写实现**（`src/hungarian.py`）：O(n³) Kuhn-Munkres，支持矩形矩阵，不依赖 SciPy，没有黑盒。
- **能杀死闪烁鬼影的轨迹状态机**：`Tentative → Confirmed → Lost → Deleted`。只有 confirmed 的轨迹才会被上报、计数、送进事件检测。
- **行为事件，不只是框。** 徘徊、超速、逆行、聚集——全部只用轨迹历史算出来，不需要额外模型。每个事件都携带证据（轨迹 id、帧号、测量值），结果可审计，不是玄学。
- **跟踪器自检。** 会话摘要里报告碎片率和质量结论，让你知道这波数字靠不靠谱，还是该回去调参。
- **检测器可插拔。** 自带免权重的 MOG2 运动检测（静态摄像头开箱即用）；继承 `Detector` 就能通过 `cv2.dnn` 接 YOLO，跟踪器一行不用动。

## 快速上手

```bash
pip install -r requirements.txt

# 摄像头，实时可视化
python -m src.main --source 0 --show

# 视频文件，无头跑，保存轨迹 + 摘要
python -m src.main --source traffic.mp4 --no-show --out runs/run1 --max-frames 900

# 跑测试（不需要摄像头）
python -m pytest tests/ -q
```

`--out/` 里会生成：

| 文件 | 内容 |
|---|---|
| `trajectories.csv` | 每个 confirmed 轨迹的 `frame, track_id, x1, y1, x2, y2, cx, cy, vx, vy` |
| `summary.yaml` | 计数、速度、事件，以及轨迹质量结论 |

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

每 60 帧打印各阶段耗时，瓶颈在哪一目了然。

## 原理

**检测。** `MotionDetector` 跑 MOG2 背景减除，去阴影，开闭运算去噪，按面积过滤轮廓。每个 blob 按尺寸和填充率打分（实心目标分数高于丝状噪声），按分数从大到小排——方便匹配器优先处理。

**跟踪。** 每帧先对所有存活轨迹做卡尔曼预测，然后：

1. *第一阶段*——轨迹和高置信度检测按 IoU 代价做最优匹配（匈牙利算法，`max_iou_dist` 截断）。
2. *第二阶段*——没匹配上的轨迹，用低置信度检测再救一次。这是 ByteTrack 的核心思想：遮挡目标的一瞥弱检测，也比一个新 ID 强。
3. 没匹配上的高置信度检测生成新的 tentative 轨迹；连续 `n_init` 次命中才转正为 confirmed。

丢失的轨迹靠运动预测续命 `max_lost` 帧，短暂遮挡不会碎掉身份。

**区域与绊线。** 多边形区域统计进出数、每个轨迹的停留时长。有向绊线报告穿越*方向*（站在 p1 面向 p2：从左向右穿算 "forward"）。

**事件。** 四个检测器只用轨迹历史：

| 事件 | 触发条件 | 关键参数 |
|---|---|---|
| 徘徊 | 轨迹在小半径内停留过久 | `radius_px`、`min_frames` |
| 超速 | 平滑速度超限 | `max_px_per_sec` |
| 逆行 | 运动方向与设定流向相反 | `flow`、`cos_thresh` |
| 聚集 | 单个区域内轨迹数超限 | `max_tracks`、`min_frames` |

每个检测器对单个轨迹只触发一次（不刷屏），每个事件都带证据。

## 配置

所有阈值都在 `config.yaml` 里，调参不用改代码：

```yaml
detector:
  min_area: 500        # 小于此面积(px^2)的 blob 忽略
  history: 500         # MOG2 背景历史（帧数）
  var_threshold: 16    # MOG2 灵敏度；越小越灵敏

tracker:
  high_thresh: 0.5     # 第一阶段关联阈值
  low_thresh: 0.15     # 第二阶段抢救阈值
  max_iou_dist: 0.7    # 代价高于此值(1 - IoU)的匹配拒绝
  n_init: 3            # Tentative -> Confirmed 需要的连续命中数
  max_age: 30          # Tentative 轨迹删除前的帧数
  max_lost: 60         # Lost 轨迹删除前的帧数

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
  wrong_way: { enabled: true, flow: [1, 0] }
  crowding:  { enabled: false, max_tracks: 5 }
```

## 测试

18 个测试，全用合成数据——不需要摄像头：

```bash
python -m pytest tests/ -q
```

- `test_hungarian.py` —— 方阵/矩形矩阵最优性、平局、空矩阵
- `test_tracker.py` —— 两个目标跟踪 60 帧；5 帧遮挡经弱检测抢救**无 ID 跳变**；闪烁鬼影永不转正；丢失轨迹最终被删除
- `test_zones.py` —— 区域进出 + 停留帧数；绊线方向；消失的轨迹被遗忘
- `test_events.py` —— 每个行为事件恰好触发一次，老实目标零误报

## 项目结构

```
src/
  detector.py   运动检测器(MOG2) + 供 DNN 插拔的 Detector 接口
  kalman.py     单目标卡尔曼滤波，过程噪声自适应
  hungarian.py  手写 Kuhn-Munkres，不依赖 SciPy
  tracker.py    多目标跟踪器：两阶段关联 + 轨迹状态机
  zones.py      多边形区域(进出/停留) + 有向绊线
  events.py     徘徊 / 超速 / 逆行 / 聚集检测器
  analytics.py  轨迹 CSV 记录 + 会话摘要 + 质量检查
  main.py       CLI、各阶段计时、实时可视化
tests/          18 个合成测试，不需要摄像头
config.yaml     所有阈值、区域、绊线、事件参数
```

## 适用场景

- 零售客流计数与停留分析
- 门口、闸机的进出计数
- 无人区域的徘徊告警
- 扶梯、走廊、单行道的逆行检测
- 静态摄像头俯视场景的计数（停车场、仓库）

## 局限（实话）

- 自带检测器是运动检测：需要**摄像头静止**，对伪装色或极慢目标吃力。通用场景请插 DNN 检测器。
- 没有外观（ReID）特征——身份靠运动预测扛过短遮挡，扛不过长遮挡。这是为零权重、纯 CPU 实时有意做的取舍。
- 速度和距离都是像素单位：要真实单位请自己标定。

## Roadmap

- [ ] YOLOv8 DNN 检测器插件（`cv2.dnn`，除权重外零新增依赖）
- [ ] 外观 embedding，应对长遮挡重识别
- [ ] 多摄像头接力
- [ ] 实时计数与事件流的 Web 面板

## 协议

MIT —— 见 [LICENSE](LICENSE)。
