# cv-tracker

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Tests](https://img.shields.io/badge/tests-18%20passing-brightgreen.svg)](#测试)
[![GitHub stars](https://img.shields.io/github/stars/closing2005/closing2005-cv-tracker.svg)](https://github.com/closing2005/closing2005-cv-tracker/stargazers)

[English](README.md) | 中文 | [术语表 Glossary](GLOSSARY.md)

</div>

给静态摄像头用的实时多目标跟踪，一条命令跑完：检测 → 卡尔曼跟踪 → 区域/绊线计数 → 行为事件告警。**不用下载权重，不用装 SciPy**。

```
视频帧 → [检测器] → [多目标跟踪器] → [区域 / 绊线] → [行为事件] → CSV + 统计报告
            MOG2         ByteTrack 式         进出统计      徘徊告警
            运动检测     两阶段关联                       超速告警
                         + 卡尔曼滤波                     逆行告警
                         + 匈牙利算法                     聚集告警
```

## 这玩意儿跟别的跟踪器有啥不一样

- **两阶段关联，偷师 ByteTrack。** 第一轮先用高置信度的框去匹配；没匹配上的轨迹，第二轮拿*低*置信度的框再救一次。目标被挡了几帧又冒出来，不会顶个新 ID——ID 乱跳是朴素跟踪器最丢人的毛病，这里治住了。
- **卡尔曼滤波，等速模型**，过程噪声按目标大小自适应。大框容差大、小框容差小，不用每个场景调一遍参数。
- **匈牙利算法手搓的**（`src/hungarian.py`）：O(n³) 的 Kuhn-Munkres，矩形矩阵照样解，不引 SciPy，没有黑盒。
- **轨迹有完整的生命周期**：`候选 → 确认 → 丢失 → 删除`。闪烁一下的鬼影永远转不了正，只有确认过的轨迹才会被计数、被分析。
- **不只画框，还能看懂行为。** 徘徊、超速、逆行、聚集——全用轨迹历史算出来，不加载任何模型。每个事件都带证据（谁、哪一帧、测到多少），可查不是玄学。
- **自带体检报告。** 跑完输出碎片率和质量结论，数字靠不靠谱自己先说清楚。
- **检测器随便换。** 自带免权重的 MOG2（摄像头不动就能跑）；想上 YOLO，继承 `Detector` 类就行，跟踪器不用动一行。

## 三分钟上手

```bash
pip install -r requirements.txt

# 接摄像头，实时看效果
python -m src.main --source 0 --show

# 跑视频文件，不开窗口，结果存下来
python -m src.main --source traffic.mp4 --no-show --out runs/run1 --max-frames 900

# 跑测试，不需要摄像头
python -m pytest tests/ -q
```

## 一键演示

没摄像头？没关系。这个脚本会生成一段合成视频（两个目标，其中一个被挡 15 帧），完整跑一遍跟踪，并存下带框、编号、轨迹和绊线计数的成品视频：

```bash
python examples/synthetic_demo.py --out demo_out
# demo_out/input.mp4    - 合成的输入视频
# demo_out/tracked.mp4  - 画好框/编号/轨迹/计数的成品
# demo_out/summary.yaml - 会话摘要
```

## 性能

640×480 @ 30fps 合成视频，纯 CPU（无 GPU、无权重）实测：

| 阶段 | 每帧耗时 |
|---|---|
| 检测（MOG2 + 形态学 + 轮廓） | ~5.5 ms |
| 跟踪（卡尔曼预测 + 两阶段匈牙利） | ~0.4 ms |
| 区域 + 事件 | <0.1 ms |

实时无压力；检测是大头，真要花预算就花在 DNN 检测器上。

跑完在 `--out/` 里拿结果：

| 文件 | 里面是啥 |
|---|---|
| `trajectories.csv` | 每条确认轨迹逐帧的 `frame, track_id, x1, y1, x2, y2, cx, cy, vx, vy` |
| `summary.yaml` | 目标数、平均速度、事件统计、轨迹质量结论 |

## 跑起来长这样

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

每 60 帧打印各阶段耗时，时间花在哪一目了然。

## 里面是怎么转的

**检测。** `MotionDetector` 用 MOG2 做背景减除，去掉阴影，开闭运算去噪，按面积过滤轮廓。打分看两样：blob 有多大、填充实不实（实心目标比丝状噪声分高），分高的先排前面——方便后面匹配先挑好的。

**跟踪。** 每帧先把所有存活轨迹用卡尔曼往前推一步，然后：

1. *第一轮*——轨迹和高置信度检测框按 IoU 算代价，匈牙利算法求全局最优匹配（`max_iou_dist` 卡掉离谱的）。
2. *第二轮*——上一轮没匹配上的轨迹，用低置信度框再救一次。这是 ByteTrack 的核心思想：遮挡目标露出的半个身子，也比一个新 ID 强。
3. 还没匹配上的高置信度框，开新轨迹（先当候选，连续 `n_init` 次命中才转正）。

跟丢的轨迹靠运动预测再续 `max_lost` 帧，短遮挡碎不了身份。

**区域和绊线。** 多边形区域数进出、算每个目标的停留时长。有向绊线报穿越*方向*（站在 p1 朝 p2 看：从左往右穿算 "forward"）。

**行为事件。** 四个检测器，只看轨迹历史：

| 事件 | 啥情况触发 | 关键参数 |
|---|---|---|
| 徘徊 | 在小范围里待太久 | `radius_px`、`min_frames` |
| 超速 | 平滑速度超标 | `max_px_per_sec` |
| 逆行 | 运动方向跟设定流向反着来 | `flow`、`cos_thresh` |
| 聚集 | 一个区域里目标太多 | `max_tracks`、`min_frames` |

单个目标每种事件只报一次，不刷屏；每次都带证据。

## 配置都在 config.yaml，改数不用改代码

```yaml
detector:
  min_area: 500        # 小于这个面积(px^2)的 blob 直接扔掉
  history: 500         # MOG2 背景历史帧数
  var_threshold: 16    # MOG2 灵敏度，越小越灵敏

tracker:
  high_thresh: 0.5     # 第一轮关联阈值
  low_thresh: 0.15     # 第二轮抢救阈值
  max_iou_dist: 0.7    # 代价高于此值(1 - IoU)的匹配不要
  n_init: 3            # 候选转正需要的连续命中数
  max_age: 30          # 候选轨迹删掉前的帧数
  max_lost: 60         # 丢失轨迹删掉前的帧数

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
  wrong_way: { enabled: true, flow: [1, 0] }   # 预期大家都往右走
  crowding:  { enabled: false, max_tracks: 5 }
```

## 测试

18 个测试，全是合成数据，不用摄像头：

```bash
python -m pytest tests/ -q
```

- `test_hungarian.py` —— 方阵/矩形矩阵最优性、平局、空矩阵
- `test_tracker.py` —— 两个目标跟 60 帧；5 帧遮挡靠弱检测救回来，**ID 不跳变**；闪烁鬼影永不转正；丢了的轨迹最终会被删掉
- `test_zones.py` —— 区域进出 + 停留帧数；绊线方向；消失的轨迹会被遗忘
- `test_events.py` —— 每种行为事件恰好触发一次，老实目标零误报

## 目录结构

```
src/
  detector.py   运动检测器(MOG2) + 给 DNN 留的 Detector 接口
  kalman.py     单目标卡尔曼滤波，过程噪声自适应
  hungarian.py  手搓 Kuhn-Munkres，不依赖 SciPy
  tracker.py    多目标跟踪：两阶段关联 + 轨迹状态机
  zones.py      多边形区域(进出/停留) + 有向绊线
  events.py     徘徊 / 超速 / 逆行 / 聚集检测
  analytics.py  轨迹 CSV 落盘 + 会话摘要 + 质量自检
  main.py       命令行入口、各阶段计时、实时可视化
tests/          18 个合成测试，不用摄像头
config.yaml     阈值、区域、绊线、事件参数全在这
```

## 能拿去干啥

- 商店客流计数、停留时长分析
- 门口、闸机进出计数
- 无人值守区域的徘徊告警
- 扶梯、走廊、单行道的逆行检测
- 摄像头俯视的固定场景计数（停车场、仓库、工地）

## 先说清楚的局限

- 自带的是运动检测：**摄像头必须静止**，伪装色、挪得很慢的目标搞不定。要通用就接 DNN 检测器，接口留好了。
- 没有外观重识别——身份靠运动预测扛短遮挡，扛不住长遮挡。这是为"零权重、纯 CPU 实时"有意做的取舍。
- 速度距离都是像素单位，要米的话自己标定。

## 后面想加的

- [ ] YOLOv8 检测器插件（`cv2.dnn`，除了权重不引新依赖）
- [ ] 外观 embedding，长遮挡也能认回来
- [ ] 多摄像头接力
- [ ] 实时计数和事件流的 Web 面板

## 参与贡献

欢迎报 bug 和提需求——请用 Issue 模板，方便复现。完整规范见 [CONTRIBUTING.md](CONTRIBUTING.md)（代码风格、测试要求、双语文档规则）。

## 协议

MIT —— 见 [LICENSE](LICENSE)。
