# Glossary 术语表

[English](README.md) | [中文](README.zh-CN.md)

Terms used in this project, in plain language. 本项目用到的术语，一句话说清。

| English | 中文 | What it means |
|---|---|---|
| Multi-object tracking (MOT) | 多目标跟踪 | Detecting and following multiple moving targets across video frames. 在视频帧序列里检测并持续跟住多个运动目标。 |
| Detection | 检测 | Finding "something is here" boxes in a single frame. 在单帧里找出"这里有东西"的框。 |
| Detector | 检测器 | The component that produces detections (e.g. MOG2, YOLO). 产出检测框的模块，如 MOG2、YOLO。 |
| Track | 轨迹 | One target's identity + history over time. 一个目标的身份及其随时间的历史。 |
| Trajectory | 运动轨迹 | The path a track's center has traveled. 轨迹中心走过的路径。 |
| Association | 关联 / 匹配 | Deciding which detection belongs to which track. 判定每个检测框属于哪条轨迹。 |
| Kalman filter | 卡尔曼滤波 | Predicts where a moving target will be next, then corrects with the new measurement. 先预测目标下一刻位置，再用新观测修正。 |
| Hungarian algorithm | 匈牙利算法 | Finds the globally optimal one-to-one matching at minimum total cost. 求最小总代价的全局最优一对一匹配。 |
| IoU (Intersection over Union) | 交并比 | Overlap ÷ union of two boxes; 1 = identical, 0 = no overlap. 两框交集除以并集；1 表示完全重合，0 表示不交叠。 |
| Cost matrix | 代价矩阵 | Table of "how bad is this pairing" (here: 1 − IoU) fed to the Hungarian solver. "这种配对有多糟"的表（这里用 1 − IoU），喂给匈牙利算法求解。 |
| ByteTrack | ByteTrack | A tracking method whose key idea this project borrows: give low-confidence detections a second chance to rescue tracks. 本项目借鉴其核心思想的跟踪方法：给低置信度检测第二次机会来抢救轨迹。 |
| ID switch | ID 跳变 | One target's ID wrongly changes to another's — the main metric trackers try to minimize. 一个目标的 ID 被错换成另一个；跟踪器最想压低的指标。 |
| Occlusion | 遮挡 | Target hidden behind something for a few frames. 目标被遮挡若干帧。 |
| Fragmentation | 碎片化 | One real target split into many short tracks — a sign the tracker is struggling. 一个真实目标碎成多条短轨迹——跟踪器吃力的信号。 |
| Tentative / Confirmed / Lost / Deleted | 候选 / 确认 / 丢失 / 删除 | Lifecycle states of a track in this project. 本项目里轨迹的生命周期状态。 |
| Background subtraction | 背景减除 | Separating moving foreground from the static background (e.g. MOG2). 把运动前景从静态背景里分离出来，如 MOG2。 |
| MOG2 | MOG2 | Mixture-of-Gaussians background subtractor; needs no weights. 混合高斯背景减除器，无需权重。 |
| Morphology (open/close) | 形态学（开/闭运算） | Cleaning up the foreground mask: open removes speckles, close fills holes. 清理前景掩膜：开运算去噪点，闭运算填空洞。 |
| Contour | 轮廓 | Outline of a connected blob in the foreground mask. 前景掩膜里连通块的外轮廓。 |
| Fill ratio | 填充率 | Contour area ÷ bounding-box area; solid objects score higher than wispy noise. 轮廓面积除以包围盒面积；实心目标高于丝状噪声。 |
| Zone / ROI | 区域 / 感兴趣区域 | A polygon where entries, exits and dwell time are counted. 统计进出数和停留时长的多边形区域。 |
| Tripwire | 绊线 | A directed line segment; crossings are counted with direction. 有向线段；按方向统计穿越。 |
| Dwell time | 停留时长 | How long a track stays inside a zone. 轨迹在区域内停留的时间。 |
| Loitering | 徘徊 | Staying within a small area for too long. 在小范围内停留过久。 |
| Crowding | 聚集 | Too many tracks inside one zone at once. 单个区域内轨迹数超限。 |
| Wrong-way | 逆行 | Moving against the declared flow direction. 与设定流向相反的运动。 |
| False positive | 误检 | A detection with no real target (e.g. a flicker ghost). 没有真实目标的检测，如闪烁鬼影。 |
| ReID | 重识别 | Recognizing the same target by appearance after long occlusion (not implemented here). 长遮挡后靠外观认出同一目标（本项目未实现）。 |
