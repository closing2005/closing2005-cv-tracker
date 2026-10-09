# 实现笔记：十二个模块的设计与取舍

记录每个模块为什么这样设计、踩过什么坑、当时怎么验证正确性。
按依赖顺序排列，也是理解本项目的推荐阅读顺序。

## 1. 匈牙利算法（`hungarian.py`）

**为什么手写**：SciPy 的 `linear_sum_assignment` 一行就能调，但手写一遍
才能真懂"行列减最小值 → 找独立零 → 覆盖线 → 调整"这个循环为什么收敛。
O(n³)，跟踪场景 n<50 时耗时可忽略。

**验证**：随机生成代价矩阵，与暴力枚举（n≤8）对比，结果必须一致。
见 `tests/test_hungarian.py`。

## 2. 卡尔曼滤波（`kalman.py`）

**设计**：状态 `[cx, cy, w, h, vx, vy]`，等速模型。过程噪声按目标尺寸
自适应——大目标允许更大的机动，小目标更保守。

**踩过的坑**：`np.float64` 会污染 OpenCV 矩阵类型，触发 `gemm` 断言失败。
所有赋给 Kalman 的矩阵一律 `astype(np.float32)`。

**验证**：给匀速直线运动加噪声，滤波轨迹应比原始观测更平滑；
遮挡时只 `predict()`，位置按速度外推。

## 3. 运动检测器（`detector.py`）

**设计**：MOG2 背景建模 → 去阴影 → 开闭运算去噪 → 找轮廓 →
按面积过滤。输出统一为 `[x1, y1, x2, y2, score]`，score 综合
目标尺寸与填充率（实心目标得分高于丝状噪声），让匹配阶段优先
处理高置信度目标。

**已知的局限**：静止目标会被 MOG2 逐渐"学进"背景而消失。
这是运动检测的固有局限，不是 bug——通用场景请换 YOLO 检测器。

## 4. 多目标跟踪器（`tracker.py`）——核心

**设计**：借鉴 ByteTrack 的两阶段关联。高分检测先匹配（可靠），
低分检测抢救被遮挡的目标（别急着删）。`Track` 状态机：
`Tentative → Confirmed → Lost → Deleted`。

第 10 步加入外观重识别后，关联变成三阶段：
IoU 匹配 → 低分抢救 → 外观匹配（位置已漂移时的最后手段）。

**验证**：两个目标交叉，ID 不应互换；目标消失 15 帧后出现，
ID 应保持。见 `examples/synthetic_demo.py`。

## 5. 区域与绊线（`zones.py`）

**设计**：`Zone` 做进入/离开/停留计数，`Tripwire` 做有向穿越
（用叉积判断穿越方向）。点在多边形内用 ray casting。

## 6. 行为事件（`events.py`）

**设计**：全部基于轨迹历史的启发式，无需模型。每个事件附带证据
（track_id、帧号、测量值），结果可审计：
徘徊=小范围停留超阈值；超速=像素速度超阈值；
逆行=速度方向与设定方向夹角>90°；聚集=局部密度超阈值。

## 7. 数据分析（`analytics.py`）

**设计**：轨迹 CSV 落盘、会话摘要（YAML）、跟踪质量自检。
核心指标是 ID fragmentation——一个真实目标被拆成几个 ID，
直接反映跟踪器的真实水平。

## 8. 主程序（`main.py`）

**设计**：CLI 入口，逐帧计时（定位性能瓶颈），`--dashboard`
启动 Web 面板。到这里 v1 的闭环完成。

## 9. YOLO 插件（`yolo_detector.py`）

**为什么是插件**：`Detector` 接口让跟踪器对检测器无感，
MOG2 和 YOLO 输出同一格式。YOLO11n 经 `cv2.dnn` 加载，
不依赖 ultralytics 包。

**取舍**：精度与语义换速度。实测见 [comparison.md](comparison.md)：
MOG2 5.7ms/帧 vs YOLO 175ms/帧（CPU）。摄像头静止且只关心
"有没有东西在动"时，MOG2 是更理性的选择。

原理细节见 [yolo_detector.md](yolo_detector.md)。

## 10. 外观重识别（`appearance.py`）

**为什么不用深度模型**：HSV 直方图在同镜头、同光照下已足够区分目标，
零权重、微秒级。接口可插拔，需要时可换 OSNet 等深度 embedding。

**设计**：embedding 做指数滑动平均（适应光照/姿态的缓慢漂移，
抵抗单帧噪声）；`ReIDGallery` 维护 track_id → 向量映射，
余弦相似度做匹配。

**验证**：目标消失 40 帧后换位置出现，ID 应恢复而非新建。
见 `tests/test_appearance.py`。

## 11. 多摄像头（`multicam.py`）

**设计**：每路独立 `MultiTracker` + 全局外观库。
`(cam, local_id) → global_id` 映射，新确认的轨迹用外观
匹配其他镜头的近期目标，`handover_window` 限制时间窗口
（目标不可能同时出现在两个镜头）。

这是务实子集：无需标定、无需视野重叠。

## 12. Web 面板（`dashboard.py`）

**设计**：纯标准库，`ThreadingHTTPServer` + SSE 推送事件流。
`DashboardState` 线程安全，跟踪线程写、HTTP 线程读。
刻意不用 Flask——少一个依赖就少一个安装失败的可能。
