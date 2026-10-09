# 本地运行指南

在你自己的电脑上跑通 cv-tracker，只需 5 分钟。

## 1. 准备环境

需要 Python 3.8+。先克隆仓库并安装依赖：

```bash
git clone https://github.com/closing2005/closing2005-cv-tracker.git
cd closing2005-cv-tracker
pip install opencv-python pyyaml pytest
```

> `opencv-python` 带图形界面（能弹窗显示画面）；
> 服务器/无显示器环境改用 `opencv-python-headless`。

## 2. 跑测试（验证环境没问题）

```bash
python -m pytest tests/ -q
```

看到 `33 passed` 说明一切正常。不需要摄像头。

## 3. 一键演示（推荐先跑这个）

```bash
python examples/synthetic_demo.py
```

自动生成一段合成视频，验证两个核心能力：
遮挡 15 帧后 ID 保持不变、绊线计数正确。

## 4. 跟踪你自己的视频

```bash
# 有界面：实时看跟踪画面
python -m src.main --source your_video.mp4

# 无界面：结果存到 runs/run1/（轨迹 CSV + 统计摘要）
python -m src.main --source your_video.mp4 --no-show --out runs/run1

# 用摄像头（0 是默认摄像头）
python -m src.main --source 0
```

常用参数：`--max-frames 900`（只处理前 900 帧）、
`--show-zones`（显示区域线）。完整参数见 `python -m src.main --help`。

## 5. 用 YOLO 检测器（可选）

默认是 MOG2 运动检测（摄像头必须静止）。想检测静止目标
或摄像头会动，换 YOLO：

```bash
python scripts/download_model.py   # 下载 yolo11n.onnx（约 11MB）到 weights/
```

然后在 `config.yaml` 里把 `detector.type` 改为 `"yolo"`，
再按第 4 步运行。`classes: ["person"]` 可只跟踪人。

## 6. 打开实时 Web 面板（可选）

```bash
python -m src.main --source 0 --dashboard 8080 --no-show
```

浏览器打开 http://127.0.0.1:8080，看实时目标数、FPS 与事件流。

## 7. 开外观重识别（可选）

长遮挡（>1 秒）后靠外观找回身份，在 `config.yaml` 里：

```yaml
tracker:
  use_reid: true
```

## 常见问题

- **摄像头打不开**：Windows 上试试 `--source 1`；Linux 检查用户是否在 `video` 组。
- **YOLO 很慢**：CPU 上约 170ms/帧属正常，需要实时请用 GPU 版 OpenCV 或换回 MOG2。
- **MOG2 检测不到静止的人**：这是运动检测的固有局限，换 YOLO 检测器。
