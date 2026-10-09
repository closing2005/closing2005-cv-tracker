# MOG2 vs YOLO：对比实验

**问题**：两种检测器，哪个更适合 cv-tracker？

**方法**：在同一段 120 帧合成视频（640×480，两个匀速运动的白色方块，
其中一个有 10 帧遮挡）上分别运行，MOG2 跑 110 帧、YOLO 跑 20 帧，
重复 5 次取平均。CPU 实测，无 GPU。

## 结果

| 指标 | MOG2（运动检测） | YOLO11n（深度检测） |
|---|---|---|
| 平均耗时 | **5.7 ± 0.3 ms/帧** | 174.9 ± 17.2 ms/帧 |
| 平均检出框数 | 1.3 | **0.0** |
| 相对速度 | 快 ~30 倍 | — |

## 解读

1. **速度**：MOG2 快 30 倍。YOLO 的 172ms 在 CPU 上意味着 ~6fps，
   做实时跟踪需要 GPU 或更小的模型；MOG2 轻松 30fps+。

2. **检出数为 0 是预期内的**：YOLO11n 在 COCO 真实照片上训练，
   白色几何方块不在其训练域内——这是深度模型的 domain gap，
   不是 bug。在真实行人/车辆视频上，YOLO 会正常检出而 MOG2
   对静止目标无能为力。两者互补，而非替代。

   注：YOLO 在 CPU 上耗时波动较大（±17ms），属正常现象；
   MOG2 非常稳定（±0.3ms）。

3. **选型建议**：
   - 摄像头静止、只关心"有没有东西在动" → MOG2（快、零权重）
   - 摄像头会动 / 需要区分人车 / 目标会静止 → YOLO（准、有语义）
   - 跟踪器对两者无感：`config.yaml` 里改 `detector.type` 即可切换。

## 可复现

```bash
python scripts/download_model.py          # 下载 yolo11n.onnx 到 weights/
# MOG2
python -m src.main --source video.mp4 --no-show --out runs/mog2
# YOLO（先在 config.yaml 设 detector.type: "yolo"）
python -m src.main --source video.mp4 --no-show --out runs/yolo
```
