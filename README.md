# YOLO Inference Tool

A powerful and flexible desktop-based GUI application built with **PyQt5** for performing real-time object detection using multiple YOLO models (YOLOv5 to YOLOv10). Designed to compare models side-by-side, customize detection settings, and visualize performance — all in a user-friendly interface.

---

## 🚀 Features

- 🔍 **Multi-Model Support**: Run inference with YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, and YOLOv10.
- 🖼️ **Input Flexibility**: Accepts image, video, and webcam streams.
- 📊 **Comparison Mode**: Visual comparison of multiple model outputs on the same input.
- ⚙️ **Custom Detection Options**: Toggle confidence thresholds, IoU, and class filters.
- 💾 **Save Results**: Export detection outputs as image/video files.
- 📈 **Performance Metrics**: View FPS, inference time, and detection counts per frame.

---

## 🛠️ Tech Stack

- **Frontend/UI**: PyQt5
- **Backend**: Python, OpenCV, Ultralytics YOLO, ONNX
- **YOLO Frameworks**: YOLOv5-v10 support (via official and custom inference handlers)

---

## 📁 Folder Structure

```bash
YOLO-Inference-Tool/
│
├── main.py                # Entry point of the application
├── ui/                    # PyQt5 UI design files
├── models/                # YOLO models (weights/configs)
├── inference/             # Model wrappers and processing code
├── assets/                # Icons, logos, static resources
├── utils/                 # Helper functions (preprocessing, drawing boxes, etc.)
└── outputs/               # Saved results (images/videos)
