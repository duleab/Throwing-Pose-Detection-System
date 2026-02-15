# Models Directory

This directory contains the YOLOv8 model files.

## Required Models

1. **yolov8n-pose.pt** - Pose estimation model (~6.5 MB)
2. **yolov8n.pt** - Person detection model (~6.5 MB)

## Automatic Download

Models will be downloaded automatically on first run if not present.

## Manual Download

If you prefer to download manually:

```bash
# Download pose estimation model
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt

# Download person detection model
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

Or use Python:

```python
from ultralytics import YOLO

# This will download the models automatically
pose_model = YOLO('yolov8n-pose.pt')
detection_model = YOLO('yolov8n.pt')
```

## Git LFS (Recommended for Repository)

If you want to include models in the repository, use Git LFS:

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add models/*.pt
git commit -m "Add model files with Git LFS"
```

## Model Information

### YOLOv8n-pose
- **Purpose**: Pose estimation (17 keypoints)
- **Size**: ~6.5 MB
- **Speed**: Fast (suitable for real-time)
- **Accuracy**: Good for most use cases

### YOLOv8n
- **Purpose**: Person detection
- **Size**: ~6.5 MB
- **Speed**: Fast (suitable for real-time)
- **Accuracy**: Good for most use cases

## Alternative Models

For better accuracy (slower):
- `yolov8s-pose.pt` - Small model
- `yolov8m-pose.pt` - Medium model
- `yolov8l-pose.pt` - Large model
- `yolov8x-pose.pt` - Extra large model

Update `src/violence_detection.py` to use different models:

```python
model_loader = ModelLoader('models/yolov8m-pose.pt', 'models/yolov8m.pt')
```
