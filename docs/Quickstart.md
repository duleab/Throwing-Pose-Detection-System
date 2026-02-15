"""
QUICK START GUIDE - Modular Violence Detection System

This guide walks you through setup, basic usage, and common scenarios.
"""

import sys
print("Violence Detection System - Quick Start Guide\n")

SETUP_INSTRUCTIONS = """
1. INSTALLATION

# Install required packages
pip install ultralytics opencv-python scikit-learn scipy deep_sort_realtime

# Download models (automatic on first use)
# - YOLOv8 Nano Pose: yolov8n-pose.pt
# - YOLOv8 Nano Detection: yolov8n.pt

2. DIRECTORY STRUCTURE

project/
├── violence_detection_modular.py   # Core system
├── usage_example.py                 # Basic pipeline
├── testing_and_best_practices.py    # Tests & patterns
├── ARCHITECTURE.md                  # Design documentation
├── COMPARISON.md                    # vs original system
├── videos/
│   └── input_video.mp4
├── images/
│   └── sample_image.jpg
└── output/
    ├── output_video_detected.mp4
    └── annotated_frames/
"""

MINIMAL_EXAMPLE = """
3. MINIMAL EXAMPLE - Video Processing

from violence_detection_modular import ModelLoader, PersonTracker, FeatureExtractor
from violence_detection_modular import ViolenceClassifier, InteractionAnalyzer, FrameRenderer
from violence_detection_modular import VideoProcessor

processor = VideoProcessor(
    ModelLoader(),
    PersonTracker(),
    FeatureExtractor(),
    ViolenceClassifier(),
    InteractionAnalyzer(),
    FrameRenderer()
)

stats = processor.process_video('input.mp4', 'output.mp4')
print(f"Detected violence in {stats['violent_frames']} frames")
"""

IMAGE_EXAMPLE = """
4. MINIMAL EXAMPLE - Image Processing

from violence_detection_modular import ModelLoader, ImageProcessor, FeatureExtractor
from violence_detection_modular import ViolenceClassifier, FrameRenderer

processor = ImageProcessor(
    ModelLoader(),
    FeatureExtractor(),
    ViolenceClassifier(),
    FrameRenderer()
)

result = processor.process_image('photo.jpg')
print(f"Violence detected: {result['results']['has_violence']}")
"""

PIPELINE_EXAMPLE = """
5. FULL PIPELINE WITH CONFIGURATION

from usage_example import ViolenceDetectionPipeline

# Create pipeline with default models
pipeline = ViolenceDetectionPipeline()

# Process video with custom settings
video_stats = pipeline.process_video(
    video_path='surveillance.mp4',
    output_path='surveillance_detected.mp4',
    min_violence_frames=6
)

# Process single image
image_result = pipeline.process_image('frame.jpg')

# Batch process images
batch_results = pipeline.process_image_batch('image_directory/')
"""

TROUBLESHOOTING = """
6. COMMON ISSUES & SOLUTIONS

Issue: "CUDA out of memory"
Solution:
  - Use smaller model: yolov8n-pose.pt instead of yolov8l-pose.pt
  - Process at lower resolution: resize frames before processing
  - Use CPU: set CUDA_VISIBLE_DEVICES=-1

Issue: "No violence detected" (false negatives)
Solution:
  - Lower min_violence_frames from 6 to 4
  - Check if violence patterns match the feature thresholds
  - Verify video quality and lighting conditions
  - Review extracted features for the specific scenario

Issue: "Too many false positives"
Solution:
  - Increase min_violence_frames to 8 or more
  - Create CustomViolenceClassifier with higher thresholds
  - Filter by interaction_score to avoid solo high-movement detection

Issue: "Slow processing speed"
Solution:
  - GPU acceleration: install CUDA-enabled PyTorch
  - Frame skipping: process every 2nd or 3rd frame
  - Lower resolution: resize to 720p or 480p
  - Batch processing: multiple images at once

Issue: "Tracker loses person identity"
Solution:
  - Increase PersonTracker max_age from 15 to 30 frames
  - Reduce n_init from 2 to 1 for faster detection
  - Ensure consistent lighting and camera motion
"""

COMPONENT_REFERENCE = """
7. COMPONENT QUICK REFERENCE

ModelLoader
  - Loads YOLOv8 pose and detection models
  - Methods: detect_persons(), extract_pose()
  - Config: pose_model_path, detection_model_path, conf_threshold

PersonTracker
  - Maintains identity across frames using DeepSORT
  - Methods: update(detections, frame)
  - Config: max_age, n_init

FeatureExtractor
  - Calculates movement metrics from keypoints
  - Methods: extract_movement_features(), match_keypoints(), calculate_iou()
  - Features: upper_body_movement, wrist_acceleration, movement_variance, hip_movement

ViolenceClassifier
  - Scores movement patterns for violence probability
  - Methods: classify(features)
  - Threshold: score >= 0.8 indicates violence

InteractionAnalyzer
  - Detects multi-person violence patterns
  - Methods: detect_interaction_violence()
  - Config: proximity_threshold

FrameRenderer
  - Annotates frames with detection results
  - Methods: render_detections(), add_frame_info()
  - Features: colored boxes, status bar, frame counter

VideoProcessor
  - Orchestrates full video processing pipeline
  - Methods: process_video()
  - Output: annotated video + statistics + violent frames

ImageProcessor
  - Processes single images or batches
  - Methods: process_image(), batch_process_images()
  - Output: detection results + annotated image
"""

CUSTOMIZATION_PATTERNS = """
8. COMMON CUSTOMIZATION PATTERNS

Pattern 1: Custom Violence Thresholds
  class CustomClassifier(ViolenceClassifier):
      @staticmethod
      def classify(features):
          is_violent, score = ViolenceClassifier.classify(features)
          if features.upper_body_movement > 150:
              is_violent = True
          return is_violent, score

Pattern 2: Add New Movement Features
  class ExtendedFeatureExtractor(FeatureExtractor):
      @staticmethod
      def extract_movement_features(current_kpts, history, fps=30):
          features = FeatureExtractor.extract_movement_features(...)
          features.head_rotation = calculate_head_rotation(history)
          return features

Pattern 3: Custom Tracker
  class CustomTracker:
      def __init__(self):
          self.tracker = MyTrackingAlgorithm()
      def update(self, detections, frame):
          return self.tracker.process(detections, frame)

Pattern 4: Alternative Rendering
  class CustomRenderer(FrameRenderer):
      @staticmethod
      def render_detections(frame, tracks, violent_ids, alert):
          # Custom visualization logic
          return annotated_frame
"""

API_INTEGRATION = """
9. REST API INTEGRATION

Example with FastAPI:

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from usage_example import ViolenceDetectionPipeline
import tempfile
import os

app = FastAPI()
pipeline = ViolenceDetectionPipeline()

@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    
    try:
        result = pipeline.process_image(tmp_path)
        return JSONResponse({
            'has_violence': result['results']['has_violence'],
            'detections': result['results']['detections']
        })
    finally:
        os.unlink(tmp_path)

@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        tmp.write(await file.read())
        video_path = tmp.name
    
    output_path = video_path.replace('.mp4', '_detected.mp4')
    
    try:
        stats = pipeline.process_video(video_path, output_path)
        return JSONResponse({
            'total_frames': stats['total_frames'],
            'violent_frames': stats['violent_frames'],
            'violence_events': stats['violent_events']
        })
    finally:
        os.unlink(video_path)

# Run: uvicorn script:app --reload
"""

PERFORMANCE_TIPS = """
10. PERFORMANCE OPTIMIZATION TIPS

GPU Setup:
  - Verify CUDA: python -c "import torch; print(torch.cuda.is_available())"
  - Monitor usage: nvidia-smi -l 1 (updates every second)

Processing Speed:
  - Nano models: 15-25 fps on GPU
  - Small models: 8-15 fps on GPU
  - CPU only: 1-3 fps (not recommended)

Memory Optimization:
  - Limit tracked persons: Only track people in ROI
  - Clear history periodically: person_movement_histories.clear()
  - Process lower resolution: Resize to 640x480 before model

Parallel Processing:
  - Use multiprocessing for batch image processing
  - Queue frames for background processing
  - Distribute across GPUs with load balancing

Monitoring:
  - Log frame processing time
  - Track model inference time separately
  - Monitor memory growth over long sessions
"""

DEPLOYMENT = """
11. DEPLOYMENT CHECKLIST

Pre-deployment:
  ✓ Test on representative videos
  ✓ Validate accuracy on known violence events
  ✓ Verify false positive rate is acceptable
  ✓ Profile performance on target hardware
  ✓ Test error handling for corrupted files

Docker Deployment:
  FROM pytorch/pytorch:latest
  RUN pip install ultralytics opencv-python scikit-learn scipy deep_sort_realtime
  COPY . /app
  WORKDIR /app
  CMD ["python", "server.py"]

Cloud Deployment:
  - AWS: Use SageMaker for model serving, Lambda for API
  - GCP: Cloud Run for containerized API
  - Azure: App Service with GPU compute

Monitoring:
  - Log all detections with timestamps
  - Alert on violence events in real-time
  - Track system health metrics
  - Version control all model checkpoints
"""

print(SETUP_INSTRUCTIONS)
print("\n" + "="*60 + "\n")
print(MINIMAL_EXAMPLE)
print("\n" + "="*60 + "\n")
print(IMAGE_EXAMPLE)
print("\n" + "="*60 + "\n")
print(PIPELINE_EXAMPLE)
print("\n" + "="*60 + "\n")
print(TROUBLESHOOTING)
print("\n" + "="*60 + "\n")
print(COMPONENT_REFERENCE)
print("\n" + "="*60 + "\n")
print(CUSTOMIZATION_PATTERNS)
print("\n" + "="*60 + "\n")
print(API_INTEGRATION)
print("\n" + "="*60 + "\n")
print(PERFORMANCE_TIPS)
print("\n" + "="*60 + "\n")
print(DEPLOYMENT)

GETTING_HELP = """
12. GETTING HELP

Documentation:
  - ARCHITECTURE.md - Complete system design
  - COMPARISON.md - Original vs modular system
  - testing_and_best_practices.py - Test examples and patterns
  - violence_detection_modular.py - Source code with docstrings

Online Resources:
  - YOLOv8 Docs: https://docs.ultralytics.com
  - DeepSORT: https://github.com/mikel-brostrom/yolo_tracking
  - OpenCV: https://docs.opencv.org

Common Questions:

Q: How do I change detection sensitivity?
A: Adjust ViolenceClassifier thresholds or min_violence_frames parameter

Q: Can I use different tracking algorithms?
A: Yes, implement the PersonTracker interface with your algorithm

Q: How do I integrate with my surveillance system?
A: Use ImageProcessor.batch_process_images() or wrap in REST API

Q: What's the minimum hardware required?
A: 8GB RAM + quad-core CPU for CPU-only, GPU recommended for real-time

Q: Can I run multiple pipelines in parallel?
A: Yes, each pipeline is independent. Use multiprocessing for true parallelism
"""

print(GETTING_HELP)

print("\n✓ Setup Complete! You're ready to use the system.")
print("\nNext steps:")
print("1. Review ARCHITECTURE.md for system design")
print("2. Run testing_and_best_practices.py to verify installation")
print("3. Execute usage_example.py with your video files")