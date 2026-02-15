# Violence Detection System - Modular Architecture

## System Overview

This document describes a production-grade, modular violence detection system built with separation of concerns and scalability at its core. The system supports both video stream analysis and static image processing, with clear interfaces between components.

## Architecture Principles

**Modularity**: Each component has a single responsibility with well-defined inputs and outputs.

**Extensibility**: Components can be replaced or extended without affecting others.

**Testability**: Isolated components enable comprehensive unit and integration testing.

**Type Safety**: Dataclasses and type hints ensure data consistency across the pipeline.

**Scalability**: Components can be distributed across multiple processes or machines.

## Component Architecture

### 1. ModelLoader
**Responsibility**: Load and manage ML models for person detection and pose estimation.

**Key Methods**:
- `detect_persons(frame, conf_threshold)`: Detects person bounding boxes and confidence scores
- `extract_pose(frame)`: Extracts 17 keypoints per person (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)

**Design Notes**:
- Lazy loads models once during initialization
- Caches model instances for efficiency
- Handles model inference with verbose disabled for cleaner output
- Returns standardized detection format for downstream processing

### 2. BoundingBox & DataClasses
**Responsibility**: Provide type-safe data structures for bounding boxes and features.

**Classes**:
- `BoundingBox`: Immutable representation with convenience methods (center, dimensions)
- `ViolenceFeatures`: Dataclass containing extracted movement characteristics
- `DetectionResult`: Complete detection information with track ID, bbox, keypoints, and classification

**Benefits**:
- Type safety prevents runtime errors
- Clear data contracts between components
- Easy serialization to JSON/database formats

### 3. PersonTracker
**Responsibility**: Maintain identity consistency across frames using DeepSORT algorithm.

**Key Methods**:
- `update(detections, frame)`: Updates tracking state with new detections

**Features**:
- Assigns unique track IDs to individuals
- Handles person entrance/exit events
- Maintains temporal identity across occlusions
- Configurable max_age and initialization frames

### 4. FeatureExtractor
**Responsibility**: Calculate movement-based features from keypoint sequences.

**Key Methods**:
- `calculate_iou(box1, box2)`: Computes intersection-over-union for box matching
- `match_keypoints(bbox, pose_results)`: Associates poses with tracked persons
- `extract_movement_features(current_kpts, history, fps)`: Computes violence indicators

**Extracted Features**:
- **upper_body_movement**: Sum of euclidean distances for upper body joints (shoulders, elbows, wrists, neck)
- **wrist_acceleration**: Maximum acceleration magnitude of wrist movements
- **movement_variance**: Variance in total body movement across recent frames
- **hip_movement**: Sum of hip/lower body displacement

**Algorithm Details**:
- Maintains sliding window history of keypoint positions
- Computes frame-to-frame deltas normalized by FPS
- Captures both magnitude and acceleration of movement

### 5. ViolenceClassifier
**Responsibility**: Score frames based on movement features using rule-based logic.

**Scoring System**:
- Upper body movement > 40: Up to 0.5 points
- Wrist acceleration > 25: Up to 0.4 points
- Movement variance > 120: Up to 0.2 points
- Static lower body with violent upper body: 0.15 points bonus

**Classification Threshold**: Score ≥ 0.8 indicates violence

**Rationale**:
- Weighted contribution ensures no single feature dominates
- Thresholds calibrated through empirical testing
- Bonus score detects stationary punching patterns
- Allows fine-tuning confidence levels per deployment

### 6. InteractionAnalyzer
**Responsibility**: Detect violence patterns based on multi-person interactions.

**Detection Logic**:
- Identifies pairs of people within proximity threshold (150 pixels default)
- Flags as potential altercation if both show high upper body movement simultaneously
- Increments interaction violence scores for involved parties

**Use Cases**:
- Distinguishes fighting from solo dancing/exercise
- Captures multi-person altercations
- Reduces false positives from solo high-movement activities

### 7. FrameRenderer
**Responsibility**: Annotate frames with detection results and status indicators.

**Rendering Features**:
- Colored bounding boxes: Red for violent, cyan for normal
- Status bar at top: Red "VIOLENCE DETECTED" or yellow "Monitoring..."
- Frame counter for debugging
- Semi-transparent overlay for visual clarity

**Design Notes**:
- Separates visualization logic from detection logic
- Enables easy format changes (colors, fonts, positions)
- Supports custom annotation through inheritance

### 8. VideoProcessor
**Responsibility**: Main pipeline orchestrator for video stream processing.

**Workflow Per Frame**:
1. Detect persons and extract poses
2. Update tracking state
3. Extract movement features for each tracked person
4. Classify violence based on features
5. Detect multi-person interactions
6. Maintain violence confirmation counter (requires 6+ consecutive frames)
7. Render annotated frame
8. Store significant violent frames (up to 9 frames with 15-frame minimum separation)

**State Management**:
- `person_histories`: Deque of keypoint sequences per person (max 15 frames)
- `violence_confirmation`: Counter preventing single-frame false positives
- `violent_frames`: Stores annotated frames for visualization

**Output Statistics**:
- Total frames processed
- Count of frames with detected violence
- Timeline of violent events

### 9. ImageProcessor
**Responsibility**: Standalone processing for single images without temporal context.

**Key Methods**:
- `process_image(image_path)`: Analyzes single image, returns detection results
- `batch_process_images(image_paths)`: Processes multiple images efficiently

**Differences from VideoProcessor**:
- No tracking (single frame context)
- No history-based features (no previous keypoints)
- Outputs per-person detection with keypoint coordinates
- Returns annotated image for visualization

**Use Cases**:
- CCTV frame analysis
- Photo database screening
- Real-time image API endpoints

## Data Flow

### Video Processing Pipeline
```
Video Input
    ↓
Frame Extraction (30 fps)
    ↓
[ModelLoader] Person Detection + Pose Extraction
    ↓
[PersonTracker] ID Assignment & Tracking
    ↓
[FeatureExtractor] Movement Feature Calculation
    ↓
[ViolenceClassifier] Violence Scoring
    ↓
[InteractionAnalyzer] Multi-person Pattern Detection
    ↓
Violence Confirmation Counter (6+ frames threshold)
    ↓
[FrameRenderer] Annotation & Visualization
    ↓
Output Video + Statistics + Violent Frame Collection
```

### Image Processing Pipeline
```
Image Input
    ↓
[ModelLoader] Person Detection + Pose Extraction
    ↓
[FeatureExtractor] Movement Feature Calculation
    ↓
[ViolenceClassifier] Violence Scoring
    ↓
[FrameRenderer] Annotation
    ↓
Results (per-person detections + confidence scores)
```

## Configuration Parameters

### ModelLoader
- `pose_model_path`: Path to YOLO pose estimation model
- `detection_model_path`: Path to YOLO object detection model
- `conf_threshold`: Minimum confidence for person detection (default: 0.4)

### PersonTracker
- `max_age`: Frames to keep track after disappearance (default: 15)
- `n_init`: Frames required to confirm new track (default: 2)

### VideoProcessor
- `min_violence_frames`: Confirmation threshold (default: 6)
- `frame_skip`: Minimum frames between stored violent frames (default: 15)

### InteractionAnalyzer
- `proximity_threshold`: Maximum distance for interaction detection (default: 150 pixels)

## Extension Points

### Adding Custom Features
Extend `FeatureExtractor` to add new movement metrics:
```python
class CustomFeatureExtractor(FeatureExtractor):
    @staticmethod
    def extract_movement_features(current_kpts, history, fps=30):
        features = super().extract_movement_features(current_kpts, history, fps)
        features.your_custom_feature = calculate_new_metric(current_kpts, history)
        return features
```

### Custom Violence Rules
Create new `ViolenceClassifier` with domain-specific logic:
```python
class CustomViolenceClassifier(ViolenceClassifier):
    @staticmethod
    def classify(features):
        is_violent, base_score = super().classify(features)
        if features.specific_pattern_detected:
            is_violent = True
        return is_violent, min(base_score + 0.2, 1.0)
```

### Alternative Tracking
Replace `PersonTracker` with other algorithms (Kalman filters, centroid tracking):
```python
class CustomTracker:
    def __init__(self):
        self.tracker = YourTrackingAlgorithm()
    
    def update(self, detections, frame):
        return self.tracker.process(detections, frame)
```

## Performance Considerations

**Memory**:
- Keypoint history per person: ~2KB per frame × 15 frames × N people
- Frame storage: Limited to 9 violent frames
- Tracker overhead: ~100 bytes per active person

**Computation**:
- Detection/Pose: ~50-100ms per frame (GPU dependent)
- Tracking: ~5ms per update
- Feature extraction: ~2ms per person
- Classification: <1ms
- Rendering: ~10ms

**Optimization Strategies**:
- Use frame skipping for non-real-time scenarios
- Enable GPU acceleration for YOLO models
- Batch process images for higher throughput
- Profile with cProfile for bottleneck identification

## Error Handling

**Robustness Features**:
- Graceful handling of missing keypoints
- Validation of bbox-pose matching (IOU > 0.3)
- Deque length constraints to prevent memory bloat
- Video file existence checks before processing

**Logging Integration**:
Add logging by extending components:
```python
import logging

logger = logging.getLogger(__name__)

class LoggingVideoProcessor(VideoProcessor):
    def process_video(self, video_path, output_path, **kwargs):
        logger.info(f"Starting video processing: {video_path}")
        try:
            stats = super().process_video(video_path, output_path, **kwargs)
            logger.info(f"Completed: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            raise
```

## Testing Strategy

**Unit Tests**:
- Test feature extraction with known keypoint sequences
- Validate violence classification with edge cases
- Test IoU calculations with overlapping boxes

**Integration Tests**:
- Process short video clips with known violence events
- Verify tracker ID consistency across frames
- Validate output format consistency

**Performance Tests**:
- Benchmark processing speed (frames/second)
- Memory profiling with long video sequences
- Verify stable memory usage over time

## Deployment Considerations

**Containerization**:
Package with Docker for consistent environment across machines.

**API Integration**:
Wrap VideoProcessor/ImageProcessor in FastAPI/Flask for REST endpoints.

**Database Storage**:
Serialize `DetectionResult` objects to JSON for database persistence.

**Real-time Monitoring**:
Queue frames for asynchronous processing with Redis/RabbitMQ.

**Model Updates**:
Implement model versioning and gradual rollout of new detection models.