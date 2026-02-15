# 🔍 Detailed Code Analysis - Violence Detection System

## Complete Logic Explanation & Feature Breakdown

**Analysis Date:** February 15, 2026  
**Purpose:** In-depth explanation of code logic, algorithms, and features

---

## 📚 Table of Contents

1. [System Overview](#system-overview)
2. [Core Components Deep Dive](#core-components-deep-dive)
3. [Data Flow & Algorithms](#data-flow--algorithms)
4. [Feature Engineering](#feature-engineering)
5. [Violence Detection Logic](#violence-detection-logic)
6. [Code Examples with Explanations](#code-examples-with-explanations)

---

## System Overview

### High-Level Architecture

The system processes video/images through a **pipeline of 9 specialized components**:

```
Input (Video/Image)
    ↓
ModelLoader (YOLOv8 Detection & Pose)
    ↓
PersonTracker (DeepSORT Tracking)
    ↓
FeatureExtractor (Movement Analysis)
    ↓
ViolenceClassifier (Rule-based Scoring)
    ↓
InteractionAnalyzer (Multi-person Detection)
    ↓
FrameRenderer (Visualization)
    ↓
Output (Annotated Video/Image + Statistics)
```

---

## Core Components Deep Dive

### 1. **BoundingBox Class** - Geometric Utilities

```python
@dataclass
class BoundingBox:
    x1: float  # Top-left X coordinate
    y1: float  # Top-left Y coordinate
    x2: float  # Bottom-right X coordinate
    y2: float  # Bottom-right Y coordinate
```

**Purpose:** Type-safe representation of detection boxes

**Key Methods:**

#### `to_ltrb()` - Convert to Left-Top-Right-Bottom format
```python
def to_ltrb(self) -> Tuple[float, float, float, float]:
    return self.x1, self.y1, self.x2, self.y2
```
**Logic:** Returns coordinates as tuple for OpenCV drawing functions

#### `to_center()` - Calculate box center point
```python
def to_center(self) -> Tuple[float, float]:
    return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2
```
**Logic:** 
- Center X = (left + right) / 2
- Center Y = (top + bottom) / 2
- Used for proximity calculations between people

#### `width()` and `height()` - Get dimensions
```python
def width(self) -> float:
    return self.x2 - self.x1

def height(self) -> float:
    return self.y2 - self.y1
```
**Logic:** Simple subtraction to get box dimensions

---

### 2. **ViolenceFeatures Class** - Movement Metrics

```python
@dataclass
class ViolenceFeatures:
    upper_body_movement: float = 0.0    # Shoulder/arm movement
    wrist_acceleration: float = 0.0     # Punching/striking speed
    movement_variance: float = 0.0      # Movement consistency
    hip_movement: float = 0.0           # Lower body stability
```

**Purpose:** Structured storage of violence indicators

**Why These Features?**

1. **upper_body_movement**: Violent actions involve rapid arm/shoulder motion
2. **wrist_acceleration**: Punches have high wrist acceleration
3. **movement_variance**: Erratic movement suggests struggle/fight
4. **hip_movement**: Stable hips + moving arms = punching stance

**Conversion to Dictionary:**
```python
def to_dict(self) -> Dict:
    return {
        'upper_body_movement': self.upper_body_movement,
        'wrist_acceleration': self.wrist_acceleration,
        'movement_variance': self.movement_variance,
        'hip_movement': self.hip_movement
    }
```
**Logic:** Enables JSON serialization for API responses

---

### 3. **ModelLoader Class** - AI Model Management

```python
class ModelLoader:
    def __init__(self, pose_model_path: str = 'yolov8n-pose.pt', 
                 detection_model_path: str = 'yolov8n.pt'):
        self.pose_model = YOLO(pose_model_path)
        self.detection_model = YOLO(detection_model_path)
```

**Purpose:** Load and manage YOLOv8 models

**Two Models Used:**

1. **Detection Model** (`yolov8n.pt`): Finds people in frames
2. **Pose Model** (`yolov8n-pose.pt`): Extracts 17 body keypoints

#### Person Detection Logic

```python
def detect_persons(self, frame: np.ndarray, conf_threshold: float = 0.4) -> List:
    # Run YOLO detection
    detections = self.detection_model(frame, verbose=False)
    person_detections = []
    
    # Filter for person class (class 0 in COCO dataset)
    for det in detections[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = det
        if int(cls) == 0 and conf > conf_threshold:  # Class 0 = person
            # Convert to [x, y, width, height] format for tracker
            person_detections.append([[x1, y1, x2 - x1, y2 - y1], conf, "person"])
    
    return person_detections
```

**Step-by-Step Logic:**

1. **Run YOLO:** Process frame through detection model
2. **Extract boxes:** Get bounding box coordinates + confidence + class
3. **Filter persons:** Keep only detections where class=0 (person) and confidence > 0.4
4. **Format conversion:** Convert from [x1,y1,x2,y2] to [x,y,w,h] for DeepSORT
5. **Return list:** Each detection = [[bbox], confidence, "person"]

**Why confidence threshold 0.4?**
- Lower = more detections but more false positives
- Higher = fewer false positives but might miss people
- 0.4 is balanced for surveillance scenarios

#### Pose Extraction Logic

```python
def extract_pose(self, frame: np.ndarray):
    return self.pose_model(frame, verbose=False)
```

**What it returns:**
- 17 keypoints per person: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
- Each keypoint has (x, y) coordinates
- Used for movement analysis

---

### 4. **PersonTracker Class** - Identity Tracking

```python
class PersonTracker:
    def __init__(self, max_age: int = 15, n_init: int = 2):
        self.tracker = DeepSort(max_age=max_age, n_init=n_init)
```

**Purpose:** Maintain consistent IDs for people across frames

**Parameters Explained:**

- **max_age=15**: Keep tracking for 15 frames after person disappears
  - Handles brief occlusions (person behind object)
  - Prevents ID switching when person reappears
  
- **n_init=2**: Require 2 consecutive detections to confirm new person
  - Reduces false positives from noise
  - Ensures person is actually present

#### Tracking Update Logic

```python
def update(self, detections: List, frame: np.ndarray) -> List:
    return self.tracker.update_tracks(detections, frame=frame)
```

**What DeepSORT does:**

1. **Feature Extraction:** Extract visual features from each person's bounding box
2. **Motion Prediction:** Predict where each tracked person will be in next frame
3. **Data Association:** Match new detections to existing tracks using:
   - Visual similarity (appearance features)
   - Motion consistency (Kalman filter)
   - IoU (box overlap)
4. **ID Assignment:** Assign consistent track_id to each person
5. **Track Management:** Create new tracks, delete old tracks

**Example:**
```
Frame 1: Person A detected → Assign ID=1
Frame 2: Person A moves → Still ID=1 (tracked)
Frame 3: Person A + Person B → ID=1 and ID=2
Frame 4: Person A hidden → ID=1 kept (max_age)
Frame 5: Person A reappears → Still ID=1 (not new ID)
```

---

### 5. **FeatureExtractor Class** - Movement Analysis

This is the **core intelligence** of the system. It analyzes body movements to detect violence.

#### IoU Calculation - Box Matching

```python
@staticmethod
def calculate_iou(box1: BoundingBox, box2: Tuple) -> float:
    x1, y1, x2, y2 = box1.to_ltrb()
    gx1, gy1, gx2, gy2 = box2
    
    # Calculate intersection rectangle
    ix1 = max(x1, gx1)  # Left edge of intersection
    iy1 = max(y1, gy1)  # Top edge
    ix2 = min(x2, gx2)  # Right edge
    iy2 = min(y2, gy2)  # Bottom edge
    
    # Calculate intersection area
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter_area = iw * ih
    
    # Calculate union area
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (gx2 - gx1) * (gy2 - gy1)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0
```

**Purpose:** Measure overlap between two bounding boxes

**Visual Explanation:**
```
Box 1:  ┌─────────┐
        │         │
        │    ┌────┼────┐  Box 2
        │    │████│    │  (████ = intersection)
        └────┼────┘    │
             │         │
             └─────────┘

IoU = Intersection Area / Union Area
```

**Why IoU?**
- Match person detection boxes with pose estimation boxes
- IoU > 0.3 means boxes likely represent same person
- Ensures we analyze the correct person's pose

#### Keypoint Matching Logic

```python
@staticmethod
def match_keypoints(bbox: BoundingBox, pose_results) -> Optional[np.ndarray]:
    if not pose_results or len(pose_results) == 0 or not pose_results[0].boxes:
        return None

    best_keypoints = None
    max_iou = 0

    # Try each detected pose
    for i in range(len(pose_results[0].boxes)):
        pose_box = pose_results[0].boxes.xyxy[i].cpu().numpy()
        keypoints = pose_results[0].keypoints.xy[i].cpu().numpy()
        iou = FeatureExtractor.calculate_iou(bbox, pose_box)
        
        # Keep pose with highest overlap
        if iou > max_iou:
            max_iou = iou
            best_keypoints = keypoints

    return best_keypoints if max_iou > 0.3 else None
```

**Step-by-Step Logic:**

1. **Check validity:** Ensure pose results exist
2. **Iterate poses:** Loop through all detected poses in frame
3. **Calculate IoU:** Compare each pose box with tracked person box
4. **Find best match:** Keep pose with highest IoU
5. **Threshold check:** Only return if IoU > 0.3 (30% overlap)
6. **Return keypoints:** 17 (x,y) coordinates for body joints

**Why this matters:**
- Multiple people in frame → multiple poses detected
- Need to match correct pose to correct tracked person
- IoU ensures we analyze the right person's movements

#### Movement Feature Extraction - THE CORE ALGORITHM

```python
@staticmethod
def extract_movement_features(current_kpts: np.ndarray, 
                              history: deque, 
                              fps: float = 30) -> ViolenceFeatures:
    features = ViolenceFeatures()

    if len(history) < 2:
        return features  # Need at least 2 frames for movement

    prev_kpts = history[-1]  # Previous frame keypoints
    if len(current_kpts) != len(prev_kpts):
        return features  # Keypoint count mismatch
```

**Initial Setup:**
- Requires at least 2 frames to calculate movement
- Compares current frame to previous frame
- Returns empty features if insufficient data

##### Feature 1: Upper Body Movement

```python
    # Keypoint indices for upper body
    upper_body_indices = [0, 5, 6, 7, 8, 9, 10]
    # 0=nose, 5=left_shoulder, 6=right_shoulder, 
    # 7=left_elbow, 8=right_elbow, 9=left_wrist, 10=right_wrist
    
    upper_movement = sum(
        sqrt((current_kpts[i][0] - prev_kpts[i][0]) ** 2 + 
             (current_kpts[i][1] - prev_kpts[i][1]) ** 2)
        for i in upper_body_indices if i < len(current_kpts)
    )
    features.upper_body_movement = upper_movement
```

**Mathematical Logic:**

1. **Euclidean Distance:** For each keypoint, calculate:
   ```
   distance = √[(x_current - x_previous)² + (y_current - y_previous)²]
   ```

2. **Sum all distances:** Total movement across all upper body joints

3. **Interpretation:**
   - Low value (< 40): Normal movement
   - Medium value (40-120): Active movement
   - High value (> 120): Violent movement

**Example:**
```
Frame 1: Shoulder at (100, 200)
Frame 2: Shoulder at (150, 220)
Distance = √[(150-100)² + (220-200)²] = √[2500 + 400] = 53.85 pixels

If all 7 upper body points move ~50 pixels → Total ≈ 350 (VIOLENT!)
```

##### Feature 2: Wrist Acceleration

```python
    if len(history) >= 3:  # Need 3 frames for acceleration
        wrist_indices = [9, 10]  # Left and right wrists
        max_accel = 0
        
        for wrist_idx in wrist_indices:
            if wrist_idx < len(current_kpts):
                curr = np.array(current_kpts[wrist_idx])
                prev = np.array(history[-1][wrist_idx])
                prev_prev = np.array(history[-2][wrist_idx])
                
                # Acceleration = change in velocity
                accel = np.linalg.norm((curr - prev) - (prev - prev_prev))
                max_accel = max(max_accel, accel)
        
        features.wrist_acceleration = max_accel
```

**Physics Behind This:**

1. **Velocity:** Change in position over time
   ```
   velocity_t1 = position_t1 - position_t0
   velocity_t2 = position_t2 - position_t1
   ```

2. **Acceleration:** Change in velocity
   ```
   acceleration = velocity_t2 - velocity_t1
                = (pos_t2 - pos_t1) - (pos_t1 - pos_t0)
   ```

3. **Why wrists?**
   - Punching involves rapid wrist acceleration
   - Normal gestures have lower acceleration
   - Distinguishes punching from waving

**Example:**
```
Frame 1: Wrist at (100, 100)
Frame 2: Wrist at (120, 100)  → Velocity = 20 pixels/frame
Frame 3: Wrist at (180, 100)  → Velocity = 60 pixels/frame

Acceleration = 60 - 20 = 40 pixels/frame² (HIGH = possible punch!)
```

##### Feature 3: Movement Variance

```python
    if len(history) >= 5:  # Need 5 frames for variance
        movements = []
        
        for i in range(len(history) - 1):
            curr_h = history[i + 1]
            prev_h = history[i]
            if len(curr_h) == len(prev_h):
                # Total movement between consecutive frames
                move = sum(
                    sqrt((c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2)
                    for c, p in zip(curr_h, prev_h)
                )
                movements.append(move)
        
        features.movement_variance = float(np.var(movements)) if movements else 0
```

**Statistical Logic:**

1. **Calculate movement per frame pair:** Total pixel displacement
2. **Build movement sequence:** [move1, move2, move3, move4]
3. **Calculate variance:** How much movement fluctuates

**Variance Formula:**
```
variance = Σ(movement_i - mean_movement)² / n
```

**Why variance matters:**

- **Low variance:** Smooth, consistent movement (walking, dancing)
- **High variance:** Erratic, unpredictable movement (fighting, struggling)

**Example:**
```
Walking: [50, 52, 51, 50, 51] → Variance ≈ 0.8 (LOW)
Fighting: [20, 150, 30, 180, 25] → Variance ≈ 5000 (HIGH!)
```

##### Feature 4: Hip Movement

```python
    hip_indices = [11, 12]  # Left and right hips
    hip_movement = sum(
        sqrt((current_kpts[i][0] - prev_kpts[i][0]) ** 2 + 
             (current_kpts[i][1] - prev_kpts[i][1]) ** 2)
        for i in hip_indices if i < len(current_kpts)
    )
    features.hip_movement = hip_movement
```

**Logic:**
- Same as upper body movement, but for hips only
- Used to detect **stationary punching**

**Why important:**

```
Scenario 1: Dancing
- Upper body moving: HIGH
- Hips moving: HIGH
- Result: NOT violence (whole body moving)

Scenario 2: Punching
- Upper body moving: HIGH
- Hips stable: LOW
- Result: VIOLENCE (punching stance)
```

---

### 6. **ViolenceClassifier Class** - Decision Logic

```python
class ViolenceClassifier:
    @staticmethod
    def classify(features: ViolenceFeatures) -> Tuple[bool, float]:
        score = 0.0
```

**Purpose:** Convert movement features into violence probability

#### Scoring Algorithm - Weighted System

```python
        # Rule 1: Upper body movement
        if features.upper_body_movement > 40:
            score += 0.5 * min(features.upper_body_movement / 120, 1.0)
```

**Logic Breakdown:**

1. **Threshold check:** Only score if movement > 40 pixels
2. **Normalization:** Divide by 120 to get 0-1 range
3. **Capping:** `min(..., 1.0)` prevents scores > 1.0
4. **Weighting:** Multiply by 0.5 (50% of total score)

**Example:**
```
Movement = 60:  score += 0.5 * (60/120) = 0.25
Movement = 120: score += 0.5 * (120/120) = 0.50
Movement = 240: score += 0.5 * min(2.0, 1.0) = 0.50 (capped)
```

```python
        # Rule 2: Wrist acceleration
        if features.wrist_acceleration > 25:
            score += 0.4 * min(features.wrist_acceleration / 60, 1.0)
```

**Logic:**
- Threshold: 25 pixels/frame²
- Max contribution: 0.4 (40% of total)
- Normalized by 60 (typical punch acceleration)

```python
        # Rule 3: Movement variance
        if features.movement_variance > 120:
            score += 0.2 * min(features.movement_variance / 350, 1.0)
```

**Logic:**
- Threshold: 120 (erratic movement)
- Max contribution: 0.2 (20% of total)
- Normalized by 350 (high variance threshold)

```python
        # Rule 4: Stationary upper body violence (punching stance)
        if features.hip_movement < 15 and features.upper_body_movement > 70:
            score += 0.15
```

**Logic:**
- Detects: Stable lower body + violent upper body
- Bonus: +0.15 points
- Identifies: Punching, striking while standing still

#### Final Classification

```python
        is_violent = score >= 0.8
        return is_violent, score
```

**Decision Threshold:**
- Score ≥ 0.8 → Violence detected
- Score < 0.8 → Normal behavior

**Why 0.8?**
- High threshold reduces false positives
- Requires multiple indicators to trigger
- Can be tuned based on use case

**Scoring Examples:**

```python
# Example 1: Normal walking
upper_body=30, wrist=10, variance=50, hip=25
→ score = 0.0 (no thresholds met) → NOT VIOLENT

# Example 2: Dancing
upper_body=80, wrist=20, variance=100, hip=60
→ score = 0.5*(80/120) = 0.33 → NOT VIOLENT (< 0.8)

# Example 3: Punching
upper_body=120, wrist=50, variance=200, hip=10
→ score = 0.5 + 0.4 + 0.2*(200/350) + 0.15 = 1.16 (capped at 1.0)
→ VIOLENT (> 0.8)
```

---

### 7. **InteractionAnalyzer Class** - Multi-Person Detection

```python
class InteractionAnalyzer:
    @staticmethod
    def detect_interaction_violence(track_data: Dict, 
                                    proximity_threshold: float = 150) -> Dict[int, float]:
        violence_scores = {tid: 0 for tid in track_data.keys()}
        track_ids = list(track_data.keys())
```

**Purpose:** Detect violence between multiple people

**Logic:** Analyze all pairs of people in frame

#### Pairwise Proximity Analysis

```python
        for i, tid1 in enumerate(track_ids):
            for j, tid2 in enumerate(track_ids):
                if i >= j:
                    continue  # Avoid duplicate pairs and self-comparison
```

**Iteration Logic:**
```
People: [1, 2, 3]
Pairs checked: (1,2), (1,3), (2,3)
Pairs skipped: (2,1), (3,1), (3,2), (1,1), (2,2), (3,3)
```

#### Distance Calculation

```python
                bbox1 = track_data[tid1]['bbox']
                bbox2 = track_data[tid2]['bbox']
                center1 = bbox1.to_center()
                center2 = bbox2.to_center()

                distance = sqrt((center1[0] - center2[0]) ** 2 + 
                              (center1[1] - center2[1]) ** 2)
```

**Euclidean Distance Between Centers:**
```
Person 1 center: (100, 200)
Person 2 center: (180, 250)
Distance = √[(180-100)² + (250-200)²] = √[6400 + 2500] = 94.3 pixels
```

#### Interaction Detection

```python
                if distance < proximity_threshold:  # Default: 150 pixels
                    feat1 = track_data[tid1]['features']
                    feat2 = track_data[tid2]['features']

                    # Both people showing high movement
                    if feat1.upper_body_movement > 80 and feat2.upper_body_movement > 80:
                        violence_scores[tid1] += 1
                        violence_scores[tid2] += 1
```

**Logic:**

1. **Proximity check:** Are people close enough to interact? (< 150 pixels)
2. **Movement check:** Are both people moving violently? (> 80 pixels)
3. **Score increment:** Add +1 to both people's interaction scores

**Why this matters:**

```
Scenario 1: Solo Exercise
- Person A: High movement, no one nearby
- Interaction score: 0
- Result: NOT flagged as violence

Scenario 2: Two People Fighting
- Person A: High movement
- Person B: High movement
- Distance: 80 pixels (close)
- Interaction score: 1 for both
- Result: FLAGGED as violence
```

**Return Value:**
```python
{
    1: 1,  # Person ID 1 has 1 violent interaction
    2: 1,  # Person ID 2 has 1 violent interaction
    3: 0   # Person ID 3 has no violent interactions
}
```

---

### 8. **FrameRenderer Class** - Visualization

#### Main Rendering Function

```python
@staticmethod
def render_detections(frame: np.ndarray, 
                     tracks, 
                     violent_ids: set,
                     global_violence_alert: bool) -> np.ndarray:
    annotated = frame.copy()  # Don't modify original
    height, width = frame.shape[:2]
```

**Purpose:** Draw bounding boxes and labels on frame

#### Drawing Bounding Boxes

```python
    for track in tracks:
        if not track.is_confirmed():
            continue  # Skip unconfirmed tracks

        x1, y1, x2, y2 = map(int, track.to_ltrb())
        track_id = track.track_id
        is_violent = track_id in violent_ids

        # Color coding
        color = (0, 0, 255) if is_violent else (255, 255, 0)  # Red or Cyan
        label = "Violence" if is_violent else ""

        # Draw rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
```

**Color Logic:**
- **Red (0, 0, 255):** Violent person
- **Cyan (255, 255, 0):** Normal person
- **BGR format:** OpenCV uses Blue-Green-Red

#### Drawing Labels

```python
        if label:
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            label_y = max(y1 - 8, 15)  # Position above box, min 15 pixels from top
            
            # Background rectangle for text
            cv2.rectangle(annotated, (x1, label_y - lh - 4), 
                        (x1 + lw + 4, label_y + 2), color, -1)
            
            # White text
            cv2.putText(annotated, label, (x1 + 2, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
```

**Visual Layout:**
```
┌─────────────────┐
│ Violence        │ ← Label with background
├─────────────────┤
│                 │
│   Person Box    │ ← Red rectangle
│                 │
└─────────────────┘
```

#### Status Bar Rendering

```python
@staticmethod
def _render_status_bar(frame: np.ndarray, alert: bool, width: int):
    overlay = frame.copy()
    bar_height = 28

    if alert:
        cv2.rectangle(overlay, (0, 0), (width, bar_height), (0, 0, 255), -1)
        text = "VIOLENCE DETECTED"
    else:
        cv2.rectangle(overlay, (0, 0), (width, bar_height), (255, 255, 0), -1)
        text = "Monitoring..."

    cv2.putText(overlay, text, (15, int(bar_height * 0.5)),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
               (255, 255, 255) if alert else (0, 0, 0), 2)
    
    # Blend overlay with original (40% opacity)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
```

**Blending Logic:**
```
final_pixel = 0.4 * overlay_pixel + 0.6 * original_pixel

Example:
overlay = (255, 0, 0) [Red]
original = (100, 100, 100) [Gray]
result = 0.4*(255,0,0) + 0.6*(100,100,100) = (162, 60, 60) [Semi-transparent red]
```

---

### 9. **VideoProcessor Class** - Pipeline Orchestration

#### Initialization

```python
def __init__(self, model_loader: ModelLoader, 
             person_tracker: PersonTracker,
             feature_extractor: FeatureExtractor,
             violence_classifier: ViolenceClassifier,
             interaction_analyzer: InteractionAnalyzer,
             frame_renderer: FrameRenderer):
    self.model_loader = model_loader
    self.tracker = person_tracker
    self.feature_extractor = feature_extractor
    self.classifier = violence_classifier
    self.analyzer = interaction_analyzer
    self.renderer = frame_renderer
    
    # State management
    self.person_histories = {}  # {track_id: deque of keypoints}
    self.violence_confirmation = {}  # {track_id: consecutive violent frames}
    self.violent_frames = []  # List of (frame_idx, frame_image)
```

**Dependency Injection Pattern:**
- All components passed as parameters
- Easy to swap implementations
- Testable in isolation

#### Main Processing Loop

```python
def process_video(self, video_path: str, output_path: str, 
                 min_violence_frames: int = 6,
                 frame_skip: int = 15) -> Dict:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                        fps, (width, height))
```

**Video Setup:**
1. Open input video
2. Get properties (FPS, dimensions)
3. Create output video writer with same properties

#### Frame-by-Frame Processing

```python
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        stats['total_frames'] += 1

        # Step 1: Detect persons
        person_dets = self.model_loader.detect_persons(frame)
        
        # Step 2: Extract poses
        pose_results = self.model_loader.extract_pose(frame)

        # Step 3: Update tracking
        tracks = self.tracker.update(person_dets, frame)
```

**Pipeline Steps:**
1. Read frame from video
2. Detect all people in frame
3. Extract pose keypoints for all people
4. Update tracking (assign IDs)

#### Per-Person Analysis

```python
        violent_ids = set()
        track_data = {}

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox_ltrb = np.array(track.to_ltrb())
            bbox = BoundingBox(*bbox_ltrb)

            # Match pose to person
            keypoints = self.feature_extractor.match_keypoints(bbox, pose_results)

            if keypoints is not None:
                # Initialize history for new person
                if track_id not in self.person_histories:
                    self.person_histories[track_id] = deque(maxlen=15)
                    self.violence_confirmation[track_id] = 0
```

**State Management:**
- Each person gets a history buffer (15 frames max)
- Each person gets a violence confirmation counter
- Deque automatically removes old frames when full

#### Feature Extraction & Classification

```python
                # Extract movement features
                features = self.feature_extractor.extract_movement_features(
                    keypoints, self.person_histories[track_id], fps
                )
                self.person_histories[track_id].append(keypoints)

                # Classify violence
                is_violent, confidence = self.classifier.classify(features)
                track_data[track_id] = {'bbox': bbox, 'features': features}

                # Update confirmation counter
                if is_violent:
                    self.violence_confirmation[track_id] += 1
                else:
                    self.violence_confirmation[track_id] = max(0, 
                        self.violence_confirmation[track_id] - 1)
```

**Confirmation Logic:**

```
Frame 1: Violent → counter = 1
Frame 2: Violent → counter = 2
Frame 3: Violent → counter = 3
...
Frame 6: Violent → counter = 6 → ALERT!
Frame 7: Not violent → counter = 5 (decrements)
```

**Why confirmation counter?**
- Prevents single-frame false positives
- Requires sustained violence (6+ frames)
- Smooths out detection noise

#### Multi-Person Interaction Analysis

```python
                # Check if confirmed violent
                if self.violence_confirmation[track_id] >= min_violence_frames:
                    violent_ids.add(track_id)

        # Analyze interactions between people
        if len(track_data) > 1:
            interaction_scores = self.analyzer.detect_interaction_violence(track_data)
            for tid, int_score in interaction_scores.items():
                if int_score > 0:
                    violent_ids.add(tid)
```

**Logic:**
1. Mark people with 6+ consecutive violent frames
2. If 2+ people in frame, check interactions
3. Add people involved in violent interactions to violent_ids

#### Frame Storage & Output

```python
        global_violence_alert = len(violent_ids) > 0

        # Render annotations
        annotated = self.renderer.render_detections(frame, tracks, 
                                                   violent_ids, global_violence_alert)
        self.renderer.add_frame_info(annotated, frame_idx)

        # Store violent frames (max 9, with spacing)
        if global_violence_alert and len(self.violent_frames) < 9 and \
           (frame_idx - last_violent_frame >= frame_skip or last_violent_frame == -1):
            self.violent_frames.append((frame_idx, annotated.copy()))
            last_violent_frame = frame_idx
            stats['violent_frames'] += 1

        if global_violence_alert:
            stats['violent_events'].append(frame_idx)

        out.write(annotated)
```

**Storage Logic:**
- Store max 9 violent frames
- Minimum 15 frames between stored frames (avoid duplicates)
- Store frame index + annotated image
- Track all violent frame indices in stats

---

### 10. **ImageProcessor Class** - Single Image Analysis

```python
def process_image(self, image_path: str) -> Dict:
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Cannot read image: {image_path}")

    person_dets = self.model_loader.detect_persons(frame)
    pose_results = self.model_loader.extract_pose(frame)

    results = {
        'image_path': image_path,
        'detections': [],
        'has_violence': False
    }
```

**Difference from Video:**
- No tracking (single frame)
- No history (no previous frames)
- Simpler analysis

#### Per-Person Detection

```python
    for det in person_dets:
        x1, y1, w, h = det[0]
        x2, y2 = x1 + w, y1 + h
        bbox = BoundingBox(x1, y1, x2, y2)

        keypoints = self.feature_extractor.match_keypoints(bbox, pose_results)

        if keypoints is not None:
            # No history available for single image
            features = self.feature_extractor.extract_movement_features(
                keypoints, deque(maxlen=1)  # Empty history
            )
            is_violent, confidence = self.classifier.classify(features)

            detection = {
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'keypoints': keypoints.tolist(),
                'is_violent': is_violent,
                'confidence': float(confidence),
                'features': features.to_dict()
            }
            results['detections'].append(detection)

            if is_violent:
                results['has_violence'] = True
```

**Limitations:**
- No movement history → features will be mostly 0
- Less accurate than video analysis
- Useful for screening static images

---

## Data Flow Summary

### Complete Pipeline Flow

```
1. VIDEO INPUT
   ↓
2. FRAME EXTRACTION (OpenCV)
   ↓
3. PERSON DETECTION (YOLOv8)
   → Bounding boxes for all people
   ↓
4. POSE ESTIMATION (YOLOv8-Pose)
   → 17 keypoints per person
   ↓
5. PERSON TRACKING (DeepSORT)
   → Consistent IDs across frames
   ↓
6. KEYPOINT MATCHING (IoU)
   → Match poses to tracked people
   ↓
7. FEATURE EXTRACTION
   → Calculate movement metrics
   ↓
8. VIOLENCE CLASSIFICATION
   → Score each person
   ↓
9. CONFIRMATION COUNTER
   → Require 6+ consecutive frames
   ↓
10. INTERACTION ANALYSIS
    → Check multi-person violence
    ↓
11. FRAME RENDERING
    → Draw boxes and labels
    ↓
12. VIDEO OUTPUT
    → Annotated video + statistics
```

---

## Key Algorithms Summary

### 1. **Movement Detection Algorithm**
```
For each person:
  1. Extract current keypoints
  2. Compare to previous 15 frames
  3. Calculate:
     - Upper body displacement
     - Wrist acceleration
     - Movement variance
     - Hip stability
  4. Generate feature vector
```

### 2. **Violence Scoring Algorithm**
```
score = 0
IF upper_body > 40:
  score += 0.5 * min(upper_body/120, 1.0)
IF wrist_accel > 25:
  score += 0.4 * min(wrist_accel/60, 1.0)
IF variance > 120:
  score += 0.2 * min(variance/350, 1.0)
IF hip < 15 AND upper_body > 70:
  score += 0.15

is_violent = (score >= 0.8)
```

### 3. **Confirmation Algorithm**
```
For each person:
  IF classified as violent:
    counter[person] += 1
  ELSE:
    counter[person] = max(0, counter[person] - 1)
  
  IF counter[person] >= 6:
    TRIGGER ALERT
```

### 4. **Interaction Detection Algorithm**
```
For each pair of people:
  distance = euclidean_distance(center1, center2)
  
  IF distance < 150 pixels:
    IF both have high movement (> 80):
      Mark both as violent
```

---

## Performance Characteristics

### Time Complexity

| Component | Complexity | Notes |
|-----------|-----------|-------|
| Detection | O(1) | Fixed per frame |
| Tracking | O(n*m) | n=tracks, m=detections |
| Feature Extraction | O(n) | n=tracked persons |
| Classification | O(1) | Per person |
| Interaction | O(n²) | All pairs |
| Rendering | O(n) | Per person |

**Overall:** O(n²) where n = number of people in frame

### Space Complexity

| Component | Memory | Notes |
|-----------|--------|-------|
| Keypoint History | 30 KB/person | 15 frames × 17 keypoints |
| Tracker State | 100 bytes/person | DeepSORT overhead |
| Violent Frames | ~10 MB | 9 frames × 1920×1080 |

**Total:** ~30KB per person + 10MB for storage

---

## Configuration Parameters

### Tunable Thresholds

```python
# Detection
conf_threshold = 0.4  # Person detection confidence

# Tracking
max_age = 15  # Frames to keep lost tracks
n_init = 2    # Frames to confirm new track

# Features
upper_body_threshold = 40
wrist_accel_threshold = 25
variance_threshold = 120
hip_threshold = 15

# Classification
violence_score_threshold = 0.8
min_violence_frames = 6

# Interaction
proximity_threshold = 150  # pixels

# Storage
max_violent_frames = 9
frame_skip = 15
```

### Effect of Parameter Changes

**Increasing `min_violence_frames` (6 → 10):**
- ✅ Fewer false positives
- ❌ Slower detection
- ❌ Might miss brief violence

**Decreasing `violence_score_threshold` (0.8 → 0.6):**
- ✅ More sensitive detection
- ❌ More false positives
- ✅ Catches subtle violence

**Increasing `proximity_threshold` (150 → 200):**
- ✅ Detects violence at greater distance
- ❌ More false positives from unrelated people

---

## Edge Cases & Handling

### 1. **Occlusion** (Person Hidden)
```python
# Tracker keeps ID for max_age=15 frames
# When person reappears, same ID assigned
# History preserved during occlusion
```

### 2. **Multiple People**
```python
# Each person tracked independently
# Interaction analysis prevents false positives
# Pairwise comparison for all people
```

### 3. **Camera Motion**
```python
# All people move together → high movement
# But interaction analysis requires proximity
# Reduces false positives from camera shake
```

### 4. **Partial Visibility**
```python
# Keypoint matching requires IoU > 0.3
# If person partially visible, fewer keypoints
# Feature extraction handles missing keypoints
```

### 5. **Lighting Changes**
```python
# YOLO robust to lighting variations
# DeepSORT uses appearance + motion
# Movement features independent of brightness
```

---

## Summary

This violence detection system uses a **sophisticated multi-stage pipeline** combining:

1. **Computer Vision:** YOLOv8 for detection and pose estimation
2. **Tracking:** DeepSORT for identity consistency
3. **Physics:** Acceleration and velocity calculations
4. **Statistics:** Variance analysis for erratic movement
5. **Geometry:** Distance and IoU calculations
6. **Rule-based AI:** Weighted scoring system

**Key Innovation:** Combining individual movement analysis with multi-person interaction detection to reduce false positives while maintaining high sensitivity.

**Strengths:**
- No training data required (rule-based)
- Interpretable decisions (not black box)
- Configurable thresholds
- Handles multiple people
- Robust to occlusions

**Limitations:**
- Fixed thresholds (not adaptive)
- No weapon detection
- Requires visible body parts
- May struggle with unusual violence types

---

*For implementation details, see QUICK_IMPLEMENTATION_GUIDE.md*  
*For architecture overview, see Architecture.md*
