import cv2
import numpy as np
from collections import deque
from math import sqrt
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def to_ltrb(self) -> Tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    def to_center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def width(self) -> float:
        return self.x2 - self.x1

    def height(self) -> float:
        return self.y2 - self.y1


@dataclass
class ViolenceFeatures:
    upper_body_movement: float = 0.0
    wrist_acceleration: float = 0.0
    movement_variance: float = 0.0
    hip_movement: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'upper_body_movement': self.upper_body_movement,
            'wrist_acceleration': self.wrist_acceleration,
            'movement_variance': self.movement_variance,
            'hip_movement': self.hip_movement
        }


@dataclass
class DetectionResult:
    track_id: int
    bbox: BoundingBox
    keypoints: np.ndarray
    is_violent: bool
    confidence: float
    features: ViolenceFeatures


class ModelLoader:
    def __init__(self, pose_model_path: str = 'yolov8n-pose.pt', 
                 detection_model_path: str = 'yolov8n.pt'):
        self.pose_model = YOLO(pose_model_path)
        self.detection_model = YOLO(detection_model_path)
        print(f"Models loaded: Pose={pose_model_path}, Detection={detection_model_path}")

    def detect_persons(self, frame: np.ndarray, conf_threshold: float = 0.4) -> List:
        detections = self.detection_model(frame, verbose=False)
        person_detections = []
        
        for det in detections[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = det
            if int(cls) == 0 and conf > conf_threshold:
                person_detections.append([[x1, y1, x2 - x1, y2 - y1], conf, "person"])
        
        return person_detections

    def extract_pose(self, frame: np.ndarray):
        return self.pose_model(frame, verbose=False)


class PersonTracker:
    def __init__(self, max_age: int = 15, n_init: int = 2):
        self.tracker = DeepSort(max_age=max_age, n_init=n_init)

    def update(self, detections: List, frame: np.ndarray) -> List:
        return self.tracker.update_tracks(detections, frame=frame)


class FeatureExtractor:
    @staticmethod
    def calculate_iou(box1: BoundingBox, box2: Tuple) -> float:
        x1, y1, x2, y2 = box1.to_ltrb()
        gx1, gy1, gx2, gy2 = box2
        ix1 = max(x1, gx1)
        iy1 = max(y1, gy1)
        ix2 = min(x2, gx2)
        iy2 = min(y2, gy2)
        
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter_area = iw * ih
        
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (gx2 - gx1) * (gy2 - gy1)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

    @staticmethod
    def match_keypoints(bbox: BoundingBox, pose_results) -> Optional[np.ndarray]:
        if not pose_results or len(pose_results) == 0 or not pose_results[0].boxes:
            return None

        best_keypoints = None
        max_iou = 0

        for i in range(len(pose_results[0].boxes)):
            pose_box = pose_results[0].boxes.xyxy[i].cpu().numpy()
            keypoints = pose_results[0].keypoints.xy[i].cpu().numpy()
            iou = FeatureExtractor.calculate_iou(bbox, pose_box)
            
            if iou > max_iou:
                max_iou = iou
                best_keypoints = keypoints

        return best_keypoints if max_iou > 0.3 else None

    @staticmethod
    def extract_movement_features(current_kpts: np.ndarray, 
                                  history: deque, 
                                  fps: float = 30) -> ViolenceFeatures:
        features = ViolenceFeatures()

        if len(history) < 2:
            return features

        prev_kpts = history[-1]
        if len(current_kpts) != len(prev_kpts):
            return features

        upper_body_indices = [0, 5, 6, 7, 8, 9, 10]
        upper_movement = sum(
            sqrt((current_kpts[i][0] - prev_kpts[i][0]) ** 2 + 
                 (current_kpts[i][1] - prev_kpts[i][1]) ** 2)
            for i in upper_body_indices if i < len(current_kpts)
        )
        features.upper_body_movement = upper_movement

        if len(history) >= 3:
            wrist_indices = [9, 10]
            max_accel = 0
            for wrist_idx in wrist_indices:
                if wrist_idx < len(current_kpts):
                    curr = np.array(current_kpts[wrist_idx])
                    prev = np.array(history[-1][wrist_idx])
                    prev_prev = np.array(history[-2][wrist_idx])
                    accel = np.linalg.norm((curr - prev) - (prev - prev_prev))
                    max_accel = max(max_accel, accel)
            features.wrist_acceleration = max_accel

        if len(history) >= 5:
            movements = []
            for i in range(len(history) - 1):
                curr_h = history[i + 1]
                prev_h = history[i]
                if len(curr_h) == len(prev_h):
                    move = sum(
                        sqrt((c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2)
                        for c, p in zip(curr_h, prev_h)
                    )
                    movements.append(move)
            features.movement_variance = float(np.var(movements)) if movements else 0

        hip_indices = [11, 12]
        hip_movement = sum(
            sqrt((current_kpts[i][0] - prev_kpts[i][0]) ** 2 + 
                 (current_kpts[i][1] - prev_kpts[i][1]) ** 2)
            for i in hip_indices if i < len(current_kpts)
        )
        features.hip_movement = hip_movement

        return features


class ViolenceClassifier:
    @staticmethod
    def classify(features: ViolenceFeatures) -> Tuple[bool, float]:
        score = 0.0

        if features.upper_body_movement > 40:
            score += 0.5 * min(features.upper_body_movement / 120, 1.0)

        if features.wrist_acceleration > 25:
            score += 0.4 * min(features.wrist_acceleration / 60, 1.0)

        if features.movement_variance > 120:
            score += 0.2 * min(features.movement_variance / 350, 1.0)

        if features.hip_movement < 15 and features.upper_body_movement > 70:
            score += 0.15

        is_violent = score >= 0.8
        return is_violent, score


class InteractionAnalyzer:
    @staticmethod
    def detect_interaction_violence(track_data: Dict, 
                                    proximity_threshold: float = 150) -> Dict[int, float]:
        violence_scores = {tid: 0 for tid in track_data.keys()}
        track_ids = list(track_data.keys())

        for i, tid1 in enumerate(track_ids):
            for j, tid2 in enumerate(track_ids):
                if i >= j:
                    continue

                bbox1 = track_data[tid1]['bbox']
                bbox2 = track_data[tid2]['bbox']
                center1 = bbox1.to_center()
                center2 = bbox2.to_center()

                distance = sqrt((center1[0] - center2[0]) ** 2 + 
                              (center1[1] - center2[1]) ** 2)

                if distance < proximity_threshold:
                    feat1 = track_data[tid1]['features']
                    feat2 = track_data[tid2]['features']

                    if feat1.upper_body_movement > 80 and feat2.upper_body_movement > 80:
                        violence_scores[tid1] += 1
                        violence_scores[tid2] += 1

        return violence_scores


class FrameRenderer:
    @staticmethod
    def render_detections(frame: np.ndarray, 
                         tracks, 
                         violent_ids: set,
                         global_violence_alert: bool) -> np.ndarray:
        annotated = frame.copy()
        height, width = frame.shape[:2]

        for track in tracks:
            if not track.is_confirmed():
                continue

            x1, y1, x2, y2 = map(int, track.to_ltrb())
            track_id = track.track_id
            is_violent = track_id in violent_ids

            color = (0, 0, 255) if is_violent else (255, 255, 0)
            label = "Violence" if is_violent else ""

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)

            if label:
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                label_y = max(y1 - 8, 15)
                cv2.rectangle(annotated, (x1, label_y - lh - 4), 
                            (x1 + lw + 4, label_y + 2), color, -1)
                cv2.putText(annotated, label, (x1 + 2, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        FrameRenderer._render_status_bar(annotated, global_violence_alert, width)
        return annotated

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
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    @staticmethod
    def add_frame_info(frame: np.ndarray, frame_idx: int):
        height, width = frame.shape[:2]
        cv2.putText(frame, f"Frame: {frame_idx}", (width - 180, 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


class VideoProcessor:
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
        
        self.person_histories = {}
        self.violence_confirmation = {}
        self.violent_frames = []

    def process_video(self, video_path: str, output_path: str, 
                     min_violence_frames: int = 6,
                     frame_skip: int = 15) -> Dict:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                            fps, (width, height))

        frame_idx = 0
        last_violent_frame = -1
        stats = {'total_frames': 0, 'violent_frames': 0, 'violent_events': []}

        print(f"Processing video: {width}x{height} @ {fps} FPS")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            stats['total_frames'] += 1

            person_dets = self.model_loader.detect_persons(frame)
            pose_results = self.model_loader.extract_pose(frame)

            tracks = self.tracker.update(person_dets, frame)

            violent_ids = set()
            track_data = {}

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                bbox_ltrb = np.array(track.to_ltrb())
                bbox = BoundingBox(*bbox_ltrb)

                keypoints = self.feature_extractor.match_keypoints(bbox, pose_results)

                if keypoints is not None:
                    if track_id not in self.person_histories:
                        self.person_histories[track_id] = deque(maxlen=15)
                        self.violence_confirmation[track_id] = 0

                    features = self.feature_extractor.extract_movement_features(
                        keypoints, self.person_histories[track_id], fps
                    )
                    self.person_histories[track_id].append(keypoints)

                    is_violent, confidence = self.classifier.classify(features)
                    track_data[track_id] = {'bbox': bbox, 'features': features}

                    if is_violent:
                        self.violence_confirmation[track_id] += 1
                    else:
                        self.violence_confirmation[track_id] = max(0, 
                            self.violence_confirmation[track_id] - 1)

                    if self.violence_confirmation[track_id] >= min_violence_frames:
                        violent_ids.add(track_id)

            if len(track_data) > 1:
                interaction_scores = self.analyzer.detect_interaction_violence(track_data)
                for tid, int_score in interaction_scores.items():
                    if int_score > 0:
                        violent_ids.add(tid)

            global_violence_alert = len(violent_ids) > 0

            annotated = self.renderer.render_detections(frame, tracks, 
                                                       violent_ids, global_violence_alert)
            self.renderer.add_frame_info(annotated, frame_idx)

            if global_violence_alert and len(self.violent_frames) < 9 and \
               (frame_idx - last_violent_frame >= frame_skip or last_violent_frame == -1):
                self.violent_frames.append((frame_idx, annotated.copy()))
                last_violent_frame = frame_idx
                stats['violent_frames'] += 1

            if global_violence_alert:
                stats['violent_events'].append(frame_idx)

            out.write(annotated)

            if frame_idx % 30 == 0:
                status = "VIOLENCE DETECTED!" if global_violence_alert else "Normal"
                print(f"Processed {frame_idx} frames... Status: {status}")

        cap.release()
        out.release()

        print(f"Video processing complete! Output saved to: {output_path}")
        return stats


class ImageProcessor:
    def __init__(self, model_loader: ModelLoader,
                 feature_extractor: FeatureExtractor,
                 violence_classifier: ViolenceClassifier,
                 frame_renderer: FrameRenderer):
        self.model_loader = model_loader
        self.feature_extractor = feature_extractor
        self.classifier = violence_classifier
        self.renderer = frame_renderer

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

        for det in person_dets:
            x1, y1, w, h = det[0]
            x2, y2 = x1 + w, y1 + h
            bbox = BoundingBox(x1, y1, x2, y2)

            keypoints = self.feature_extractor.match_keypoints(bbox, pose_results)

            if keypoints is not None:
                features = self.feature_extractor.extract_movement_features(
                    keypoints, deque(maxlen=1)
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

        annotated = self._annotate_image(frame, results)
        return {
            'results': results,
            'annotated_frame': annotated
        }

    def _annotate_image(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        annotated = frame.copy()

        for det in results['detections']:
            x1, y1, x2, y2 = det['bbox']
            color = (0, 0, 255) if det['is_violent'] else (255, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"Violence ({det['confidence']:.2f})" if det['is_violent'] else "Person"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x1, y1 - lh - 8), 
                        (x1 + lw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return annotated

    def batch_process_images(self, image_paths: List[str]) -> List[Dict]:
        results = []
        for img_path in image_paths:
            try:
                result = self.process_image(img_path)
                results.append(result)
                print(f"Processed: {img_path}")
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")

        return results