import numpy as np
from collections import deque
from violence_detection_modular import (
    BoundingBox, ViolenceFeatures, FeatureExtractor,
    ViolenceClassifier, InteractionAnalyzer, PersonTracker
)


class TestViolenceDetection:
    """Unit tests for individual components"""

    @staticmethod
    def test_bounding_box():
        """Verify BoundingBox utility methods"""
        bbox = BoundingBox(100, 50, 200, 300)
        
        assert bbox.width() == 100
        assert bbox.height() == 250
        
        center = bbox.to_center()
        assert center == (150, 175)
        
        ltrb = bbox.to_ltrb()
        assert ltrb == (100, 50, 200, 300)
        
        print("✓ BoundingBox tests passed")

    @staticmethod
    def test_feature_extraction():
        """Verify feature calculation with synthetic data"""
        extractor = FeatureExtractor()
        
        current_kpts = np.array([
            [100, 100],  # nose
            [90, 80],    # left_eye
            [110, 80],   # right_eye
            [85, 75],    # left_ear
            [115, 75],   # right_ear
            [80, 150],   # left_shoulder
            [120, 150],  # right_shoulder
            [75, 200],   # left_elbow
            [125, 200],  # right_elbow
            [70, 250],   # left_wrist
            [130, 250],  # right_wrist
            [90, 300],   # left_hip
            [110, 300],  # right_hip
            [85, 350],   # left_knee
            [115, 350],  # right_knee
            [80, 400],   # left_ankle
            [120, 400],  # right_ankle
        ])
        
        prev_kpts = current_kpts + np.random.randn(17, 2) * 5
        history = deque([prev_kpts], maxlen=15)
        
        features = extractor.extract_movement_features(current_kpts, history)
        
        assert isinstance(features.upper_body_movement, float)
        assert isinstance(features.wrist_acceleration, float)
        assert isinstance(features.movement_variance, float)
        assert isinstance(features.hip_movement, float)
        
        assert features.upper_body_movement >= 0
        assert features.wrist_acceleration >= 0
        
        print("✓ Feature extraction tests passed")

    @staticmethod
    def test_violence_classification():
        """Verify violence classification logic"""
        classifier = ViolenceClassifier()
        
        normal_features = ViolenceFeatures(
            upper_body_movement=20,
            wrist_acceleration=10,
            movement_variance=50,
            hip_movement=15
        )
        is_violent, score = classifier.classify(normal_features)
        assert not is_violent
        assert score < 0.8
        
        violent_features = ViolenceFeatures(
            upper_body_movement=100,
            wrist_acceleration=50,
            movement_variance=200,
            hip_movement=5
        )
        is_violent, score = classifier.classify(violent_features)
        assert is_violent
        assert score >= 0.8
        
        edge_case_features = ViolenceFeatures(
            upper_body_movement=0,
            wrist_acceleration=0,
            movement_variance=0,
            hip_movement=0
        )
        is_violent, score = classifier.classify(edge_case_features)
        assert not is_violent
        assert score == 0.0
        
        print("✓ Violence classification tests passed")

    @staticmethod
    def test_iou_calculation():
        """Verify intersection over union computation"""
        extractor = FeatureExtractor()
        
        bbox1 = BoundingBox(0, 0, 100, 100)
        box2 = (0, 0, 100, 100)
        iou = extractor.calculate_iou(bbox1, box2)
        assert iou == 1.0
        
        bbox3 = BoundingBox(0, 0, 50, 50)
        box4 = (25, 25, 75, 75)
        iou2 = extractor.calculate_iou(bbox3, box4)
        assert 0.1 < iou2 < 0.3
        
        bbox5 = BoundingBox(0, 0, 50, 50)
        box6 = (200, 200, 250, 250)
        iou3 = extractor.calculate_iou(bbox5, box6)
        assert iou3 == 0.0
        
        print("✓ IoU calculation tests passed")

    @staticmethod
    def test_interaction_analysis():
        """Verify multi-person violence detection"""
        analyzer = InteractionAnalyzer()
        
        track_data = {
            1: {
                'bbox': BoundingBox(0, 0, 100, 100),
                'features': ViolenceFeatures(upper_body_movement=90)
            },
            2: {
                'bbox': BoundingBox(80, 20, 180, 120),  # Close to person 1
                'features': ViolenceFeatures(upper_body_movement=85)
            }
        }
        
        scores = analyzer.detect_interaction_violence(track_data)
        assert scores[1] == 1
        assert scores[2] == 1
        
        track_data_distant = {
            1: {
                'bbox': BoundingBox(0, 0, 100, 100),
                'features': ViolenceFeatures(upper_body_movement=90)
            },
            2: {
                'bbox': BoundingBox(500, 500, 600, 600),  # Far from person 1
                'features': ViolenceFeatures(upper_body_movement=85)
            }
        }
        
        scores2 = analyzer.detect_interaction_violence(track_data_distant)
        assert scores2[1] == 0
        assert scores2[2] == 0
        
        print("✓ Interaction analysis tests passed")


class UsagePatterns:
    """Best practice examples for using the modular system"""

    @staticmethod
    def basic_video_processing():
        """Simplest way to process a video file"""
        from violence_detection_modular import ModelLoader, PersonTracker, FrameRenderer
        from violence_detection_modular import FeatureExtractor, ViolenceClassifier
        from violence_detection_modular import InteractionAnalyzer, VideoProcessor
        
        components = [
            ModelLoader(),
            PersonTracker(),
            FeatureExtractor(),
            ViolenceClassifier(),
            InteractionAnalyzer(),
            FrameRenderer()
        ]
        
        processor = VideoProcessor(*components)
        stats = processor.process_video('input.mp4', 'output.mp4')
        
        print(f"Processed {stats['total_frames']} frames")
        print(f"Violent events detected: {len(stats['violent_events'])}")

    @staticmethod
    def advanced_video_with_custom_settings():
        """Video processing with custom parameters"""
        from violence_detection_modular import (
            ModelLoader, PersonTracker, VideoProcessor, FrameRenderer,
            FeatureExtractor, ViolenceClassifier, InteractionAnalyzer
        )
        
        model_loader = ModelLoader(
            pose_model_path='yolov8m-pose.pt',  # Medium model for accuracy
            detection_model_path='yolov8m.pt'
        )
        
        tracker = PersonTracker(max_age=30, n_init=3)  # More persistent tracking
        
        processor = VideoProcessor(
            model_loader,
            tracker,
            FeatureExtractor(),
            ViolenceClassifier(),
            InteractionAnalyzer(),
            FrameRenderer()
        )
        
        stats = processor.process_video(
            'surveillance_footage.mp4',
            'output_annotated.mp4',
            min_violence_frames=8,  # Stricter confirmation
            frame_skip=20
        )
        
        return stats

    @staticmethod
    def single_image_analysis():
        """Analyze a single image for violence indicators"""
        from violence_detection_modular import (
            ModelLoader, ImageProcessor, FeatureExtractor,
            ViolenceClassifier, FrameRenderer
        )
        
        components = [
            ModelLoader(),
            FeatureExtractor(),
            ViolenceClassifier(),
            FrameRenderer()
        ]
        
        processor = ImageProcessor(*components[:1], *components[1:])
        result = processor.process_image('photo.jpg')
        
        print(f"Violence detected: {result['results']['has_violence']}")
        for i, det in enumerate(result['results']['detections']):
            print(f"Person {i}: Violence={det['is_violent']}, Score={det['confidence']:.2f}")

    @staticmethod
    def batch_image_processing():
        """Process multiple images from a directory"""
        from violence_detection_modular import (
            ModelLoader, ImageProcessor, FeatureExtractor,
            ViolenceClassifier, FrameRenderer
        )
        import os
        
        processor = ImageProcessor(
            ModelLoader(),
            FeatureExtractor(),
            ViolenceClassifier(),
            FrameRenderer()
        )
        
        image_dir = 'cctv_frames'
        results = processor.batch_process_images(
            [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
        )
        
        violent_count = sum(1 for r in results if r['results']['has_violence'])
        print(f"Violent frames: {violent_count} / {len(results)}")

    @staticmethod
    def custom_classifier():
        """Extend system with custom violence detection rules"""
        from violence_detection_modular import (
            ViolenceClassifier, ViolenceFeatures, ModelLoader,
            PersonTracker, FrameRenderer, FeatureExtractor,
            InteractionAnalyzer, VideoProcessor
        )
        from typing import Tuple
        
        class StrictViolenceClassifier(ViolenceClassifier):
            @staticmethod
            def classify(features: ViolenceFeatures) -> Tuple[bool, float]:
                is_violent, base_score = ViolenceClassifier.classify(features)
                
                if is_violent:
                    if features.wrist_acceleration < 20:
                        is_violent = False
                    elif features.hip_movement > 50:
                        is_violent = False
                
                return is_violent, base_score
        
        processor = VideoProcessor(
            ModelLoader(),
            PersonTracker(),
            FeatureExtractor(),
            StrictViolenceClassifier(),
            InteractionAnalyzer(),
            FrameRenderer()
        )
        
        return processor

    @staticmethod
    def real_time_processing_with_logging():
        """Stream processing with detailed logging"""
        import logging
        from violence_detection_modular import (
            ModelLoader, PersonTracker, VideoProcessor, FrameRenderer,
            FeatureExtractor, ViolenceClassifier, InteractionAnalyzer
        )
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
        
        processor = VideoProcessor(
            ModelLoader(),
            PersonTracker(),
            FeatureExtractor(),
            ViolenceClassifier(),
            InteractionAnalyzer(),
            FrameRenderer()
        )
        
        try:
            stats = processor.process_video('stream.mp4', 'output.mp4')
            logger.info(f"Processing completed: {stats}")
        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            raise

    @staticmethod
    def performance_profiling():
        """Profile component performance"""
        import cProfile
        import pstats
        from io import StringIO
        from violence_detection_modular import VideoProcessor, ModelLoader
        from violence_detection_modular import PersonTracker, FrameRenderer
        from violence_detection_modular import FeatureExtractor, ViolenceClassifier
        from violence_detection_modular import InteractionAnalyzer
        
        processor = VideoProcessor(
            ModelLoader(),
            PersonTracker(),
            FeatureExtractor(),
            ViolenceClassifier(),
            InteractionAnalyzer(),
            FrameRenderer()
        )
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        processor.process_video('test_video.mp4', 'output.mp4')
        
        profiler.disable()
        
        stream = StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('cumulative')
        stats.print_stats(20)
        
        print(stream.getvalue())


class BestPractices:
    """Recommended patterns and guidelines"""

    @staticmethod
    def validate_input_frames():
        """Always validate frame quality"""
        import cv2
        
        def validate_frame(frame):
            if frame is None:
                raise ValueError("Frame is None")
            if not isinstance(frame, np.ndarray):
                raise TypeError("Frame must be numpy array")
            if frame.dtype != np.uint8:
                raise TypeError("Frame must be uint8")
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                raise ValueError("Frame must be RGB/BGR with 3 channels")
            if frame.shape[0] < 100 or frame.shape[1] < 100:
                raise ValueError("Frame too small (min 100x100)")
            
            return True
        
        return validate_frame

    @staticmethod
    def handle_corrupted_videos():
        """Graceful error handling for problematic videos"""
        import cv2
        
        def safe_process_video(video_path, processor):
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count == 0:
                raise ValueError("Video has no frames")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps > 120:
                fps = 30  # Use default for invalid fps
            
            cap.release()
            
            return processor.process_video(video_path, 'output.mp4')
        
        return safe_process_video

    @staticmethod
    def memory_efficient_processing():
        """Process large videos without memory issues"""
        from violence_detection_modular import VideoProcessor
        import gc
        
        def extended_video_processor(processor: VideoProcessor, video_path: str):
            stats = processor.process_video(video_path, 'output.mp4')
            
            gc.collect()
            
            if len(processor.violent_frames) > 0:
                print(f"Captured {len(processor.violent_frames)} violent frames")
            
            return stats
        
        return extended_video_processor

    @staticmethod
    def configuration_best_practices():
        """Configuration recommendations for different scenarios"""
        
        configs = {
            'high_precision': {
                'min_violence_frames': 10,
                'classifier_threshold': 0.85,
                'proximity_threshold': 120
            },
            'real_time': {
                'min_violence_frames': 4,
                'classifier_threshold': 0.75,
                'proximity_threshold': 150
            },
            'surveillance': {
                'min_violence_frames': 6,
                'classifier_threshold': 0.80,
                'proximity_threshold': 140
            },
            'development': {
                'min_violence_frames': 3,
                'classifier_threshold': 0.70,
                'proximity_threshold': 160
            }
        }
        
        return configs


if __name__ == "__main__":
    print("Testing Core Components")
    print("=" * 50)
    
    TestViolenceDetection.test_bounding_box()
    TestViolenceDetection.test_feature_extraction()
    TestViolenceDetection.test_violence_classification()
    TestViolenceDetection.test_iou_calculation()
    TestViolenceDetection.test_interaction_analysis()
    
    print("\nAll tests passed!")
    
    configs = BestPractices.configuration_best_practices()
    print("\nRecommended configurations:")
    for scenario, config in configs.items():
        print(f"\n{scenario.upper()}:")
        for key, value in config.items():
            print(f"  {key}: {value}")