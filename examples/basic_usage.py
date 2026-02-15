from violence_detection_modular import (
    ModelLoader, PersonTracker, FeatureExtractor, ViolenceClassifier,
    InteractionAnalyzer, FrameRenderer, VideoProcessor, ImageProcessor
)
from typing import Dict, List
import matplotlib.pyplot as plt
import os


class ViolenceDetectionPipeline:
    def __init__(self, pose_model: str = 'yolov8n-pose.pt',
                 detection_model: str = 'yolov8n.pt'):
        self.model_loader = ModelLoader(pose_model, detection_model)
        self.tracker = PersonTracker(max_age=15, n_init=2)
        self.feature_extractor = FeatureExtractor()
        self.classifier = ViolenceClassifier()
        self.analyzer = InteractionAnalyzer()
        self.renderer = FrameRenderer()

        self.video_processor = VideoProcessor(
            self.model_loader,
            self.tracker,
            self.feature_extractor,
            self.classifier,
            self.analyzer,
            self.renderer
        )

        self.image_processor = ImageProcessor(
            self.model_loader,
            self.feature_extractor,
            self.classifier,
            self.renderer
        )

    def process_video(self, video_path: str, output_path: str = None,
                     min_violence_frames: int = 6) -> Dict:
        if output_path is None:
            base, ext = os.path.splitext(video_path)
            output_path = f"{base}_detected{ext}"

        stats = self.video_processor.process_video(
            video_path, output_path, 
            min_violence_frames=min_violence_frames
        )

        self._visualize_violent_frames(self.video_processor.violent_frames)
        return stats

    def process_image(self, image_path: str) -> Dict:
        result = self.image_processor.process_image(image_path)
        self._display_image_result(result)
        return result

    def process_image_batch(self, image_dir: str) -> List[Dict]:
        image_paths = [
            os.path.join(image_dir, f) 
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]

        results = self.image_processor.batch_process_images(image_paths)
        return results

    @staticmethod
    def _visualize_violent_frames(violent_frames: List):
        if not violent_frames:
            print("No violent frames detected.")
            return

        num_frames = len(violent_frames)
        cols = 3
        rows = (num_frames + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten() if num_frames > 1 else [axes]

        for i, (frame_idx, frame_img) in enumerate(violent_frames):
            ax = axes[i]
            import cv2
            ax.imshow(cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB))
            ax.set_title(f'Violent Frame {frame_idx}', fontweight='bold')
            ax.axis('off')

        for i in range(num_frames, len(axes)):
            axes[i].axis('off')

        plt.suptitle('Detected Violent Frames', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _display_image_result(result: Dict):
        import cv2
        results = result['results']
        annotated = result['annotated_frame']

        print(f"\nImage: {results['image_path']}")
        print(f"Total detections: {len(results['detections'])}")
        print(f"Violence detected: {results['has_violence']}")

        for i, det in enumerate(results['detections']):
            print(f"\nPerson {i + 1}:")
            print(f"  Confidence: {det['confidence']:.2f}")
            print(f"  Violence: {'Yes' if det['is_violent'] else 'No'}")
            print(f"  Features:")
            for key, val in det['features'].items():
                print(f"    {key}: {val:.2f}")

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Violence Detected: {results['has_violence']}", 
                    fontweight='bold', fontsize=14)
        ax.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    pipeline = ViolenceDetectionPipeline()

    print("=" * 60)
    print("VIDEO PROCESSING EXAMPLE")
    print("=" * 60)
    video_path = "input_video.mp4"
    if os.path.exists(video_path):
        stats = pipeline.process_video(video_path, min_violence_frames=6)
        print(f"\nVideo Statistics:")
        print(f"  Total frames: {stats['total_frames']}")
        print(f"  Violent frames: {stats['violent_frames']}")
    else:
        print(f"Video file not found: {video_path}")

    print("\n" + "=" * 60)
    print("IMAGE PROCESSING EXAMPLE")
    print("=" * 60)
    image_path = "sample_image.jpg"
    if os.path.exists(image_path):
        result = pipeline.process_image(image_path)
    else:
        print(f"Image file not found: {image_path}")

    print("\n" + "=" * 60)
    print("BATCH IMAGE PROCESSING EXAMPLE")
    print("=" * 60)
    image_dir = "images_directory"
    if os.path.isdir(image_dir):
        results = pipeline.process_image_batch(image_dir)
        print(f"\nBatch processing complete. Processed {len(results)} images.")
    else:
        print(f"Directory not found: {image_dir}")