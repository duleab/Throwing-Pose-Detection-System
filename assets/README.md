# Assets Directory

This directory contains demo images, videos, and output examples for the Violence Detection System.

## 📁 Structure

```
assets/
├── demo_images/          # Input demo images
│   ├── demo1.jpg
│   ├── demo2.jpg
│   ├── demo3.jpg
│   └── ...
├── demo_videos/          # Input demo videos
│   ├── demo1.mp4
│   ├── demo2.mp4
│   └── ...
└── output_examples/      # Processed output examples
    ├── output1.jpg
    ├── output2.jpg
    ├── output3.jpg
    └── ...
```

## 📸 Demo Images

Place your input demo images in `demo_images/` folder.

**Recommended naming:**
- `demo1.jpg` - Normal behavior
- `demo2.jpg` - Violent behavior
- `demo3.jpg` - Throwing pose
- `demo4.jpg` - Multi-person
- etc.

## 🎥 Demo Videos

Place your input demo videos in `demo_videos/` folder.

**Recommended naming:**
- `demo1.mp4` - Normal behavior
- `demo2.mp4` - Violent behavior
- `demo3.mp4` - Fighting scene
- etc.

## 🖼️ Output Examples

Place your processed output images in `output_examples/` folder.

**Recommended naming:**
- `output1.jpg` - Detected violence with skeleton
- `output2.jpg` - Normal detection
- `output3.jpg` - Throwing pose detection
- `output4.jpg` - Multi-person detection
- etc.

## 📝 Usage in README

These assets are referenced in the main README.md file to showcase the system's capabilities.

## 🔒 Git LFS (Optional)

For large video files, consider using Git LFS:

```bash
git lfs install
git lfs track "assets/demo_videos/*.mp4"
git lfs track "assets/demo_videos/*.avi"
git add .gitattributes
```

## 📊 File Size Recommendations

- **Images:** Keep under 1 MB each (use JPG with 80-90% quality)
- **Videos:** Keep under 10 MB each (use compressed MP4)
- **Total assets:** Keep under 50 MB for GitHub

## 🎨 Image Guidelines

For best showcase results:
- Use high-quality images (1920x1080 or similar)
- Show clear examples of violence detection
- Include variety: single person, multiple people, different poses
- Include both violent and non-violent examples
- Show skeleton visualization clearly

## 🎬 Video Guidelines

For best showcase results:
- Use short clips (5-15 seconds)
- Show clear violence detection scenarios
- Include timeline visualization
- Show before/after comparison
- Compress for web (H.264 codec recommended)
