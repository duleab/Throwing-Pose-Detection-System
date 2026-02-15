# 📸 How to Add Demo Images and Videos to README

## 🎯 Quick Guide

Your README is now set up with a **4x2 grid** for images and **2x2 grid** for videos. Follow these steps to add your demo files.

---

## 📁 Step 1: Organize Your Files

### **Directory Structure:**
```
assets/
├── demo_images/          # Video thumbnails (8 images)
│   ├── demo1.jpg
│   ├── demo2.jpg
│   ├── demo3.jpg
│   └── demo4.jpg
├── demo_videos/          # Demo videos (4 videos)
│   ├── demo1.mp4
│   ├── demo2.mp4
│   ├── demo3.mp4
│   └── demo4.mp4
└── output_examples/      # Detection results (8 images)
    ├── output1.jpg
    ├── output2.jpg
    ├── output3.jpg
    ├── output4.jpg
    ├── output5.jpg
    ├── output6.jpg
    ├── output7.jpg
    └── output8.jpg
```

---

## 🖼️ Step 2: Prepare Output Images (4x2 Grid)

### **Required: 8 Images**

Place your detection result images in `assets/output_examples/`:

**Row 1 (Violence Detection Examples):**
1. `output1.jpg` - Throwing Pose Detection
2. `output2.jpg` - Fighting Scene
3. `output3.jpg` - Multi-Person Detection
4. `output4.jpg` - Striking Pose

**Row 2 (Normal & Feature Examples):**
5. `output5.jpg` - Normal Behavior
6. `output6.jpg` - Skeleton Visualization
7. `output7.jpg` - Custom Colors
8. `output8.jpg` - With Threshold Display

### **Image Specifications:**
- **Format:** JPG or PNG
- **Resolution:** 1920x1080 or 1280x720 (16:9 aspect ratio)
- **File Size:** < 500 KB each (compress if needed)
- **Quality:** 80-90% JPG quality

### **How to Create:**
1. Run your Streamlit app
2. Upload images and analyze
3. Take screenshots of results
4. Crop to show just the detection output
5. Resize to recommended resolution
6. Save with correct filenames

---

## 🎥 Step 3: Prepare Demo Videos (2x2 Grid)

### **Required: 4 Videos + 4 Thumbnails**

**Videos** (`assets/demo_videos/`):
1. `demo1.mp4` - Fighting Scene Detection
2. `demo2.mp4` - Multi-Person Tracking
3. `demo3.mp4` - Throwing Pose Detection
4. `demo4.mp4` - Timeline Visualization

**Thumbnails** (`assets/demo_images/`):
1. `demo1.jpg` - Thumbnail for demo1.mp4
2. `demo2.jpg` - Thumbnail for demo2.mp4
3. `demo3.jpg` - Thumbnail for demo3.mp4
4. `demo4.jpg` - Thumbnail for demo4.mp4

### **Video Specifications:**
- **Format:** MP4 (H.264 codec)
- **Duration:** 5-15 seconds
- **Resolution:** 1280x720 (720p)
- **File Size:** < 10 MB each
- **FPS:** 24-30 fps

### **Thumbnail Specifications:**
- **Format:** JPG
- **Resolution:** 1280x720 (same as video)
- **File Size:** < 300 KB each
- **Quality:** 85% JPG quality
- **Content:** First frame or representative frame from video

### **How to Create Thumbnails:**

**Using FFmpeg:**
```bash
# Extract first frame as thumbnail
ffmpeg -i demo1.mp4 -ss 00:00:01 -vframes 1 demo1.jpg

# Or extract a specific frame (e.g., at 3 seconds)
ffmpeg -i demo1.mp4 -ss 00:00:03 -vframes 1 demo1.jpg
```

**Using VLC:**
1. Open video in VLC
2. Pause at desired frame
3. Video → Take Snapshot
4. Rename and move to `assets/demo_images/`

**Using Python:**
```python
import cv2

# Extract first frame
cap = cv2.VideoCapture('demo1.mp4')
ret, frame = cap.read()
cv2.imwrite('demo1.jpg', frame)
cap.release()
```

---

## 🎨 Step 4: Optimize Images

### **Compress Images (if too large):**

**Using Online Tools:**
- TinyPNG: https://tinypng.com/
- Squoosh: https://squoosh.app/

**Using Command Line:**
```bash
# Using ImageMagick
magick output1.jpg -quality 85 -resize 1920x1080 output1_compressed.jpg

# Using FFmpeg
ffmpeg -i output1.jpg -q:v 3 output1_compressed.jpg
```

**Using Python:**
```python
from PIL import Image

img = Image.open('output1.jpg')
img = img.resize((1920, 1080), Image.LANCZOS)
img.save('output1_compressed.jpg', quality=85, optimize=True)
```

---

## 📦 Step 5: Compress Videos

### **Using FFmpeg:**

```bash
# Compress video to < 10 MB
ffmpeg -i input.mp4 -vcodec h264 -acodec aac -crf 28 -preset fast demo1.mp4

# For smaller file size
ffmpeg -i input.mp4 -vcodec h264 -acodec aac -crf 32 -preset fast demo1.mp4

# Resize to 720p
ffmpeg -i input.mp4 -vf scale=1280:720 -crf 28 demo1.mp4
```

### **Using Online Tools:**
- CloudConvert: https://cloudconvert.com/mp4-converter
- FreeConvert: https://www.freeconvert.com/video-compressor

---

## ✅ Step 6: Verify File Names

### **Checklist:**

**Output Examples (8 files):**
- [ ] `assets/output_examples/output1.jpg`
- [ ] `assets/output_examples/output2.jpg`
- [ ] `assets/output_examples/output3.jpg`
- [ ] `assets/output_examples/output4.jpg`
- [ ] `assets/output_examples/output5.jpg`
- [ ] `assets/output_examples/output6.jpg`
- [ ] `assets/output_examples/output7.jpg`
- [ ] `assets/output_examples/output8.jpg`

**Demo Videos (4 files):**
- [ ] `assets/demo_videos/demo1.mp4`
- [ ] `assets/demo_videos/demo2.mp4`
- [ ] `assets/demo_videos/demo3.mp4`
- [ ] `assets/demo_videos/demo4.mp4`

**Demo Thumbnails (4 files):**
- [ ] `assets/demo_images/demo1.jpg`
- [ ] `assets/demo_images/demo2.jpg`
- [ ] `assets/demo_images/demo3.jpg`
- [ ] `assets/demo_images/demo4.jpg`

---

## 🚀 Step 7: Test Locally

### **View README Locally:**

**Using VS Code:**
1. Open README.md
2. Press `Ctrl+Shift+V` (or `Cmd+Shift+V` on Mac)
3. Preview will show images

**Using Grip (GitHub README Preview):**
```bash
pip install grip
grip README.md
# Open http://localhost:6419
```

**Using Browser:**
```bash
# Simple Python server
python -m http.server 8000
# Open http://localhost:8000/README.md
```

---

## 📤 Step 8: Push to GitHub

### **Option 1: Include Assets in Git**

```bash
# Add all assets
git add assets/

# Commit
git commit -m "Add demo images and videos"

# Push
git push
```

### **Option 2: Use Git LFS (for large files)**

```bash
# Install Git LFS
git lfs install

# Track video files
git lfs track "assets/demo_videos/*.mp4"

# Add .gitattributes
git add .gitattributes

# Add assets
git add assets/

# Commit and push
git commit -m "Add demo assets with Git LFS"
git push
```

---

## 🎨 Customization Options

### **Change Image Captions:**

Edit `README.md` and modify the `<p align="center"><b>Caption</b></p>` lines:

```html
<p align="center"><b>Your Custom Caption</b></p>
```

### **Change Grid Layout:**

**For 3x3 Grid (9 images):**
```html
<td width="33.33%">
```

**For 5x2 Grid (10 images):**
```html
<td width="20%">
```

### **Add More Videos:**

Copy the table row structure and add more:

```html
<tr>
  <td width="50%">
    <h4>🎥 Your Video Title</h4>
    <a href="assets/demo_videos/demo5.mp4">
      <img src="assets/demo_images/demo5.jpg" alt="Video Demo 5" width="100%"/>
    </a>
    <p><b>Description</b> - Details about the video</p>
  </td>
  <td width="50%">
    <!-- Another video -->
  </td>
</tr>
```

---

## 💡 Tips for Best Results

### **Image Tips:**
1. **Consistent Aspect Ratio** - Use 16:9 for all images
2. **Clear Visibility** - Ensure skeletons and bounding boxes are clearly visible
3. **Variety** - Show different scenarios (violence, normal, multi-person)
4. **Quality** - Use high-quality source images
5. **Lighting** - Ensure good lighting in source images

### **Video Tips:**
1. **Short Duration** - Keep videos under 15 seconds
2. **Clear Action** - Show clear examples of detection
3. **Smooth Playback** - Use 24-30 FPS
4. **Good Compression** - Balance quality and file size
5. **Representative Thumbnails** - Choose frames that show the action

### **General Tips:**
1. **Test Links** - Verify all image/video links work
2. **Check File Sizes** - Keep total assets under 50 MB
3. **Use Descriptive Names** - Name files clearly
4. **Optimize for Web** - Compress for faster loading
5. **Update .gitignore** - If files are too large, exclude and host elsewhere

---

## 🔧 Troubleshooting

### **Images Not Showing:**
- Check file paths are correct
- Verify files exist in correct directories
- Ensure file names match exactly (case-sensitive)
- Check file extensions (.jpg vs .jpeg)

### **Videos Not Playing:**
- Ensure MP4 format with H.264 codec
- Check file size (< 10 MB recommended)
- Verify GitHub supports the video format
- Consider hosting large videos externally

### **File Too Large:**
- Compress images/videos
- Use Git LFS for large files
- Host videos on YouTube/Vimeo and embed
- Reduce resolution

---

## 📊 Alternative: External Hosting

### **For Large Videos:**

**YouTube:**
1. Upload video to YouTube
2. Get embed code
3. Replace video link in README

```html
<a href="https://www.youtube.com/watch?v=YOUR_VIDEO_ID">
  <img src="assets/demo_images/demo1.jpg" alt="Video Demo 1" width="100%"/>
</a>
```

**Or embed directly:**
```html
<iframe width="100%" height="315" src="https://www.youtube.com/embed/YOUR_VIDEO_ID" frameborder="0" allowfullscreen></iframe>
```

---

## ✅ Final Checklist

Before pushing to GitHub:

- [ ] All 8 output images added
- [ ] All 4 demo videos added
- [ ] All 4 video thumbnails added
- [ ] File sizes optimized
- [ ] File names match README exactly
- [ ] Images display correctly locally
- [ ] Video links work
- [ ] README preview looks good
- [ ] .gitignore updated if needed
- [ ] Git LFS configured if using large files

---

## 🎉 You're Done!

Your README now has:
- ✅ **4x2 grid** of detection result images
- ✅ **2x2 grid** of video demonstrations
- ✅ Professional showcase of your system
- ✅ Ready for GitHub!

**Your repository will look amazing!** 🚀
