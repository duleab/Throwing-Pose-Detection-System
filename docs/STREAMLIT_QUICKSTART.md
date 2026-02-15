# 🚀 Streamlit Web App - Quick Start Guide

## Running the Violence Detection Web Application

### Prerequisites

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

This will install:
- Core ML libraries (ultralytics, opencv-python, numpy)
- Tracking (deep-sort-realtime)
- Scientific computing (scikit-learn, scipy)
- Visualization (matplotlib)
- **Web interface (streamlit, Pillow)**

---

## 🎯 Launch the Application

### Method 1: Simple Launch
```bash
streamlit run streamlit_app.py
```

### Method 2: Custom Port
```bash
streamlit run streamlit_app.py --server.port 8080
```

### Method 3: Network Access (Allow external connections)
```bash
streamlit run streamlit_app.py --server.address 0.0.0.0
```

---

## 📱 Access the Application

After running the command, Streamlit will automatically:
- ✅ Open your default web browser
- ✅ Navigate to `http://localhost:8501`
- ✅ Display the Violence Detection interface

**Manual Access:**
- Local: `http://localhost:8501`
- Network: `http://YOUR_IP:8501`

---

## 🎨 Features

### 📸 Image Detection Tab

1. **Upload Image**
   - Click "Browse files" or drag & drop
   - Supported formats: JPG, JPEG, PNG, BMP

2. **Analyze**
   - Click "🔍 Analyze Image" button
   - Wait for processing (1-3 seconds)

3. **View Results**
   - ✅ Annotated image with bounding boxes
   - ⚠️ Violence alert (if detected)
   - 📊 Detection statistics
   - 🔍 Detailed per-person analysis
   - 📈 Feature visualization charts

### 🎥 Video Detection Tab

1. **Upload Video**
   - Click "Browse files" or drag & drop
   - Supported formats: MP4, AVI, MOV, MKV

2. **View Video Info**
   - Duration, FPS, frame count, resolution

3. **Analyze**
   - Click "🔍 Analyze Video" button
   - Watch real-time progress bar
   - Processing time depends on video length

4. **View Results**
   - ⚠️ Violence alert summary
   - 📊 Comprehensive statistics
   - ⏱️ Violence timeline visualization
   - 🖼️ Captured violent frames (up to 9)
   - 💾 Download annotated video

### ⚙️ Settings (Sidebar)

**Detection Parameters:**
- **Minimum Violence Frames** (3-15)
  - Default: 6 frames
  - Higher = fewer false positives
  - Lower = faster detection

- **Confidence Threshold** (0.5-0.95)
  - Default: 0.8 (80%)
  - Higher = more strict
  - Lower = more sensitive

---

## 💡 Usage Examples

### Example 1: Quick Image Check
```bash
# 1. Start app
streamlit run streamlit_app.py

# 2. Go to "Image Detection" tab
# 3. Upload test image
# 4. Click "Analyze"
# 5. View results instantly
```

### Example 2: Video Analysis
```bash
# 1. Start app
streamlit run streamlit_app.py

# 2. Go to "Video Detection" tab
# 3. Upload video file
# 4. Adjust settings if needed (sidebar)
# 5. Click "Analyze Video"
# 6. Wait for processing
# 7. Download annotated video
```

### Example 3: Batch Image Processing
```bash
# Process multiple images:
# 1. Upload first image → Analyze
# 2. Upload second image → Analyze
# 3. Repeat as needed
# Each result is displayed immediately
```

---

## 🎨 User Interface Overview

### Main Layout

```
┌─────────────────────────────────────────────────────────┐
│  🛡️ Violence Detection System                          │
│  AI-Powered Violence Detection for Images & Videos      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  [📸 Image Detection] [🎥 Video Detection] [📚 Docs]   │
│                                                          │
│  ┌──────────────────┐  ┌──────────────────┐            │
│  │  Upload Image    │  │  Results         │            │
│  │                  │  │                  │            │
│  │  [Browse files]  │  │  ✅ No Violence  │            │
│  │                  │  │                  │            │
│  │  [🔍 Analyze]    │  │  📊 Statistics   │            │
│  └──────────────────┘  └──────────────────┘            │
│                                                          │
└─────────────────────────────────────────────────────────┘

Sidebar:
┌──────────────┐
│ ⚙️ Settings  │
├──────────────┤
│ 🎯 Detection │
│   Parameters │
│              │
│ 📊 System    │
│   Info       │
│              │
│ ℹ️ About     │
└──────────────┘
```

---

## 🔧 Troubleshooting

### Issue: Models Not Loading

**Error:** "Error loading models"

**Solution:**
```bash
# Models download automatically on first run
# Ensure internet connection
# Wait for download to complete (~50MB)

# Manual download (if needed):
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8n-pose.pt')"
```

### Issue: Import Error

**Error:** "ModuleNotFoundError: No module named 'Violence_detection_modular'"

**Solution:**
```bash
# Ensure you're in the correct directory
cd "d:\Project\ICE Agent DETECTION\Violence detection system architecture"

# Check file exists
dir Violence_detection_modular.py

# Run from correct location
streamlit run streamlit_app.py
```

### Issue: Slow Processing

**Problem:** Video processing is very slow

**Solutions:**
1. **Use GPU** (if available)
   ```python
   # Models automatically use GPU if available
   # Check GPU: nvidia-smi
   ```

2. **Reduce Video Resolution**
   ```bash
   # Use video editing tool to resize
   # Recommended: 720p or lower
   ```

3. **Process Shorter Clips**
   ```bash
   # Split long videos into shorter segments
   # Process each segment separately
   ```

### Issue: Port Already in Use

**Error:** "Address already in use"

**Solution:**
```bash
# Use different port
streamlit run streamlit_app.py --server.port 8502

# Or kill existing process
# Windows:
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# Linux/Mac:
lsof -ti:8501 | xargs kill -9
```

---

## 📊 Performance Tips

### For Faster Processing

1. **GPU Acceleration**
   - Install CUDA-enabled PyTorch
   - Models automatically use GPU
   - 10-20x faster than CPU

2. **Optimize Settings**
   - Increase frame skip (process fewer frames)
   - Lower confidence threshold (faster decisions)
   - Reduce minimum violence frames

3. **Video Preprocessing**
   - Resize to 720p or lower
   - Reduce FPS (e.g., 15 FPS instead of 30)
   - Trim unnecessary parts

### For Better Accuracy

1. **Adjust Thresholds**
   - Increase confidence threshold (0.85-0.90)
   - Increase minimum violence frames (8-10)
   - Fine-tune based on your use case

2. **Quality Input**
   - Use high-resolution images/videos
   - Ensure good lighting
   - Avoid heavy compression

---

## 🎯 Advanced Usage

### Custom Configuration

Edit `streamlit_app.py` to customize:

```python
# Line 40-50: Adjust default settings
min_violence_frames = st.slider(
    "Minimum Violence Frames",
    min_value=3,
    max_value=15,
    value=6,  # Change default here
)

# Line 60-70: Modify model paths
model_loader = ModelLoader(
    'yolov8n-pose.pt',  # Change to 'yolov8s-pose.pt' for better accuracy
    'yolov8n.pt'        # Change to 'yolov8s.pt' for better accuracy
)
```

### Add Custom Features

```python
# Add to streamlit_app.py

# Example: Add email alerts
def send_alert(result):
    if result['has_violence']:
        # Send email notification
        send_email("Violence detected!", result)

# Example: Save to database
def log_detection(result):
    # Save to database
    db.insert(result)
```

---

## 📚 Documentation

### In-App Documentation
- Click **"📚 Documentation"** tab in the app
- Complete system overview
- How it works
- Feature explanations
- Troubleshooting guide

### External Documentation
- `Architecture.md` - System design
- `CODE_ANALYSIS_DETAILED.md` - Algorithm details
- `AI_EXPERT_ANALYSIS.md` - Mathematical foundations
- `Quickstart.md` - General usage guide

---

## 🚀 Deployment

### Local Network Access

```bash
# Allow access from other devices on your network
streamlit run streamlit_app.py --server.address 0.0.0.0

# Access from other devices:
# http://YOUR_COMPUTER_IP:8501
```

### Cloud Deployment

#### Streamlit Cloud (Free)
```bash
# 1. Push code to GitHub
git init
git add .
git commit -m "Violence detection app"
git push origin main

# 2. Go to share.streamlit.io
# 3. Connect GitHub repo
# 4. Deploy!
```

#### Docker Deployment
```dockerfile
# Create Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0"]
```

```bash
# Build and run
docker build -t violence-detector .
docker run -p 8501:8501 violence-detector
```

---

## 🎨 Customization

### Change Theme

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

### Modify Layout

Edit `streamlit_app.py`:

```python
# Change column ratios
col1, col2 = st.columns([2, 1])  # 2:1 ratio

# Add more tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📸 Image", "🎥 Video", "📊 Batch", "📚 Docs"
])

# Custom CSS
st.markdown("""
<style>
    .custom-class {
        /* Your styles */
    }
</style>
""", unsafe_allow_html=True)
```

---

## ✅ Quick Checklist

Before running:
- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] In correct directory
- [ ] `Violence_detection_modular.py` exists
- [ ] Internet connection (for first run)

To run:
- [ ] Open terminal/command prompt
- [ ] Navigate to project directory
- [ ] Run `streamlit run streamlit_app.py`
- [ ] Wait for browser to open
- [ ] Upload image or video
- [ ] Click analyze button

---

## 🎉 You're Ready!

```bash
# Start the app
streamlit run streamlit_app.py

# Open browser to: http://localhost:8501
# Upload an image or video
# Click "Analyze"
# View results!
```

**Enjoy your Violence Detection Web Application!** 🛡️

---

*For technical details, see AI_EXPERT_ANALYSIS.md*  
*For code explanations, see CODE_ANALYSIS_DETAILED.md*
