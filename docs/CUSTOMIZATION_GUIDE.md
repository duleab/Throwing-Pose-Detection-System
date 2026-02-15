# 🎨 Customization Guide - Bounding Box & Skeleton Colors

## ✅ New Features Added!

### **1. Color Customization** 🎨
- Choose custom colors for bounding boxes
- Choose custom colors for skeletons
- Choose custom colors for keypoints
- Separate colors for violent vs normal detections

### **2. Thickness Customization** 📏
- Adjust bounding box line thickness (1-10)
- Adjust skeleton line thickness (1-10)
- Adjust keypoint circle size (2-15)

---

## 🎨 How to Use

### **Step 1: Open Customization Panel**

1. Go to **"📸 Image Detection"** tab
2. Scroll down to **"🎨 Visualization Customization"**
3. Click on **"🎨 Color & Style Settings"** expander

### **Step 2: Choose Colors**

**For Violent Detections:**
- **Bounding Box Color** - Default: Red (#FF0000)
- **Skeleton Color** - Default: Red (#FF0000)
- **Keypoint Color** - Default: Blue (#0000FF)

**For Normal Detections:**
- **Bounding Box Color** - Default: Green (#00FF00)
- **Skeleton Color** - Default: Green (#00FF00)
- **Keypoint Color** - Default: Cyan (#00FFFF)

### **Step 3: Adjust Thickness**

**Line Thickness:**
- **Bounding Box** - Default: 3 (range: 1-10)
- **Skeleton Lines** - Default: 2 (range: 1-10)
- **Keypoint Size** - Default: 5 (range: 2-15)

### **Step 4: Analyze Image**

Click **"🔍 Analyze Image"** and your custom colors/thickness will be applied!

---

## 🎨 Color Picker Guide

### **How to Pick Colors:**

1. **Click on color box** - Opens color picker
2. **Choose from palette** - Quick color selection
3. **Or enter hex code** - Precise color control

### **Popular Color Schemes:**

#### **Scheme 1: Classic (Default)**
```
Violent:  Red (#FF0000)
Normal:   Green (#00FF00)
```

#### **Scheme 2: High Contrast**
```
Violent:  Yellow (#FFFF00)
Normal:   Blue (#0000FF)
```

#### **Scheme 3: Dark Mode**
```
Violent:  Orange (#FF8800)
Normal:   Cyan (#00FFFF)
```

#### **Scheme 4: Pastel**
```
Violent:  Pink (#FF69B4)
Normal:   Light Blue (#87CEEB)
```

#### **Scheme 5: Neon**
```
Violent:  Magenta (#FF00FF)
Normal:   Lime (#00FF00)
```

---

## 📏 Thickness Guide

### **Bounding Box Thickness:**

| Value | Appearance | Use Case |
|-------|------------|----------|
| 1-2 | Thin lines | High-res images, minimal overlay |
| 3-4 | Medium | **Recommended** - Good balance |
| 5-7 | Thick | Low-res images, emphasis |
| 8-10 | Very thick | Presentations, demos |

### **Skeleton Thickness:**

| Value | Appearance | Use Case |
|-------|------------|----------|
| 1 | Thin | Detailed pose analysis |
| 2-3 | Medium | **Recommended** - Clear visibility |
| 4-6 | Thick | Presentations |
| 7-10 | Very thick | Large displays |

### **Keypoint Size:**

| Value | Appearance | Use Case |
|-------|------------|----------|
| 2-3 | Small dots | Minimal overlay |
| 4-6 | Medium | **Recommended** - Clear points |
| 7-10 | Large | Emphasis on joints |
| 11-15 | Very large | Presentations |

---

## 🎨 Example Configurations

### **Configuration 1: Subtle**
```
Violent:
- Bounding Box: Dark Red (#8B0000), Thickness: 2
- Skeleton: Dark Red (#8B0000), Thickness: 1
- Keypoints: Maroon (#800000), Size: 3

Normal:
- Bounding Box: Dark Green (#006400), Thickness: 2
- Skeleton: Dark Green (#006400), Thickness: 1
- Keypoints: Forest Green (#228B22), Size: 3
```

### **Configuration 2: Bold**
```
Violent:
- Bounding Box: Bright Red (#FF0000), Thickness: 5
- Skeleton: Orange (#FF8800), Thickness: 4
- Keypoints: Yellow (#FFFF00), Size: 8

Normal:
- Bounding Box: Bright Green (#00FF00), Thickness: 5
- Skeleton: Cyan (#00FFFF), Thickness: 4
- Keypoints: Blue (#0000FF), Size: 8
```

### **Configuration 3: Professional**
```
Violent:
- Bounding Box: Red (#FF0000), Thickness: 3
- Skeleton: Red (#FF0000), Thickness: 2
- Keypoints: Blue (#0000FF), Size: 5

Normal:
- Bounding Box: Green (#00FF00), Thickness: 3
- Skeleton: Green (#00FF00), Thickness: 2
- Keypoints: Cyan (#00FFFF), Size: 5
```
*(This is the default configuration)*

---

## 🎯 Tips for Best Results

### **For Clear Visibility:**
1. Use **high contrast** colors
2. Use **medium thickness** (3-4 for boxes, 2-3 for skeleton)
3. Use **medium keypoint size** (4-6)

### **For Presentations:**
1. Use **bright, bold** colors
2. Use **thick lines** (5-7)
3. Use **large keypoints** (7-10)

### **For Analysis:**
1. Use **subtle colors**
2. Use **thin lines** (1-2)
3. Use **small keypoints** (2-4)

### **For Dark Images:**
1. Use **bright colors** (yellow, cyan, magenta)
2. Avoid dark colors (dark red, dark green)

### **For Light Images:**
1. Use **dark or saturated** colors
2. Avoid light colors (light blue, pink)

---

## 🖼️ Visual Examples

### **Default Settings:**
```
Violent Person:
├─ Red bounding box (thickness: 3)
├─ Red skeleton lines (thickness: 2)
└─ Blue keypoint dots (size: 5)

Normal Person:
├─ Green bounding box (thickness: 3)
├─ Green skeleton lines (thickness: 2)
└─ Cyan keypoint dots (size: 5)
```

### **Custom Example 1: High Visibility**
```
Violent Person:
├─ Yellow bounding box (thickness: 5)
├─ Orange skeleton lines (thickness: 4)
└─ Red keypoint dots (size: 8)

Normal Person:
├─ Cyan bounding box (thickness: 5)
├─ Blue skeleton lines (thickness: 4)
└─ Green keypoint dots (size: 8)
```

### **Custom Example 2: Minimal**
```
Violent Person:
├─ Dark red bounding box (thickness: 1)
├─ Dark red skeleton lines (thickness: 1)
└─ Maroon keypoint dots (size: 2)

Normal Person:
├─ Dark green bounding box (thickness: 1)
├─ Dark green skeleton lines (thickness: 1)
└─ Forest green keypoint dots (size: 2)
```

---

## 🎨 Color Psychology

### **Red (Violent):**
- **Meaning:** Danger, alert, violence
- **Effect:** Immediately draws attention
- **Best for:** Highlighting threats

### **Green (Normal):**
- **Meaning:** Safe, normal, OK
- **Effect:** Calming, reassuring
- **Best for:** Showing safe behavior

### **Alternative Colors:**

**For Violent:**
- **Orange:** Warning, caution
- **Yellow:** Alert, attention
- **Magenta:** High priority
- **Purple:** Unusual, concerning

**For Normal:**
- **Blue:** Calm, neutral
- **Cyan:** Clear, visible
- **White:** Clean, simple
- **Gray:** Neutral, background

---

## 🔧 Technical Details

### **Color Format:**
- **Input:** Hex color (#RRGGBB)
- **Conversion:** RGB → BGR (OpenCV format)
- **Example:** #FF0000 (Red) → (0, 0, 255) BGR

### **Thickness:**
- **Unit:** Pixels
- **Range:** 1-10 for lines, 2-15 for circles
- **Effect:** Higher = thicker/larger

### **Drawing Order:**
1. Skeleton lines (underneath)
2. Keypoint circles (on top of skeleton)
3. Bounding box (around everything)
4. Label text (on top)

---

## 💡 Pro Tips

### **1. Match Your Brand Colors**
```
If your company uses specific colors:
- Use brand colors for bounding boxes
- Keeps consistent visual identity
```

### **2. Accessibility**
```
Consider color-blind users:
- Use high contrast
- Combine color with thickness
- Red-green is common issue
- Try red-blue or yellow-blue instead
```

### **3. Save Screenshots**
```
Different settings for different purposes:
- Subtle for analysis
- Bold for presentations
- High contrast for reports
```

### **4. Test on Your Images**
```
What looks good depends on:
- Image brightness
- Background colors
- Image resolution
- Display device
```

---

## 🎉 Summary

### **What You Can Customize:**
- ✅ 6 different colors (violent & normal × 3 elements)
- ✅ 3 thickness settings (box, skeleton, keypoints)
- ✅ Real-time preview
- ✅ Easy color picker interface

### **Benefits:**
- 🎨 Match your brand/style
- 👁️ Improve visibility
- 📊 Better presentations
- 🔍 Clearer analysis
- ♿ Accessibility options

---

## 🚀 Quick Start

1. **Open app** → Image Detection tab
2. **Click** "🎨 Color & Style Settings"
3. **Pick colors** using color pickers
4. **Adjust sliders** for thickness
5. **Upload image** and analyze
6. **See your custom style!**

---

**Enjoy your fully customizable violence detection visualization!** 🎨

*The app will automatically reload with your new customization features.*
