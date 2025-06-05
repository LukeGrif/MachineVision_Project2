# 🚗 Machine Vision Assignment: Speed Sign Detector

## 📘 Overview

This project presents a complete speed sign detection and recognition pipeline using image processing and machine learning techniques. The system:

- Detects speed signs using HSV color filtering  
- Refines candidate regions using morphological operations  
- Classifies the detected regions using a 1-Nearest Neighbour (1-NN) classifier  

It outputs both **visual bounding boxes** and **terminal logs** of detected signs.

**Contributors:**
- Luke Griffin  
- Patrick Crotty  
- Michael Cronin  
- Aaron Smith  
- Cullen Toal  

---

## 🧠 Background

Modern driver assistance systems rely on real-time speed sign detection. This project explores traditional computer vision and lightweight ML methods to implement such a system.

Key concepts used include:

- **HSV color space segmentation**  
- **Morphological image processing**  
- **1-NN classification using exemplar matching**  
- **Edge handling and fallback calibration for poor lighting**

> 📖 Includes discussion of RGB-HSV conversion, edge detection, and shape analysis.

![image](https://github.com/user-attachments/assets/abffc777-bbbe-4010-b705-b4651bac87ef)

**🖼️ HSV Color Model Diagram**

---

## 🧱 Architecture Overview

The project is divided into three Python modules:

### 1. `roidmds.py` – Region Proposer
- Converts RGB to HSV
- Applies hue, saturation, and value masks
- Refines results with morphological opening/closing
- Adds erosion/dilation for robustness to lighting conditions

---

### 2. `classifier.py` – 1-Nearest Neighbour Classifier
- Loads precomputed exemplar descriptors
- Preprocesses regions (grayscale, resize, normalize)
- Uses Euclidean distance to classify speed sign
- 
---

### 3. `SignSelector.py` – System Orchestrator
- Handles full detection pipeline
- Displays bounding boxes and labels on images
- Logs results to terminal

---

## 🔧 Testing & Calibration

Automated testing was implemented with:
- Image naming conventions used to parse expected results
- HSV and morphological thresholds tuned through trial
- Erosion masks added to minimize false positives
- Fallback logic added to catch signs in dark scenes

**Threshold Summary Table:**

| Parameter     | Value  |
|---------------|--------|
| Hue Min       | 0.949  |
| Hue Max       | 0.048  |
| Saturation    | 0.265  |
| Value         | 0.11   |

---

## 📊 Results

Tested on images including faded, skewed, or dimly lit signs. The algorithm accurately:
- Proposes valid regions
- Eliminates false positives
- Classifies signs such as 40, 50, 60, 80, 100, 120 km/h

![image](https://github.com/user-attachments/assets/661a5ab9-2038-448c-b170-ac918ab2ba86)

**🖼️ Input image example**  

![image](https://github.com/user-attachments/assets/e4cd787a-ebe4-4768-a5e2-530f40128c02)

**🖼️ HSV split image**  

![image](https://github.com/user-attachments/assets/b6cc7d95-8c84-44bf-87fa-2ea1b1b6e816)

**🖼️ Thresholded channels**  

![image](https://github.com/user-attachments/assets/1d170105-ab4e-4414-822e-3228b615ce51)

**🖼️ Combined binary mask**  

![image](https://github.com/user-attachments/assets/88aa0fd3-5eed-4f4c-8c2d-0396a4c94718)

**🖼️ Morphologically refined output**  

![image](https://github.com/user-attachments/assets/80115c2d-2cfb-4d40-a6b6-6dc51dbf2def)

**🖼️ Final detection output**  

![image](https://github.com/user-attachments/assets/59a1d0c6-e154-437c-83cc-225add0d875a)

**🖼️ Other detection examples**

---

## ✅ Conclusion

This project successfully developed a robust pipeline for speed sign detection using classic computer vision tools combined with basic machine learning:

- HSV thresholding isolates sign colors  
- Morphological operations reduce noise  
- 1-NN classification accurately identifies speed values  

The system is resilient to lighting variance, occlusion, and multiple sign types, making it well-suited for real-world extension.

---

## 📎 References

1. Stallkamp et al., *German Traffic Sign Recognition Benchmark*, IJCNN, 2011  
2. Krizhevsky et al., *ImageNet with Deep CNNs*, NIPS, 2012  
3. Lowe, *SIFT Keypoints*, IJCV, 2004  
4. Dalal & Triggs, *HOG Features*, CVPR, 2005  
5. Gonzalez & Woods, *Digital Image Processing*, 4th Ed.  
6. Burger & Burge, *Digital Image Processing: Core Algorithms*  
7. Flach, *Machine Learning: The Art and Science*, 2012  
8. Bishop, *Pattern Recognition and Machine Learning*, 2006  
