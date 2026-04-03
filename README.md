# Self-Supervised Monocular Depth Estimation

## Overview

This project implements a simplified version of a **self-supervised monocular depth estimation** model from scratch.

The goal is to learn depth from a single image **without using ground truth depth labels**, by leveraging reconstruction-based loss functions.

---

## Objectives

* Implement depth estimation from scratch
* Understand self-supervised learning concepts
* Validate training pipeline using a toy dataset
* Compare with official implementation

---

## Methodology

The pipeline consists of:

1. **Depth Network**

   * ResNet18-based encoder
   * Outputs a depth map

2. **Pose Network**

   * Takes two images as input
   * Predicts relative camera motion

3. **Loss Functions**

   * Photometric Loss (L1)
   * Smoothness Loss

Total Loss:

```
L = L_photo + 0.1 * L_smooth
```

---

## Dataset

Due to computational and time constraints, a **toy dataset** was used.

* Random tensors simulate input images
* No real dataset (e.g., KITTI) used
* Used to validate:

  * Model architecture
  * Training pipeline
  * Loss computation

---

## Results

* Training performed for 2 epochs

* Loss values:

  * Epoch 0: ~1.14
  * Epoch 1: ~1.14

* A sample depth map was generated and saved in:

```
results/depth.png
```

---

## Project Structure

```
paper-implementation-Assignment3/
│
├── src/
│   ├── model.py
│   ├── loss.py
│   └── train.py
│
├── data/
├── results/
│   ├── depth.png
│   └── loss.txt
│
├── report/
│   └── Report_Assignment3.pdf
│
├── README.md
└── requirements.txt
```

---

## Comparison with Official Implementation

| Feature     | Official  | This Work |
| ----------- | --------- | --------- |
| Dataset     | KITTI     | Synthetic |
| Warping     | Yes       | No        |
| Multi-scale | Yes       | No        |
| Loss        | SSIM + L1 | L1        |
| Geometry    | Yes       | No        |

---

## Limitations

* No real dataset used
* No geometric modeling (camera intrinsics)
* No image warping (view synthesis)
* Simplified loss function

---

## Conclusion

This project successfully demonstrates the core idea of **self-supervised depth estimation**, showing that depth can be learned without labeled data using reconstruction-based loss.

---

## Report

The detailed report is available in:

```
report/Report_Assignment3.pdf
```

---

## Author

**Manvi Sheth**
Roll No: 24B0015
Group I
