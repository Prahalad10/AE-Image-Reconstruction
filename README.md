# 🧠 Image Denoising using Convolutional Autoencoders

This project demonstrates the use of **deep learning-based autoencoders** for **image denoising and reconstruction**.  
Two convolutional autoencoder models were developed and trained on **MNIST** and **CIFAR-10** datasets using **TensorFlow/Keras**.

---

## 📁 Project Overview

Image denoising is an important image restoration task that aims to remove noise while preserving meaningful features.  
This project implements and compares **two convolutional autoencoders** to reconstruct clean images from their noisy counterparts.

**Models Implemented:**
1. **Model 1:** MNIST Denoising Autoencoder (Grayscale digits)
2. **Model 2:** CIFAR-10 Denoising Autoencoder (Color images)

---

## ⚙️ Technologies Used
- **Python 3.x**
- **TensorFlow / Keras**
- **NumPy**
- **Matplotlib**
- **scikit-image**
- **OpenCV**

---

## 🧩 Model 1 – MNIST Denoising Autoencoder

### **Preprocessing**
- Input: 28×28 grayscale digit images.
- Normalized to [0,1] and reshaped to (28,28,1).
- Gaussian noise (std=0.5) added for denoising training.

### **Architecture**
- **Encoder:** Stacked Conv2D layers (32–64–128 filters) + MaxPooling.
- **Bottleneck:** Compact latent feature representation.
- **Decoder:** UpSampling + Conv2D layers to reconstruct clean images.
- **Activation:** ReLU and final Sigmoid.
- **Loss:** Binary Crossentropy, optimized with Adam.

### **Results**
- Final Validation MAE ≈ 0.033
- PSNR and SSIM show significant improvement over noisy inputs.
- Output images clearly restored digit structure with minimal blurring.

---

## 🌈 Model 2 – CIFAR-10 Denoising Autoencoder

### **Preprocessing**
- Input: 32×32 RGB images across 10 classes.
- Normalized to [0,1]; Gaussian noise (std=0.3) added to inputs.

### **Architecture**
- Similar convolutional autoencoder design adapted for RGB images.
- Batch Normalization layers added for stability.
- Symmetric encoder–decoder with ReLU activations.

### **Results**
- Final Validation MAE ≈ 0.059
- Slightly higher reconstruction error due to texture and color complexity.
- Successfully denoised noisy color images while retaining key features.

---

## 📊 Comparative Analysis

| Model | Dataset | Validation MAE | PSNR Gain | SSIM Gain | Notes |
|:------|:--------|:---------------|:-----------|:-----------|:------|
| **1** | MNIST | **0.0327** | High | High | Simpler grayscale data enables cleaner reconstruction |
| **2** | CIFAR-10 | **0.0594** | Moderate | Moderate | More complex textures and colors increase difficulty |

**Conclusion:**  
The MNIST model achieved superior quantitative results due to its simpler image domain, while the CIFAR-10 model validated the autoencoder’s ability to generalize to complex RGB images.

---

## 📈 Visual Results
Both models include visualizations for:
- Original vs Noisy vs Denoised images
- Training/Validation loss curves
- PSNR and SSIM metric distributions

---

