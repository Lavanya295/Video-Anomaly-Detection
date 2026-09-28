# Video Anomaly Detection using Convolutional Autoencoders

A lightweight, unsupervised **video anomaly detection system** that identifies unusual events in CCTV footage using a Convolutional AutoEncoder (ConvAE).

The system learns the visual distribution of **normal pedestrian activity** and detects anomalous frames through reconstruction error. It is designed as a lightweight baseline for near-real-time surveillance applications where inference speed and low false-alarm rates are important.

## Overview

Manual monitoring of multiple CCTV feeds is expensive and fatigue-prone. This project explores an automated approach for identifying anomalous events in surveillance footage without requiring anomaly labels during training.

The model is trained **exclusively on normal video frames**. When presented with an unseen anomalous frame, the autoencoder typically reconstructs the unexpected content poorly, producing a higher pixel-level reconstruction error.

### Key Idea

> **Normal frame → accurate reconstruction → low MSE → Normal**

> **Anomalous frame → poor reconstruction → high MSE → Anomaly**

The system uses the **95th percentile of training reconstruction errors** as the anomaly threshold.

---

## Dataset

The project uses the **UCSD Pedestrian Dataset — Ped2**.

| Property          | Details                                           |
| ----------------- | ------------------------------------------------- |
| Dataset           | UCSD Pedestrian Ped2                              |
| Training clips    | 16                                                |
| Training frames   | 2,550 normal frames                               |
| Test clips        | 12                                                |
| Test frames       | 2,010                                             |
| Frame format      | Grayscale                                         |
| Input resolution  | 128 × 192                                         |
| Training paradigm | One-class / unsupervised                          |
| Anomaly threshold | 95th percentile of training reconstruction errors |
| Threshold value   | `0.000734`                                        |

The test set contains anomalous events such as **bicyclists, skaters, and carts** appearing in the pedestrian walkway.

---

## System Architecture

```text
                 CCTV / Video Stream
                         │
                         ▼
                  Frame Ingestion
                         │
                         ▼
              Grayscale + Resize
                  128 × 192
                         │
                         ▼
                 Normalization
                    [0, 1]
                         │
                         ▼
              ┌──────────────────┐
              │   ConvAutoEncoder│
              │                  │
              │ Encoder → Latent │
              │        → Decoder │
              └──────────────────┘
                         │
                         ▼
               Reconstructed Frame
                         │
                         ▼
              Pixel-wise MSE Error
                         │
                         ▼
              ┌──────────────────┐
              │ MSE > Threshold? │
              └──────────────────┘
                   │          │
                  YES         NO
                   │           │
                   ▼           ▼
               ANOMALY       NORMAL
```

---

## Model Architecture

The baseline model is a lightweight **Convolutional AutoEncoder**.

### Encoder

* 4 convolutional layers
* Stride-2 downsampling
* Batch Normalization
* LeakyReLU activations
* Bottleneck representation: `16 × 24 × 128`

### Decoder

The decoder mirrors the encoder architecture and reconstructs the original `128 × 192` frame.

### Model Size

* Trainable parameters: **257,761**
* Checkpoint size: approximately **2 MB**

### Training Objective

The model minimizes pixel-wise Mean Squared Error:

```text
MSE = mean((x - x̂)²)
```

where:

* `x` = original input frame
* `x̂` = reconstructed frame

No anomaly labels are required during training.

---

## Training

The model was trained using:

| Parameter                          | Value                      |
| ---------------------------------- | -------------------------- |
| GPU                                | NVIDIA RTX 4050 Laptop GPU |
| Epochs                             | 30                         |
| Batch size                         | 64                         |
| Optimizer                          | Adam                       |
| Learning-rate scheduler            | CosineAnnealingLR          |
| Training time                      | ~143 seconds               |
| Final training reconstruction loss | ~0.000367                  |

The reconstruction loss decreased rapidly during training and stabilized by the end of the 30 epochs.

---

## Anomaly Detection

After training, reconstruction errors from the normal training frames are used to establish an operating threshold.

```text
Training reconstruction errors
            │
            ▼
     95th percentile
            │
            ▼
   Threshold = 0.000734
```

For each incoming frame:

```python
if reconstruction_mse > threshold:
    prediction = "ANOMALY"
else:
    prediction = "NORMAL"
```

The underlying assumption is that the model learns to reconstruct normal pedestrian scenes well while producing larger reconstruction errors for previously unseen visual patterns.

---

## Visual Explanation

For a normal frame, the reconstructed image is visually similar to the original, producing relatively uniform and low reconstruction error.

For an anomalous frame containing an unexpected object such as a bicycle, the reconstruction error increases around the anomalous region.

```text
Original Frame
      │
      ▼
ConvAutoEncoder
      │
      ▼
Reconstructed Frame
      │
      ▼
Absolute / Pixel-wise Error
      │
      ▼
High-error regions → Anomaly
```

---

## Results

Evaluation was performed on the UCSD Ped2 test set.

| Metric                |             Result |
| --------------------- | -----------------: |
| Precision             |          **0.927** |
| Recall                |          **0.217** |
| F1 Score              |          **0.352** |
| AUC-ROC               |         **0.6428** |
| GPU inference latency | **< 5 ms / frame** |

### Confusion Matrix

|                    | Actual Positive | Actual Negative |
| ------------------ | --------------: | --------------: |
| Predicted Positive |             358 |              28 |
| Predicted Negative |            1290 |             334 |

The system achieves high precision, but recall is substantially lower. This reflects the conservative anomaly threshold and the limitations of detecting temporal events using a frame-based architecture.

---

## Performance

The model was designed as a lightweight baseline rather than a computationally expensive state-of-the-art architecture.

### Inference

* GPU inference: **< 5 ms/frame**
* Real-time frame budget at 30 FPS: approximately **33 ms/frame**
* Model checkpoint: approximately **2 MB**

This leaves room for additional processing and serving overhead in a potential streaming deployment.

---

## Proposed Deployment Architecture

A production-oriented deployment architecture was designed around edge inference and event streaming:

```text
             CCTV Cameras
                  │
                  ▼
          Edge Inference
      Jetson Nano / Raspberry Pi
                  │
                  ▼
             FastAPI
          Serving Layer
                  │
                  ▼
            Kafka Bus
         Alert / Event Stream
                  │
          ┌───────┴────────┐
          ▼                ▼
      Grafana          Analyst
     Monitoring        Verification
```

### Components

**Edge Inference**

Models can run close to the camera to reduce network latency and bandwidth requirements.

**FastAPI**

Provides a lightweight serving layer for receiving frames and returning anomaly predictions.

**Kafka**

Provides an event-streaming layer for decoupling inference from downstream consumers and alert processing.

**Grafana**

Provides real-time monitoring of detections, system health, and stream statistics.

**Analyst Feedback**

Human analysts can confirm or reject alerts, creating a potential feedback loop for future system improvements.

---

## Design Decisions

### Why an AutoEncoder?

The project focuses on a **one-class anomaly detection** setting where normal examples are available but anomalous examples may be limited or unavailable during training.

A reconstruction-based approach allows the model to learn the normal visual distribution without requiring anomaly labels.

### Why Start with ConvAE?

The goal was to establish a lightweight and interpretable baseline before introducing more computationally expensive approaches such as PatchCore or DRAEM.

This provides a useful reference point for evaluating whether more complex architectures justify their additional computational cost.

### Why Prioritize Precision?

In surveillance systems, excessive false alarms can lead to **alert fatigue** and reduce analyst trust.

The initial operating point therefore prioritizes precision, while recognizing that this comes at the cost of lower recall.

---

## Limitations

This implementation has several important limitations:

### 1. Frame-Based Detection

The model processes individual frames rather than temporal sequences.

As a result, it may miss anomalies that are better characterized by **motion or temporal context**.

### 2. Conservative Threshold

The 95th-percentile threshold produces a relatively conservative operating point, contributing to high precision but lower recall.

### 3. Dataset Scope

Evaluation is based on the UCSD Ped2 dataset, so performance may not directly generalize to different camera environments, viewpoints, lighting conditions, or anomaly types.

### 4. Baseline Performance

The achieved AUC-ROC of **0.6428** leaves significant room for improvement compared with stronger anomaly-detection architectures.

The project therefore treats the ConvAE as a **baseline rather than a final production model**.

---

## Future Improvements

The next iteration would focus on improving temporal and spatial representation.

### Temporal Modeling

Replace single-frame processing with short video clips using a **3D Convolutional AutoEncoder**.

```text
Single Frame
     ↓
2D ConvAE

vs.

8-Frame Clip
     ↓
3D ConvAE
     ↓
Temporal + Spatial Features
```

This would allow the model to capture motion-based anomalies.

### Patch-Level Feature Extraction

Explore **PatchCore** for stronger localized feature representations and improved anomaly detection.

### Optical Flow

Add optical-flow information as an additional input channel to explicitly represent motion.

### Inference Optimization

Potential deployment optimizations include:

* FP16 inference
* TensorRT export
* INT8 quantization
* Dynamic batching
* Scene-statistics caching

---

## Tech Stack

**Machine Learning**

* Python
* PyTorch
* Convolutional AutoEncoder
* Computer Vision
* Unsupervised / One-Class Anomaly Detection

**Data & Evaluation**

* UCSD Pedestrian Dataset — Ped2
* Mean Squared Error
* Precision / Recall / F1
* ROC-AUC
* Pixel-level evaluation

**Deployment Architecture**

* FastAPI
* Kafka
* Grafana
* NVIDIA Jetson / Edge Devices

---

## Project Highlights

* Built a lightweight **unsupervised video anomaly detection baseline**
* Trained exclusively on normal pedestrian frames
* Achieved **0.927 precision**
* Achieved **0.6428 ROC-AUC**
* Achieved **<5 ms GPU inference per frame**
* Used a **257K-parameter** convolutional autoencoder
* Designed an edge-to-streaming deployment architecture using **FastAPI + Kafka + Grafana**
* Identified the primary limitation of the baseline as insufficient temporal modeling
* Proposed 3D ConvAE, PatchCore, and optical-flow extensions

---

## Repository Structure

```text
video-anomaly-detection/
│
├── README.md
├── notebooks/
│   └── video_anomaly_detection.ipynb
│
├── src/
│   ├── model.py
│   ├── preprocessing.py
│   ├── inference.py
│   └── evaluation.py
│
├── configs/
│   └── config.yaml
│
├── results/
│   ├── anomaly_scores/
│   ├── visualizations/
│   └── metrics/
│
├── models/
│   └── convae.pth
│
└── requirements.txt
```

> Update the repository structure above to match the actual files you upload to GitHub.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/video-anomaly-detection.git
cd video-anomaly-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

Download and organize the **UCSD Ped2** dataset according to the project's preprocessing configuration.

### 4. Train the model

```bash
python src/train.py
```

### 5. Run inference

```bash
python src/inference.py
```

### 6. Evaluate

```bash
python src/evaluation.py
```

---

## Key Takeaway

This project demonstrates how a **small, unsupervised convolutional autoencoder** can provide a fast baseline for video anomaly detection without requiring anomaly labels during training.

The most important engineering trade-off observed was:

```text
Higher precision
      ↑
      │
Conservative threshold
      │
      ↓
Lower recall
```

The results motivate a second iteration focused on **temporal modeling, optical flow, and stronger feature representations** rather than simply increasing model size.

---


