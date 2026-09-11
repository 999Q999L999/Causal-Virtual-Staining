# Towards Generalizable and Robust Virtual Staining via Causal Intervention

Official implementation of **"Towards Generalizable and Robust Virtual Staining via Causal Intervention"**.

> **Paper:** Towards Generalizable and Robust Virtual Staining via Causal Intervention  
> **Authors:** Weiping Lin, Baoshun Wang, Yihuang Hu, Runchen Zhu, Liansheng Wang  
> **Affiliation:** Xiamen University, China

---

## Overview

Virtual staining is an image-to-image translation technique that generates digitally stained pathology images from unstained or differently processed tissue images. Although recent virtual staining methods can achieve high-quality results under specific experimental settings, their performance may degrade when applied to unseen datasets, different staining conditions, or other real-world variations.

A key challenge is that virtual staining models may learn **spurious correlations** from diagnosis-irrelevant factors, such as staining variability, scanner characteristics, demographic biases, and other dataset-specific variations. These factors can cause the model to rely on non-causal information and consequently reduce its generalization and robustness.

In this work, we propose a **causal intervention-based training framework** for virtual staining. From a causal perspective, pathological image features are considered to contain both diagnosis-relevant components and diagnosis-irrelevant components. Our framework introduces a unified strategy to identify diagnosis-irrelevant features through a **confounder space classifier**, and then performs causal intervention during virtual staining training to suppress these spurious features.

The proposed framework is **model-agnostic** and can be integrated into existing virtual staining models in a plug-and-play manner.

We evaluate the proposed method on five datasets covering two clinically relevant virtual staining tasks:

- **H&E-to-IHC**
- **FFPE-to-H&E**

Experiments demonstrate improved virtual staining quality, generalization to unseen datasets, and robustness to content-preserving perturbations.

---

## Method

### Problem Formulation

Let

- $S$ denote the source image domain,
- $T$ denote the target staining domain,
- $X$ denote diagnosis-relevant information,
- $C$ denote diagnosis-irrelevant information.

A conventional virtual staining model may learn a biased mapping:

$$
G : S(X,C) \rightarrow T.
$$

The goal of our framework is to learn a more causal mapping that focuses on diagnosis-relevant information:

$$
G : X \rightarrow T.
$$

Instead of explicitly modeling every possible type of nuisance variation, we introduce the concept of a **confounder space**.

Images with similar diagnosis-irrelevant characteristics are considered to belong to the same confounder space. Depending on the available metadata, confounder spaces can be defined according to factors such as staining characteristics, acquisition site, scanning equipment, or other shared processing conditions.

In the worst case, when explicit nuisance annotations are unavailable, the WSI identity can be used to define the confounder space. Images originating from the same WSI are treated as belonging to the same confounder space.

### Confounder Space Classifier

We construct positive and negative image pairs according to confounder-space membership.

- Images from the same confounder space form positive pairs.
- Images from different confounder spaces form negative pairs.
- Additional color transformations are used to increase the diversity of training pairs.

A pairwise classifier is trained to estimate whether two target images belong to the same confounder space.

The classifier consists of an image encoder and a similarity scorer and produces similarity and dissimilarity scores for image pairs.

This classifier provides a mechanism to capture diagnosis-irrelevant information implicitly rather than requiring explicit annotations of every nuisance factor.

### Causal Intervention

After the confounder space classifier has been trained, it is frozen and incorporated into the virtual staining training process.

For an input source image $s_i$, the virtual staining model generates a target image:

$$
\hat{t}_i = G(s_i).
$$

The generated images are then evaluated by the frozen confounder space classifier. The virtual staining model is optimized so that the generated outputs become less distinguishable according to their confounder-space information.

In this way, the generated images are encouraged to suppress diagnosis-irrelevant features while preserving the information required for the staining transformation.

To improve computational efficiency and pair diversity, we further introduce a **queue dictionary** that stores generated target images and dynamically updates its contents during training.

---

## Framework

The overall training procedure consists of two main stages:

### Stage 1: Confounder Space Classifier

1. Construct image pairs according to confounder-space membership.
2. Apply color transformations to improve pair diversity.
3. Train a pairwise confounder-space classifier.
4. Learn representations that capture diagnosis-irrelevant information.

### Stage 2: Causal Intervention

1. Train the virtual staining model with its original objective.
2. After the warm-up stage, freeze the confounder space classifier.
3. Generate target images using the virtual staining model.
4. Compare generated images using the frozen confounder classifier.
5. Optimize the virtual staining model to suppress confounder-related information.
6. Use a queue dictionary to improve computational efficiency and pair diversity.

The resulting framework can be integrated with different supervised virtual staining architectures.

---

## Supported Tasks

We evaluate the framework on two virtual staining tasks.

### H&E-to-IHC

The H&E-to-IHC experiments focus on HER2 staining and scoring.

The datasets include:

- **BCI**
- **MIST-HER2**
- **self-HER2**

The self-HER2 dataset contains 3,206 accurately aligned H&E/IHC image pairs from 30 pairs of whole-slide images and covers the HER2 scoring categories:

- 0
- 1+
- 2+
- 3+

Image patches are extracted at 20× magnification with a resolution of $1024 \times 1024$ pixels.

### FFPE-to-H&E

The FFPE-to-H&E experiments include:

- **self-FFPE1**
- **self-FFPE2**

The two datasets were independently collected following nearly identical processing protocols, with different source institutions. FFPE and H&E images are strictly registered, and image patches are extracted at 40× magnification.

---

## Datasets

The datasets used in the experiments are summarized below.

| Dataset | Task | Train | Test | Public |
| --- | --- | ---: | ---: | :---: |
| BCI | H&E-to-IHC | - | 977 | ✓ |
| MIST-HER2 | H&E-to-IHC | 4642 | 998 | ✓ |
| self-HER2 | H&E-to-IHC | 2632 | 574 | ✗ |
| self-FFPE1 | FFPE-to-H&E | 4232 | 1075 | ✗ |
| self-FFPE2 | FFPE-to-H&E | - | 1398 | ✗ |

The train/test numbers above describe how the datasets are used in our experiments and do not necessarily indicate the original dataset partitioning.

### Data Availability

The public datasets should be downloaded from their respective publications.

The private datasets (`self-HER2`, `self-FFPE1`, and `self-FFPE2`) are not distributed with this repository.

Please make sure that you have the corresponding permissions and licenses before using any dataset.

---

## Baseline Models

We use two representative supervised virtual staining models as the base architectures:

- **Pix2Pix**
- **Pix2PixHD**

The proposed causal intervention framework is designed to be model-agnostic and can be incorporated into existing virtual staining models.

The experiments compare:

- Baseline model
- Baseline model + our causal intervention framework

---

## Evaluation

We evaluate the generated images using two groups of metrics.

### Image Quality Metrics

The following metrics measure pixel-level fidelity:

- **PSNR** ↑
- **SSIM** ↑
- **MS-SSIM** ↑

Higher values indicate better performance.

### Feature-Based Metrics

The following metrics evaluate perceptual and semantic differences using deep feature representations:

- **FID** ↓
- **KID** ↓
- **DISTS** ↓

Lower values indicate better performance.

---

## Experimental Settings

The implementation is based on **PyTorch 2.0.1**.

The experiments were conducted on workstations equipped with:

- 8 × NVIDIA RTX 3090
- 24 GB GPU memory per GPU

For the confounder-space classifier, we use:

- **Backbone:** ResNet-18
- **Training pairs:** 50,000 image pairs
- **Dictionary length:** approximately 5% of the training set
- **Warm-up:** 60 epochs
- **Causal intervention:** activated after the warm-up stage

The loss weight for suppressing diagnosis-irrelevant features is set according to the experimental configuration described in the paper.

---

## Repository Structure

The repository is organized as follows:

```text
Causal-Virtual-Staining/
│
├── data/
│   └── ...
│
├── models/
│   └── ...
│
├── options/
│   └── ...
│
├── scripts/
│   └── ...
│
├── util/
│   └── ...
│
├── docs/
│   └── ...
│
├── imgs/
│   └── ...
│
├── train.py
├── test.py
├── environment.yml
├── LICENSE
└── README.md
