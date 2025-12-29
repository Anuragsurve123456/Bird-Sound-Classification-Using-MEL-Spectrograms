# Bird-Sound-Classification-Using-MEL-Spectrograms

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![EfficientNet](https://img.shields.io/badge/Backbone-EfficientNet--B0-purple)
![Task](https://img.shields.io/badge/Task-Multi--label%20Audio%20Classification-orange)
![Metric](https://img.shields.io/badge/Metric-Macro%20ROC--AUC-success)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

Deep learning pipeline for **BirdCLEF 2025** style bird-call classification using **mel-spectrograms** and an **EfficientNet-B0** CNN backbone. The goal is to **replicate and understand** a strong benchmark solution: preprocessing choices, augmentation, training strategy, and evaluation under real-world noise + class imbalance. :contentReference[oaicite:0]{index=0}

---

## Team

- Wilona Nguyen  
- Wali Siddiqui  
- Anurag Surve  
**Supervised by:** Amir Jafari :contentReference[oaicite:1]{index=1}

---

## Project Overview

Bird species classification from audio supports ecological monitoring, but manual labeling does not scale. This project uses **mel spectrogram inputs** and a **PyTorch CNN pipeline** built around **EfficientNet-B0**, trained for **multi-label classification** with severe class imbalance and noisy field recordings. :contentReference[oaicite:2]{index=2}

---

## Dataset

**Source:** Kaggle — BirdCLEF 2025 competition dataset  
- 4 top-level classes: **Aves, Mammalia, Amphibia, Insecta**
- 206 species sub-classes
- 28,564 audio recordings
- Sources include Xeno-Canto (XC), iNaturalist (iNat), and Colombian Sound Archive (CSA)
- Audio standardized to **32 kHz** and **.ogg** :contentReference[oaicite:3]{index=3}

Folders (as described in the report):
- `train_audio/`: labeled clips (variable length), primary + optional secondary labels
- `train_soundscapes/`: unlabeled 1-minute field recordings
- `test_soundscapes/`: empty by default in Kaggle; for local inference this project **manually populated** it and created **5713 segments** by splitting into **5-second windows** :contentReference[oaicite:4]{index=4}

---

## Model & Training Summary

### Architecture
- **EfficientNet-B0** backbone (timm + PyTorch)
- Modified head for multi-label output (sigmoid probabilities)
- Uses adaptive pooling + linear classifier head :contentReference[oaicite:5]{index=5}

### Input Representation
- Mel-spectrograms generated from audio (or loaded from precomputed `.npy`)
- Normalized to **[0,1]** and resized to **256×256**
- Uses **128 mel bins** and then resized to target image shape :contentReference[oaicite:6]{index=6}

### Loss / Optimizer / Scheduler
- **BCEWithLogitsLoss** (multi-label)
- **AdamW**, LR = **5e-4**, weight decay ≈ **1e-5**
- **CosineAnnealingLR**, min LR = **1e-6**, `T_max = epochs` :contentReference[oaicite:7]{index=7}

### Augmentations / Regularization
- **Mixup** (α = 0.5)
- Spectrogram augmentations including time stretch, pitch shift, volume adjustment
- Dropout + stochastic depth (drop path) inside EfficientNet backbone :contentReference[oaicite:8]{index=8}

### Experimental Setup
- 5-second centered crops (crop or zero-pad)
- Training epochs: **10**
- Batch size: **32**
- 5-fold stratified cross-validation :contentReference[oaicite:9]{index=9}

---

## Results

Primary metric: **Macro-averaged ROC-AUC (excluding empty classes)**.  
Mean ROC-AUC across 5 folds: **0.9476** :contentReference[oaicite:10]{index=10}

| Fold | ROC-AUC |
|------|--------:|
| 0    | 0.9451 |
| 1    | 0.9513 |
| 2    | 0.9438 |
| 3    | 0.9475 |
| 4    | 0.9501 |
| **Mean** | **0.9476** |

The metric skips classes that do not appear in a validation fold (no positives), which avoids undefined AUC and keeps evaluation fair for rare species. :contentReference[oaicite:11]{index=11}

---

## Streamlit Demo (Project Feature)

A Streamlit app is included for:
- Uploading audio
- Generating spectrogram on-the-fly
- Producing top-k species predictions (top-5 in the report discussion) :contentReference[oaicite:12]{index=12}

---

## Example Inference (Outside Dataset)

The report demonstrates inference on a sample audio clip (“sample_01.mp3”) outside the dataset:
- Waveform shows distinct call syllables amid noise
- Spectrogram shows energy concentrated around 2–4 kHz
- Model top prediction matched the field ID (Yellow-chinned Spinetail at ~50%) :contentReference[oaicite:13]{index=13}

---

## Key Challenges, Strengths, Limitations

### Challenges
- Severe class imbalance
- Variable recording lengths + noisy environments
- Overlapping signals and corrupted files :contentReference[oaicite:14]{index=14}

### Strengths
- Supports both precomputed and on-the-fly spectrogram generation
- Stratified k-fold + skip-empty-class macro-AUC gives robust evaluation :contentReference[oaicite:15]{index=15}

### Limitations / Future Work
- Fixed 5s window may miss calls near clip boundaries → sliding window voting
- Add feature diversity (MFCCs, temporal attention modules)
- Model compression (pruning/quantization) for edge deployment :contentReference[oaicite:16]{index=16}

---

## 🚀 Getting Started

### 1) Clone the Repository
```bash
git clone https://github.com/waliahmed24/Final_Project_Group6.git
```

### 2. Navigate to The Project Directory

```bash
cd Final_Project_Group6
```

### 3. Download The Dataset

```bash
wget -O Data.zip "https://www.dropbox.com/scl/fi/8jd9okreb87ojim7p513u/Data.zip?rlkey=islyrmkgagvnoxd1rdyolcslv&st=w23hljqi&dl=1"
```

### 4. Unzip The Dataset

```bash
unzip Data.zip
```

### 5. Navigate to the Code Directory

```bash
cd Code
```

### 6. Create a Virtual Environment and Install Dependencies

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

### 7. Run the Training Script

```bash
python3 main.py
```


This would run training and validation and once done will save the best model based on the validation set accuracy.


### Refer to the project directory structure below within the Code folder:

<img width="490" alt="Screenshot 2025-05-01 at 3 16 48 PM" src="https://github.com/user-attachments/assets/83444649-47a8-4d64-be02-6203ad3d48cd" />














