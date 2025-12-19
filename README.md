# Audio Regression Project with Multi-Modal Deep Learning

A comprehensive audio regression solution combining traditional feature engineering, deep learning embeddings, and ensemble modeling with multi-modal LLM fine-tuning.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset Structure](#dataset-structure)
- [Installation](#installation)
- [Architecture](#architecture)
- [Usage](#usage)
- [Feature Extraction](#feature-extraction)
- [Model Training](#model-training)
- [Results](#results)
- [Project Structure](#project-structure)

## 🎯 Overview

This project implements an advanced audio regression pipeline that predicts continuous values from audio files. It combines:

- **Ultra-Advanced Audio Features**: 160+ handcrafted features
- **Deep Learning Embeddings**: Wav2Vec2, HuBERT, WavLM
- **Multi-Modal Architecture**: Audio + Text description fusion
- **Ensemble Learning**: XGBoost, LightGBM, CatBoost, RF, ET
- **LoRA Fine-tuning**: Efficient parameter-efficient training

## ✨ Features

### Audio Feature Extraction

#### 1. **Ultra-Advanced Features (160 dimensions)**
- **Basic**: Duration (1 feature)
- **Energy**: RMS statistics and temporal dynamics (14 features)
- **Zero-Crossing & Silence**: ZCR analysis and pause detection (10 features)
- **Spectral**: Centroid, bandwidth, rolloff, contrast, flatness, flux (35 features)
- **Pitch & Voice**: F0 analysis, jitter, shimmer, HNR (15 features)
- **MFCCs**: Mean and delta features (24 features)
- **Mel Spectrogram**: Energy distribution and temporal features (15 features)
- **Chroma**: STFT and CQT chroma features (16 features)
- **Rhythm & Tempo**: Onset strength, beat tracking, tempogram (8 features)
- **Harmonic/Percussive**: Separation and ratio analysis (12 features)
- **Audio Quality**: Crest factor, dynamic range, SNR, spectral bands (10 features)

#### 2. **GPU-Accelerated Features (200 dimensions)**
- Optimized feature extraction using PyTorch and torchaudio
- Real-time processing on CUDA-enabled GPUs
- Includes spectral, MFCC, pitch, mel, rhythm, and quality features

#### 3. **Deep Learning Embeddings**
- **Wav2Vec2**: Self-supervised audio representations
- **HuBERT**: Hidden-Unit BERT for speech
- **WavLM**: WavLM for robust audio understanding

#### 4. **Metadata Features (9 dimensions)**
- File size, duration, sample rate
- Amplitude statistics
- Peak characteristics

### Data Augmentation

- **Time Stretching**: Random rate variation (0.95-1.05x)
- **Pitch Shifting**: Random pitch shift (-2 to +2 semitones)
- **Noise Addition**: Gaussian noise injection (σ=0.005)

### Test Time Augmentation (TTA)

- Multiple augmented predictions averaged
- Configurable number of augmentations (default: 3)
- Improves model robustness

## 📁 Dataset Structure

```
dataset/
├── csvs/
│   ├── train.csv          # Training labels (filename, label)
│   └── test.csv           # Test filenames
└── audios/
    ├── train/             # Training audio files (.wav)
    └── test/              # Test audio files (.wav)
```

### Data Format

**train.csv**:
```
filename,label
audio_173,3.0
audio_138,3.0
audio_127,2.0
```

**test.csv**:
```
filename
audio_141
audio_114
audio_17
```

## 🔧 Installation

### Requirements

```bash
# Core dependencies
pip install torch torchvision torchaudio
pip install transformers peft
pip install librosa soundfile
pip install numpy pandas scikit-learn scipy
pip install xgboost lightgbm catboost
pip install tqdm joblib
pip install warnings
```

### Kaggle Environment

This notebook is optimized for Kaggle with GPU acceleration:
- Tesla P100 or T4 GPU recommended
- 16GB+ RAM
- Internet access for model downloads

## 🏗️ Architecture

### Multi-Modal Regression Model

```
┌─────────────────┐     ┌──────────────────┐
│   Audio Input   │     │ Text Description │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         ▼                       ▼
  ┌─────────────┐         ┌─────────────┐
  │  Wav2Vec2   │         │ DistilBERT  │
  │  Extractor  │         │   Encoder   │
  └──────┬──────┘         └──────┬──────┘
         │                       │
         ▼                       ▼
   ┌──────────┐           ┌──────────┐
   │ Audio    │           │  Text    │
   │ Proj.    │           │ Features │
   │ (256-d)  │           │ (768-d)  │
   └────┬─────┘           └────┬─────┘
        │                      │
        └──────────┬───────────┘
                   ▼
            ┌────────────┐
            │   Fusion   │
            │  Regressor │
            └──────┬─────┘
                   ▼
            ┌────────────┐
            │ Prediction │
            └────────────┘
```

### Ensemble Pipeline

```
                    ┌──────────────┐
                    │ Audio Files  │
                    └──────┬───────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
   ┌──────────┐     ┌──────────┐     ┌──────────┐
   │Traditional│     │   Deep   │     │ Metadata │
   │ Features  │     │ Features │     │ Features │
   │ (160-d)   │     │ (1536-d) │     │  (9-d)   │
   └─────┬────┘     └─────┬────┘     └─────┬────┘
         │                │                 │
         └────────────────┼─────────────────┘
                          ▼
                  ┌───────────────┐
                  │ Feature Pool  │
                  │   (289-d)     │
                  └───────┬───────┘
                          │
         ┌────────────────┼────────────────┐
         │                │                │
         ▼                ▼                ▼
   ┌──────────┐     ┌──────────┐     ┌──────────┐
   │ XGBoost  │     │ LightGBM │     │ CatBoost │
   └─────┬────┘     └─────┬────┘     └─────┬────┘
         │                │                 │
         └────────────────┼─────────────────┘
                          ▼
                   ┌─────────────┐
                   │   Ensemble  │
                   │  Prediction │
                   └─────────────┘
```

## 🚀 Usage

### Quick Start

```python
# 1. Load and prepare data
import pandas as pd
import numpy as np

train_df = pd.read_csv('csvs/train.csv')
test_df = pd.read_csv('csvs/test.csv')

# 2. Extract features
from feature_extraction import EfficientAudioFeatureExtractor

extractor = EfficientAudioFeatureExtractor()
train_features = extractor.extract_all(train_audio_files)
test_features = extractor.extract_all(test_audio_files)

# 3. Train ensemble models
from models import train_ensemble

models = train_ensemble(train_features, train_labels)

# 4. Make predictions
predictions = ensemble_predict(models, test_features)

# 5. Save submission
submission = pd.DataFrame({
    'filename': test_df['filename'],
    'label': predictions
})
submission.to_csv('submission.csv', index=False)
```

### Advanced Usage: Multi-Modal Training

```python
from multi_modal import train_multimodal_model

# Train with audio + text descriptions
model, best_rmse, predictions = train_multimodal_model()

# Features:
# - Wav2Vec2 audio embeddings
# - DistilBERT text processing
# - LoRA efficient fine-tuning
# - Feature caching
```

## 🔍 Feature Extraction

### Cell A: Ultra-Advanced Features

```python
# Extract 160 comprehensive audio features
features = extract_ultra_advanced_features(audio_path)

# Features include:
# - Energy (RMS, percentiles, temporal)
# - Spectral (centroid, bandwidth, rolloff, contrast, flatness, flux)
# - Pitch (F0, jitter, shimmer, HNR)
# - MFCCs and deltas
# - Mel spectrogram statistics
# - Chroma features
# - Rhythm and tempo
# - Harmonic/percussive separation
# - Audio quality metrics
```

### Cell B: Feature Combination

```python
# Combine multiple feature sets
combined_features = combine_features(
    mfcc_features,      # 120 features
    ultra_features,     # 160 features
    metadata_features   # 9 features
)  # Total: 289 features
```

### GPU-Accelerated Extraction

```python
# Use GPU for faster processing
features = extract_gpu_features(audio_path)

# Leverages:
# - PyTorch tensors on CUDA
# - torchaudio transforms
# - Parallel processing
```

## 🎓 Model Training

### Cross-Validation Strategy

- **5-Fold Cross-Validation**
- **Stratified splits** (if applicable)
- **Random state**: 42 for reproducibility

### Ensemble Models

#### XGBoost
```python
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 2000,
    'early_stopping_rounds': 50
}
```

#### LightGBM
```python
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'verbosity': -1
}
```

#### CatBoost
```python
params = {
    'loss_function': 'RMSE',
    'iterations': 2000,
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 3,
    'verbose': False
}
```

### Multi-Modal Training

```python
# LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_lin", "k_lin", "v_lin"]
)

# Training settings
batch_size = 4
learning_rate = 2e-5
epochs = 15
warmup_ratio = 0.1
```

## 📊 Results

### Cross-Validation Performance

```
Fold 1: RMSE = 0.6342
Fold 2: RMSE = 0.6177
Fold 3: RMSE = 0.7097
Fold 4: RMSE = 0.6055
Fold 5: RMSE = 0.7607

Mean CV RMSE: 0.6656 ± 0.0598
OOF RMSE: 0.6680
```

### Model Contributions

- **XGBoost**: Strong baseline performance
- **LightGBM**: Fast training, good generalization
- **CatBoost**: Robust to overfitting
- **Random Forest**: Ensemble diversity
- **Extra Trees**: Additional randomization

### Test Predictions

```
Range: [2.455, 3.887]
Mean: 2.988
Median: 2.951
Std: 0.281
```

## 📂 Project Structure

```
├── Code.ipynb                       # Main notebook
├── README.md                        # This file
│
├── features/                        # Feature storage
│   ├── ultraadvanced_features_train.npz
│   ├── ultraadvanced_features_test.npz
│   ├── train_features_fixed.npz
│   ├── test_features_fixed.npz
│   ├── meta_train_feats.npz
│   ├── meta_test_feats.npz
│   ├── combined_train_v4.npz
│   └── combined_test_v4.npz
│
├── cache/                           # Model cache
│   └── audio_*.npy                  # Cached features
│
└── submissions/
    ├── optimized_ensemble_submission.csv
    └── multimodal_llm_submission.csv
```

## 🔬 Technical Details

### Audio Processing

- **Sample Rate**: 22,050 Hz (traditional), 16,000 Hz (deep learning)
- **Hop Length**: 512 samples
- **FFT Size**: 2048
- **Mel Bands**: 128
- **MFCCs**: 20-40 coefficients
- **Max Duration**: 45 seconds (padded/trimmed)

### Feature Imputation

- **Method**: Median imputation
- **NaN Handling**: Robust to missing values
- **Scaling**: StandardScaler / RobustScaler

### Memory Optimization

- **Garbage Collection**: Every 25 files
- **GPU Cache**: Cleared periodically
- **Feature Caching**: Disk-based storage
- **Batch Processing**: Configurable batch sizes

## 🎯 Key Innovations

1. **Multi-Scale Feature Extraction**
   - Combines handcrafted and learned features
   - Multiple time scales and representations

2. **GPU Acceleration**
   - PyTorch-based feature extraction
   - 10x faster than CPU-only

3. **Multi-Modal Fusion**
   - Audio embeddings + text descriptions
   - Cross-modal attention mechanisms

4. **LoRA Fine-tuning**
   - Parameter-efficient training
   - Only 2% of parameters trained

5. **Ensemble Diversity**
   - Multiple model architectures
   - Different feature subsets
   - Various hyperparameters

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Reduce batch size
config.batch_size = 2

# Clear cache more frequently
config.gc_frequency = 10
```

**2. Audio Loading Errors**
```python
# Ensure .wav extension
filename = filename if filename.endswith('.wav') else f"{filename}.wav"

# Check audio file integrity
librosa.load(audio_path, sr=None)
```

**3. Feature Dimension Mismatch**
```python
# Verify feature count
assert len(features) == TARGET_FEATURES
```

## 🔄 Workflow

1. **Data Preparation**
   - Load CSV files
   - Verify audio file paths
   - Add .wav extensions if needed

2. **Feature Extraction**
   - Extract ultra-advanced features (Cell A)
   - Generate MFCC features (Cell B)
   - Extract metadata features
   - Combine all features

3. **Model Training**
   - 5-fold cross-validation
   - Train ensemble models
   - Multi-modal training (optional)

4. **Prediction**
   - Test time augmentation
   - Ensemble averaging
   - Generate submission

5. **Submission**
   - Format predictions
   - Remove .wav extensions
   - Save to CSV

## 📈 Performance Tips

### Speed Optimization

- Enable GPU acceleration
- Use cached features
- Reduce TTA augmentations
- Parallelize feature extraction

### Accuracy Improvement

- Increase ensemble diversity
- Add more feature types
- Tune hyperparameters
- Use pseudo-labeling
- Increase TTA augmentations

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional feature engineering
- New model architectures
- Hyperparameter optimization
- Data augmentation techniques
- Documentation improvements

## 📝 License

This project is provided for educational and research purposes.

## 🙏 Acknowledgments

- **Librosa**: Audio processing library
- **Transformers**: Hugging Face transformers
- **XGBoost/LightGBM/CatBoost**: Ensemble libraries
- **PyTorch**: Deep learning framework
- **Kaggle**: Platform and compute resources

## 📧 Contact

For questions or issues, please open an issue in the repository.

---

**Last Updated**: December 2025
**Author**: Audio ML Researcher
**Version**: 2.1
