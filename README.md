# 🧬 ALS Risk Assessment Platform

An AI-powered web application for early detection and risk stratification of Amyotrophic Lateral Sclerosis (ALS) using voice biomarkers and acoustic features.

## Overview

This platform uses a trained XGBoost model to predict ALS risk probability based on voice analysis features. It provides:

- **Individual risk predictions** with probability scores (0-100%)
- **Risk stratification** (No Risk, Low, Moderate, High, Very High)
- **Population-level analytics** including age-based risk profiles
- **Model performance validation** with confusion matrix and classification metrics
- **Interactive dashboard** with filters and data export capabilities

## Features

### 🎙️ Multi-Modal Input Support
- **📊 Tabular Data**: Upload CSV files with biomarker features
- **🎙️ Voice Samples**: Upload WAV, MP3, FLAC, or OGG audio files for automatic feature extraction
- **🎤 Live Recording**: Record voice samples directly in the app (requires streamlit-webrtc)

### 📊 Executive Dashboard
- Risk distribution pie chart (patient segmentation by risk level)
- Probability density histogram
- Age-based risk trajectory analysis
- Voice sample analysis with spectrograms and feature distributions

### 📋 Patient Analysis
- Detailed patient stratification table
- Risk probability filtering (0-100%)
- CSV export of results
- Audio feature visualization for voice samples

### ⚙️ Model Performance
- Diagnostic accuracy, ROC AUC, and recall metrics (when labels provided)
- Confusion matrix visualization
- Detailed classification report
- Reference training metrics display

## Project Structure

```
als_detection/
├── streamlit_app.py              # Main Streamlit application
├── train_tabular.py              # Model training script
├── prepare_data.py               # Data preparation utilities
├── synthetic_data.py             # Synthetic data generation
├── requirements.txt              # Python dependencies
├── README.md                      # This file
├── models/
│   ├── xgb_als_model.pkl         # Trained XGBoost classifier
│   ├── scaler.pkl                # StandardScaler for feature normalization
│   ├── feature_names.pkl         # List of 131 required features
│   └── metrics.json              # Training metrics (accuracy, ROC AUC)
├── Minsk2020_ALS_dataset.csv     # Source dataset
├── processed_features.csv        # Processed features for training
└── synthetic_univariate_10000_ALS.csv  # Synthetic test data
```

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup

1. **Clone or extract the project**:
   ```bash
   cd C:\Users\karwa\Desktop\als_detection
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   ```powershell
   # Windows PowerShell
   & C:/.venv/Scripts/Activate.ps1
   
   # Or use Command Prompt
   .venv\Scripts\activate.bat
   ```

4. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Usage

### Running the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501` in your default browser.

**Steps to use**:
1. Select input mode in the sidebar:
   - **📊 Tabular Data**: Click "Upload Patient Data (CSV)" and select your file
   - **🎙️ Voice Samples**: Click "Upload Voice Samples" and select WAV, MP3, FLAC, or OGG files
   - **🎤 Live Recording**: Click the record button to capture audio directly
2. View risk predictions and analytics in the dashboard tabs

### Voice Sample Processing

When uploading voice samples, the app automatically extracts the following acoustic features:

- **Energy & Amplitude**: Mean, standard deviation, maximum amplitude
- **Spectral Features**: Centroid, rolloff
- **Zero Crossing Rate (ZCR)**: Voice activity indicator
- **MFCC (Mel-Frequency Cepstral Coefficients)**: 13 coefficients capturing voice characteristics
- **Chroma Features**: 12 pitch-class features
- **Fundamental Frequency (F0)**: Pitch estimates (mean, std, min, max)
- **RMS Energy**: Root Mean Square energy of audio frames
- **Onset Strength**: Voice articulation measure

These features are then fed into the trained model to predict ALS risk probability.

### Training the Model

To retrain the model on new data:

```bash
python train_tabular.py
```

This will:
- Load features from `processed_features.csv`
- Train an XGBoost classifier
- Save model, scaler, and features to `models/`
- Generate `models/metrics.json` with performance metrics

### Preparing Data

To prepare raw data for training:


   ```bash
   python prepare_data.py
   ```

   This script processes raw voice biomarker data and extracts features.

   ### Generating Synthetic Data

   For testing, development, and validation purposes, the platform provides tools to generate synthetic voice samples and tabular data.

   #### 1. Synthetic Voice Samples

   Generate synthetic voice recordings with ALS-like characteristics:

   ```bash
   python synthetic_audio.py
   ```

   This creates a `synthetic voice` folder with 20 subjects:
   - **10 Controls (CT001-CT010)**: Healthy voice samples with low noise
   - **10 Patients (PZ001-PZ010)**: ALS-simulated voice samples with dysarthria effects

   **Generated Phonations** (8 types per subject):

   | Type | Description | Duration |
   |------|-------------|----------|
   | phonationA | Vowel "A" sustain | 3 seconds |
   | phonationE | Vowel "E" sustain | 3 seconds |
   | phonationI | Vowel "I" sustain | 3 seconds |
   | phonationO | Vowel "O" sustain | 3 seconds |
   | phonationU | Vowel "U" sustain | 3 seconds |
   | rhythmKA | Repeated syllables | ~2.5 seconds (5 bursts) |
   | rhythmPA | Repeated syllables | ~2.5 seconds (5 bursts) |
   | rhythmTA | Repeated syllables | ~2.5 seconds (5 bursts) |

   **Control vs. ALS-Simulated Characteristics**:

   | Aspect | Controls | ALS Patients |
   |--------|----------|-------------|
   | Noise Level | 0.005 | 0.05 (10x higher) |
   | Tremor | None | 5Hz tremolo modulation |
   | Burst Timing | Regular | Irregular (dysarthria) |
   | Voice Quality | Clean | Breathy/dysarthric |

   **Output Format**: WAV files (8kHz sample rate, 32-bit float)

   #### 2. Synthetic Tabular Data

   Generate synthetic feature datasets for model testing:

   ```bash
   python synthetic_data.py
   ```

   This creates `synthetic_univariate_10000_ALS.csv` with:
   - **10,000 synthetic records**
   - **131 biomarker features** matching the training model
   - **Binary labels** (0: No ALS, 1: ALS)
   - **Realistic feature distributions** based on training data

   **Use Cases**:
   - Quick model validation without uploading real patient data
   - Stress testing the dashboard with large datasets
   - Feature extraction pipeline verification
   - Performance benchmarking

   #### 3. Folder Structure After Generation

   ```
   als_detection/
   ├── synthetic voice/              # Generated voice samples
   │   ├── phonationA/
   │   │   ├── CT001_phonationA.wav
   │   │   ├── CT002_phonationA.wav
   │   │   └── ...
   │   ├── phonationE/
   │   ├── phonationI/
   │   ├── phonationO/
   │   ├── phonationU/
   │   ├── rhythmKA/
   │   ├── rhythmPA/
   │   └── rhythmTA/
   └── synthetic_univariate_10000_ALS.csv  # Generated tabular data
   ```
## Input Data Format

Your input can be one of three formats:

### 1. Tabular Data (CSV)
- **Features**: All 131 biomarker columns used during training (e.g., `J1_a`, `J3_a`, `S1_a`, `DPF_a`, `HNR_a`, `GNEa_μ`, `Ha(1)_mu`, `CCa(1)`, etc.)
- **Optional**: `Age` column for age-based analysis
- **Optional**: `Sex` column for demographic analysis
- **Optional**: Label column (`label`, `Diagnosis(ALS)`, `Diagnosis (ALS)`, `Diagnosis`, or `ALS`) with values 0 (no ALS) or 1 (ALS) for model validation

**Missing features** will be imputed with 0 (may impact accuracy).

### 2. Voice Samples (Audio Files)
- **Formats**: WAV, MP3, FLAC, OGG
- **Sample Rate**: Any sample rate (will be resampled to 8kHz for feature extraction)
- **Duration**: 3+ seconds recommended for stable feature extraction
- **Content**: Sustained vowel phonations (A, E, I, O, U) or repeated syllables (KA, PA, TA) are ideal

**Features automatically extracted**:
- MFCC coefficients (13)
- Spectral features (centroid, rolloff)
- Energy metrics
- Pitch (F0) estimates
- Zero crossing rate

### 3. Live Recording
- Click the "Record" button to capture audio directly from your microphone
- Record for at least 3 seconds for stable feature extraction
- Audio will be processed using the same pipeline as uploaded samples

## Dependencies

- **streamlit** ≥1.20 – Interactive web app framework
- **streamlit-webrtc** ≥0.47.0 – Live audio recording capability
- **pandas** ≥1.5 – Data manipulation
- **numpy** ≥1.24 – Numerical computing
- **scikit-learn** ≥1.2 – Machine learning utilities (metrics, preprocessing)
- **xgboost** ≥1.7 – Gradient boosting classifier
- **joblib** ≥1.2 – Model serialization
- **plotly** ≥5.14 – Interactive visualizations
- **librosa** ≥0.10.0 – Audio feature extraction
- **soundfile** ≥0.12.0 – Audio file I/O
- **scipy** ≥1.10.0 – Signal processing
- **matplotlib** ≥3.6 – Plotting library

See `requirements.txt` for exact versions.

## Model Details

### Algorithm
- **XGBoost Classifier** with 300 estimators
- Learning rate: 0.05
- Max depth: 6
- Subsample & colsample: 0.8
- Stratified train-test split (80/20)
- StandardScaler normalization

### Training Data
- **Dataset**: Minsk2020 (voice samples from ALS patients and controls)
- **Features**: 131 acoustic & voice biomarkers (Jitter, Shimmer, HNR, Cepstral Coefficients, etc.)
- **Classes**: Binary (0: No ALS, 1: ALS)

### Performance (Training)
- Accuracy: Available in `models/metrics.json`
- ROC AUC: Available in `models/metrics.json`

## Features Explained

The model uses voice biomarker features including:

- **Jitter & Shimmer**: Frequency and amplitude perturbation measures
- **HNR (Harmonics-to-Noise Ratio)**: Voice quality indicator
- **DPF, PFR, PPE**: Fundamental frequency derivatives
- **GNE (Glottal-to-Noise Excitation Ratio)**: Voice source characteristics
- **MFCC & Cepstral Coefficients**: Spectral features
- **Delta coefficients**: Feature velocity and acceleration

## Risk Categories

| Category | Probability | Color | Interpretation |
|----------|------------|-------|-----------------|
| No Risk | 0% | Gray | Negligible ALS risk |
| Low | 0-25% | Light Blue | Low probability |
| Moderate | 25-50% | Blue | Moderate concern |
| High | 50-75% | Dark Blue | High probability |
| Very High | 75-100% | Navy | Very high probability |

## Troubleshooting

### "Model files not found"
Ensure the `models/` directory exists with:
- `xgb_als_model.pkl`
- `scaler.pkl`
- `feature_names.pkl`

### "Could not coerce label column to integer"
Label columns must contain numeric values (0 or 1). Rename string labels if needed.

### "Imputing missing features"
Your CSV may be missing some of the 131 required features. Missing features default to 0, which may reduce accuracy. Ensure all feature columns are included.

### Performance metrics not showing
If no label column is detected, the app will show a warning and display only reference training metrics. Ensure your CSV has a column named `label` or `Diagnosis (ALS)` with 0/1 values.

### "Error extracting audio features"
- Ensure audio file is not corrupted
- Try converting to WAV format (most reliable)
- Minimum duration should be 1 second
- Supported formats: WAV, MP3, FLAC, OGG

### Live recording not working
- Install streamlit-webrtc: `pip install streamlit-webrtc`
- Ensure your browser allows microphone access
- Check that you have an active microphone device
- Use a modern browser (Chrome, Firefox, Edge recommended)

## Output

### Downloaded CSV
When you download results, the CSV includes:
- All original columns from your input
- `ALS_Risk_Probability`: Model prediction (0-100%)
- `Risk_Category`: Risk level classification

## Author & Attribution

**Owner**: Hiren-Karwani  
**Repository**: [als-detection](https://github.com/Hiren-Karwani/als-detection)  
**License**: [Specify if applicable]

## Citation

If you use this platform in research, please cite:

```bibtex
@software{als_detection_2025,
  title={ALS Risk Assessment Platform},
  author={Karwani, Hiren},
  year={2025},
  url={https://github.com/Hiren-Karwani/als-detection}
}
```

## Support & Contributing

For issues, feature requests, or contributions:
- Open an issue on the [GitHub repository](https://github.com/Hiren-Karwani/als-detection)
- Submit a pull request with improvements
- Contact the maintainers

## Disclaimer

**This tool is for research and educational purposes only.** It should not be used for clinical diagnosis without validation by qualified medical professionals. Always consult with healthcare providers for ALS diagnosis and treatment decisions.

---

**Last Updated**: November 2025  
**Version**: 1.0

```
als_detection/
├── streamlit_app.py                          # Main Streamlit application
├── train_tabular.py                          # Model training script
├── prepare_data.py                           # Data preparation utilities
├── synthetic_data.py                         # Generates synthetic tabular data (10k records, 131 features)
├── synthetic_audio.py                        # Generates synthetic voice samples (WAV files with ALS characteristics)
├── requirements.txt                          # Python dependencies
├── README.md                                 # This file
├── models/
│   ├── xgb_als_model.pkl                    # Trained XGBoost classifier
│   ├── scaler.pkl                           # StandardScaler for feature normalization
│   ├── feature_names.pkl                    # List of 131 required features
│   └── metrics.json                         # Training metrics (accuracy, ROC AUC)
├── Minsk2020_ALS_dataset.csv                # Source dataset
├── processed_features.csv                   # Processed features for training
├── synthetic_univariate_10000_ALS.csv       # Synthetic tabular test data
├── synthetic voice/                         # Generated voice samples (created by synthetic_audio.py)
│   ├── phonationA/, phonationE/, ... (8 directories)
│   └── Contains 160 WAV files (8 phonation types × 20 subjects)
└── voice_sample/                            # Real voice samples from VOC-ALS dataset
   └── VOC-ALS/ (reference dataset structure)
```

- **Individual risk predictions** with probability scores (0-100%)
- **Risk stratification** (No Risk, Low, Moderate, High, Very High)
- **Population-level analytics** including age-based risk profiles
- **Model performance validation** with confusion matrix and classification metrics
- **Interactive dashboard** with filters and data export capabilities
- **Synthetic data generation** for testing and development (voice samples & tabular datasets)
- **Multi-modal input support** (CSV, WAV, MP3, FLAC, OGG, and live recording)

## Synthetic Data Guide

### Overview

Synthetic data is crucial for:
- **Development & Testing**: Safely test new features without real patient data
- **Model Validation**: Verify the pipeline works correctly
- **Performance Benchmarking**: Test scalability with large datasets
- **Educational Use**: Demonstrate platform capabilities

### Generation Scripts

#### synthetic_audio.py

**Purpose**: Generates synthetic voice samples simulating healthy controls and ALS patients

**Key Parameters**:
```python
n_controls = 10       # Number of control subjects
n_patients = 10       # Number of ALS-simulated subjects
sr = 8000             # Sample rate in Hz
duration = 3.0        # Duration per phonation in seconds
```

**Audio Generation Algorithm**:

1. **Base Tone Generation**:
   - Fundamental frequency: 80-300 Hz (realistic voice range)
   - Harmonics: 1st, 2nd, and 3rd order (voice-like quality)
   - Gaussian noise layer for natural variation

2. **Control Characteristics** (Healthy):
   - Low noise level: 0.005
   - Stable pitch and amplitude
   - Regular burst timing (for rhythm phonations)

3. **ALS-Simulated Characteristics**:
   - High noise level: 0.05 (10× increase = breathiness)
   - Tremolo modulation: 5Hz tremor (voice instability)
   - Irregular burst timing (±50ms variation)
   - Irregular silence duration (dysarthria simulation)

**Feature Extraction from Synthetic Audio**:
When uploaded, synthetic voice samples extract:
- Amplitude statistics (mean, std, max)
- Spectral centroid and rolloff
- MFCCs (13 coefficients)
- Fundamental frequency (F0) estimates
- Energy and RMS metrics
- Zero crossing rate

#### synthetic_data.py

**Purpose**: Generates synthetic feature vectors for quick model validation

**Output**:
- **Filename**: `synthetic_univariate_10000_ALS.csv`
- **Rows**: 10,000 synthetic records
- **Columns**: 131 biomarker features + 1 label column

**Feature Generation**:
- Based on statistical properties of training data (mean, std)
- Realistic correlations between features preserved
- Two classes: 0 (No ALS), 1 (ALS)

**Use Cases**:
```bash
# Quick validation without real data
python synthetic_data.py
# Then upload synthetic_univariate_10000_ALS.csv to streamlit app
```

### Testing Workflow

**Step 1: Generate Synthetic Data**
```bash
python synthetic_audio.py      # Creates synthetic voice/
python synthetic_data.py       # Creates synthetic_univariate_10000_ALS.csv
```

**Step 2: Test Voice Processing**
```bash
streamlit run streamlit_app.py
# Select "Voice Samples" mode
# Upload WAV from synthetic voice/phonationA/
# Verify feature extraction and risk prediction
```

**Step 3: Test Tabular Data Processing**
```bash
# In streamlit app, select "Tabular Data" mode
# Upload synthetic_univariate_10000_ALS.csv
# View dashboard with 10,000 records
# Verify model inference and performance metrics
```

### Expected Results

**Voice Samples**:
- Controls: Lower risk probabilities (typically <40%)
- ALS patients: Higher risk probabilities (typically >60%)
- Due to simulated dysarthria effects (noise, tremor)

**Tabular Data**:
- Accuracy: ~70-85% on synthetic data
- ROC AUC: ~0.75-0.90
- Balanced class distribution in risk categories

### Customizing Synthetic Data

Edit these files to customize generation:

**synthetic_audio.py**:
```python
n_controls = 50        # Increase controls
n_patients = 50        # Increase patients
noise_level = 0.08     # Higher noise = more dysarthric
```

**synthetic_data.py**:
```python
n_samples = 50000      # Change dataset size
random_state = 42      # Reproducible results
```

### Limitations

Synthetic data is **not a substitute for real clinical data**:
- Simplified acoustic model (single harmonics)
- Does not capture all real dysarthria characteristics
- Feature distributions may differ from real patients
- Should not be used for clinical deployment

**Use for**: Development, testing, and education only
