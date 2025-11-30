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

### 📊 Executive Dashboard
- Risk distribution pie chart (patient segmentation by risk level)
- Probability density histogram
- Age-based risk trajectory analysis

### 📋 Patient Analysis
- Detailed patient stratification table
- Risk probability filtering (0-100%)
- CSV export of results

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
1. Click "Upload Patient Data (CSV)" in the sidebar
2. Select your CSV file (must contain biomarker features)
3. View risk predictions and analytics in the dashboard tabs

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

## Input Data Format

Your CSV file should include:

- **Features**: All 131 biomarker columns used during training (e.g., `J1_a`, `J3_a`, `S1_a`, `DPF_a`, `HNR_a`, `GNEa_μ`, `Ha(1)_mu`, `CCa(1)`, etc.)
- **Optional**: `Age` column for age-based analysis
- **Optional**: `Sex` column for demographic analysis
- **Optional**: Label column (`label`, `Diagnosis(ALS)`, `Diagnosis (ALS)`, `Diagnosis`, or `ALS`) with values 0 (no ALS) or 1 (ALS) for model validation

**Missing features** will be imputed with 0 (may impact accuracy).

## Dependencies

- **streamlit** ≥1.20 – Interactive web app framework
- **pandas** ≥1.5 – Data manipulation
- **numpy** ≥1.24 – Numerical computing
- **scikit-learn** ≥1.2 – Machine learning utilities (metrics, preprocessing)
- **xgboost** ≥1.7 – Gradient boosting classifier
- **joblib** ≥1.2 – Model serialization
- **plotly** ≥5.14 – Interactive visualizations
- **seaborn** ≥0.12 – Statistical visualizations
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
