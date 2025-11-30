# ALS Risk Assessment Platform

An AI-powered system for early detection and risk stratification of Amyotrophic Lateral Sclerosis (ALS) using machine learning and voice biomarkers.

## Overview

This platform leverages XGBoost classification to assess ALS risk based on acoustic and speech biomarkers. It provides:
- **Individual risk probabilities** for patients
- **Population-level risk distribution** analysis
- **Comparative analysis** by age and gender
- **Model performance validation** with ground truth labels (if available)

## Features

- 🧬 **Biomarker Analysis**: Processes 130+ acoustic and speech features
- 📊 **Interactive Dashboard**: Real-time visualization of risk distribution and patient stratification
- 🔍 **Detailed Patient Analysis**: Filter and export individual risk assessments
- 📈 **Performance Metrics**: Confusion matrix, ROC AUC, accuracy, and detailed classification reports
- 🎯 **Risk Categorization**: Automatic categorization into No Risk, Low, Moderate, High, and Very High
- 📥 **CSV Import**: Batch processing of patient data
- 📥 **Results Export**: Download predictions and analysis results

## Project Structure

```
als_detection/
├── streamlit_app.py                    # Main Streamlit application
├── train_tabular.py                    # Model training script
├── prepare_data.py                     # Data preprocessing utilities
├── synthetic_data.py                   # Synthetic data generation
├── requirements.txt                    # Python dependencies
├── models/
│   ├── xgb_als_model.pkl              # Trained XGBoost classifier
│   ├── scaler.pkl                      # Feature scaler (StandardScaler)
│   ├── feature_names.pkl               # List of feature names
│   └── metrics.json                    # Training metrics
├── Minsk2020_ALS_dataset.csv           # Original ALS dataset
├── processed_features.csv              # Processed feature set
└── synthetic_univariate_10000_ALS.csv  # Synthetic test data
```

## Installation

### Prerequisites
- Python 3.10+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Hiren-Karwani/als-detection.git
cd als_detection
```

2. Create a virtual environment:
```bash
python -m venv .venv
```

3. Activate the virtual environment:
   - **Windows PowerShell:**
     ```powershell
     & .venv\Scripts\Activate.ps1
     ```
   - **macOS/Linux:**
     ```bash
     source .venv/bin/activate
     ```

4. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### Running the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`. Upload a CSV file with patient biomarker data to generate risk assessments.

### Training the Model

To retrain the model on your own data:

```bash
python train_tabular.py
```

Ensure your data includes:
- Feature columns matching the trained model's 130+ biomarkers
- A `label` column with binary values (0 = No ALS, 1 = ALS)

### Data Preparation

Preprocess raw audio/speech data using:

```bash
python prepare_data.py
```

## Data Format

### Input CSV Requirements

Your CSV file should contain:
- **Biomarker columns**: Acoustic and speech features (e.g., `J1_a`, `S1_a`, `DPF_a`, etc.)
- **Optional columns**:
  - `Age`: Patient age for age-based analysis
  - `Sex`: Patient gender (M/F)
  - `Diagnosis (ALS)` or `label`: Ground truth (0 or 1) for model validation

Example structure:
```
ID,Age,Sex,J1_a,J3_a,S1_a,...,Diagnosis (ALS)
1,58,M,0.32,0.14,6.04,...,1
2,57,F,0.34,0.18,1.97,...,1
```

## Dependencies

Core dependencies (see `requirements.txt`):
- **pandas** (≥1.5) – Data manipulation
- **numpy** (≥1.24) – Numerical computing
- **scikit-learn** (≥1.2) – ML utilities and metrics
- **xgboost** (≥1.7) – Gradient boosting classifier
- **joblib** (≥1.2) – Model serialization
- **streamlit** (≥1.20) – Web app framework
- **plotly** (≥5.14) – Interactive visualizations
- **seaborn** (≥0.12) – Statistical plotting
- **matplotlib** (≥3.6) – Plotting library

## Model Details

- **Algorithm**: XGBoost Classifier
- **Training Set Size**: ~130 features, balanced class distribution
- **Hyperparameters**:
  - `n_estimators=300`
  - `learning_rate=0.05`
  - `max_depth=6`
  - `subsample=0.8`
  - `colsample_bytree=0.8`

### Training Metrics

The model was trained with the following performance (stored in `models/metrics.json`):
- **Accuracy**: Monitored during training
- **ROC AUC**: Computed on validation set

## Dashboard Overview

### 📊 Executive Dashboard
- **Risk Distribution**: Pie chart showing patient segmentation by risk level
- **Probability Density**: Histogram of predicted risk probabilities
- **Age-Based Risk Profile**: Line chart showing average risk by age group

### 📋 Patient Analysis
- **Detailed Stratification**: View individual patient predictions with filtering
- **Risk Filters**: Filter by probability range
- **Results Export**: Download predictions as CSV

### ⚙️ Model Performance
- **Diagnostic Performance**: Accuracy, ROC AUC, and Recall metrics (if labels provided)
- **Confusion Matrix**: True positive/negative vs predicted
- **Classification Report**: Precision, recall, F1-score per class
- **Reference Metrics**: Original training performance shown if ground truth unavailable

## Risk Categories

Patients are automatically categorized as:
- **No Risk**: 0% probability
- **Low**: 0–25%
- **Moderate**: 25–50%
- **High**: 50–75%
- **Very High**: 75–100%

## Features & Biomarkers

The model processes the following feature groups:

### Acoustic Features (by phonation type)
- **J (Jitter)**: Frequency perturbation (a, i)
- **S (Shimmer)**: Amplitude perturbation (a, i)
- **DPF**: Differential phonation frequency
- **PFR**: Phonation frequency range
- **PPE**: Pitch period entropy
- **PVI**: Pitch-voiced intervals
- **HNR**: Harmonics-to-noise ratio
- **GNE**: Glottal-to-noise energy

### Harmonic Features
- **Ha**: Harmonic components (8 per phonation type)
  - Mean (`Ha(1)_{mu}` to `Ha(8)_{mu}`)
  - Standard deviation (`Ha(1)_{sd}` to `Ha(8)_{sd}`)
  - Relative energy (`Ha(1)_{rel}` to `Ha(8)_{rel}`)

### Cepstral Coefficients
- **CC**: Cepstral coefficients (12 per phonation type)
- **dCC**: Differential cepstral coefficients (12 per phonation type)

### Summary Metrics
- **d_1**: Duration feature
- **F2_i**: Second formant (isolated)
- **F2_conv**: Second formant (continuous)

## Troubleshooting

### Model files not found
Ensure `models/` directory contains all required pickle files:
- `xgb_als_model.pkl`
- `scaler.pkl`
- `feature_names.pkl`
- `metrics.json`

### Label detection fails
If your label column isn't recognized, ensure the column name matches one of:
- `label`
- `Diagnosis (ALS)`
- `Diagnosis(ALS)`
- `Diagnosis`
- `ALS`

Or use a custom column and convert manually to binary (0/1).

### Missing features warning
If biomarker columns are missing, the app will impute them with 0. For accuracy, ensure all required features are present in your input CSV.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License – see the LICENSE file for details.

## Citation

If you use this platform in your research, please cite:

```bibtex
@software{als_detection_2024,
  title={ALS Risk Assessment Platform},
  author={Karwani, Hiren},
  year={2024},
  url={https://github.com/Hiren-Karwani/als-detection}
}
```

## Disclaimer

This tool is intended for **research and educational purposes only**. It is not a medical device and should not be used for clinical diagnosis. Always consult qualified healthcare professionals for medical advice.

## Support

For issues, questions, or suggestions, please open an [issue](https://github.com/Hiren-Karwani/als-detection/issues) on GitHub.

## Acknowledgments

- Dataset: Minsk 2020 ALS Speech Dataset
- Framework: Streamlit
- ML Library: XGBoost, scikit-learn
- Visualization: Plotly

---

**Last Updated**: November 2024  
**Version**: 1.0.0
