import os
import glob
import numpy as np
import pandas as pd
import librosa
import parselmouth
from parselmouth.praat import call
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

def extract_features(file_path):
    """Extracts features and handles Nyquist frequency limits for low-SR audio."""
    try:
        # Load for Librosa features
        y, sr = librosa.load(file_path, sr=None)
        
        # 1. MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        
        # 2. Spectral Contrast (FIX: Adjust n_bands for low sampling rates)
        # Default is 6. We reduce it if Nyquist (sr/2) is too low.
        nyquist = sr / 2
        n_bands = 6
        while (200 * (2**n_bands)) > nyquist and n_bands > 1:
            n_bands -= 1
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=n_bands)
        
        # 3. Clinical Markers (Parselmouth/Praat)
        sound = parselmouth.Sound(file_path)
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
        local_jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        local_shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        hnr = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        mean_hnr = call(hnr, "Get mean", 0, 0)
        
        # Aggregate all features
        features = []
        features.extend([np.mean(mfccs), np.std(mfccs)])
        features.extend([np.mean(contrast), np.std(contrast)])
        features.extend([local_jitter, local_shimmer, mean_hnr])
        
        return np.array(features)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    base_dir = Path("c:/Users/karwa/Downloads/als_detection")
    data_dir = base_dir / "voice_sample" / "VOC-ALS"
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Feature Names for Plotting
    feat_names = ["mfcc_mean", "mfcc_std", "contrast_mean", "contrast_std", "jitter", "shimmer", "hnr"]
    
    print("Loading data and extracting features...")
    wav_files = glob.glob(str(data_dir / "**" / "*.wav"), recursive=True)
    X, y = [], []
    
    for f in wav_files:
        label = 1 if os.path.basename(f).startswith("PZ") else 0
        feat = extract_features(f)
        if feat is not None:
            X.append(feat)
            y.append(label)
            
    X, y = np.array(X), np.array(y)
    
    # 5-Fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    
    print(f"Starting Training on {len(X)} samples...")
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, eval_metric='logloss')
        model.fit(X_train_s, y_train)
        
        preds = model.predict(X_test_s)
        print(f"Fold {fold+1} Accuracy: {accuracy_score(y_test, preds):.4f}")

    # Save final artifacts
    joblib.dump(model, models_dir / "final_xgb_model.pkl")
    joblib.dump(scaler, models_dir / "final_scaler.pkl")
    
    # Feature Importance Plot
    plt.figure(figsize=(10, 6))
    pd.Series(model.feature_importances_, index=feat_names).sort_values().plot(kind='barh')
    plt.title("Clinical Feature Importance")
    plt.savefig(models_dir / "importance_plot.png")
    print(f"Results saved to {models_dir}")

if __name__ == "__main__":
    main()