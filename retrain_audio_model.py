import os
import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import parselmouth
from parselmouth.praat import call
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier


def compute_parselmouth_features(audio_path):
    feats = {}
    try:
        snd = parselmouth.Sound(audio_path)
        try:
            pitch = snd.to_pitch()
            f0_values = pitch.selected_array['frequency']
            f0_nonzero = f0_values[f0_values > 0]
            feats['f0_mean'] = float(f0_nonzero.mean()) if len(f0_nonzero) > 0 else 0.0
            feats['f0_std'] = float(f0_nonzero.std()) if len(f0_nonzero) > 0 else 0.0
        except Exception:
            feats['f0_mean'] = 0.0
            feats['f0_std'] = 0.0

        try:
            point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
            try:
                feats['jitter_local'] = float(call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
            except Exception:
                feats['jitter_local'] = 0.0
            try:
                feats['jitter_rap'] = float(call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3))
            except Exception:
                feats['jitter_rap'] = 0.0
            try:
                feats['jitter_ppq5'] = float(call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3))
            except Exception:
                feats['jitter_ppq5'] = 0.0

            try:
                feats['shimmer_local'] = float(call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
            except Exception:
                feats['shimmer_local'] = 0.0
            try:
                feats['shimmer_apq3'] = float(call([snd, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
            except Exception:
                feats['shimmer_apq3'] = 0.0
            try:
                feats['shimmer_apq5'] = float(call([snd, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
            except Exception:
                feats['shimmer_apq5'] = 0.0
        except Exception:
            feats.setdefault('jitter_local', 0.0)
            feats.setdefault('jitter_rap', 0.0)
            feats.setdefault('jitter_ppq5', 0.0)
            feats.setdefault('shimmer_local', 0.0)
            feats.setdefault('shimmer_apq3', 0.0)
            feats.setdefault('shimmer_apq5', 0.0)

        try:
            harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            feats['hnr_mean'] = float(call(harmonicity, "Get mean", 0, 0))
        except Exception:
            feats['hnr_mean'] = 0.0

    except Exception:
        feats = {
            'f0_mean': 0.0, 'f0_std': 0.0,
            'jitter_local': 0.0, 'jitter_rap': 0.0, 'jitter_ppq5': 0.0,
            'shimmer_local': 0.0, 'shimmer_apq3': 0.0, 'shimmer_apq5': 0.0,
            'hnr_mean': 0.0
        }

    return feats


def extract_audio_features(audio_path, sr=8000):
    try:
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
        features = {}
        features['Mean_Amplitude'] = float(np.mean(np.abs(y)))
        features['Std_Amplitude'] = float(np.std(np.abs(y)))
        features['Max_Amplitude'] = float(np.max(np.abs(y)))

        energy = float(np.sum(y ** 2))
        features['Energy'] = energy
        features['Energy_Entropy'] = float(-np.sum((y ** 2) * np.log(y ** 2 + 1e-10)) / max(1, len(y)))

        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['ZCR_Mean'] = float(np.mean(zcr))
        features['ZCR_Std'] = float(np.std(zcr))

        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['Spectral_Centroid_Mean'] = float(np.mean(spec_centroid))
        features['Spectral_Centroid_Std'] = float(np.std(spec_centroid))

        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['Spectral_Rolloff_Mean'] = float(np.mean(spec_rolloff))
        features['Spectral_Rolloff_Std'] = float(np.std(spec_rolloff))

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'MFCC_{i}_Mean'] = float(np.mean(mfcc[i, :]))
            features[f'MFCC_{i}_Std'] = float(np.std(mfcc[i, :]))

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        for i in range(12):
            features[f'Chroma_{i}_Mean'] = float(np.mean(chroma[i, :]))

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        features['Onset_Strength_Mean'] = float(np.mean(onset_env))
        features['Onset_Strength_Std'] = float(np.std(onset_env))

        rms = librosa.feature.rms(y=y)[0]
        features['RMS_Mean'] = float(np.mean(rms))
        features['RMS_Std'] = float(np.std(rms))

        # Merge parselmouth clinical features
        pm_feats = compute_parselmouth_features(audio_path)
        features.update(pm_feats)

        return features
    except Exception as e:
        print(f"Error extracting features for {audio_path}: {e}")
        return None


def collect_dataset(data_dir):
    wav_files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
    X = []
    y = []
    names = []
    for fp in wav_files:
        feats = extract_audio_features(fp)
        if feats is None:
            continue
        label = 1 if os.path.basename(fp).startswith("PZ") else 0
        X.append(feats)
        y.append(label)
        names.append(os.path.basename(fp))

    if not X:
        raise RuntimeError(f"No features extracted from {data_dir}. Check files and dependencies.")

    # Build DataFrame with consistent column order
    all_keys = sorted({k for d in X for k in d.keys()})
    df = pd.DataFrame([{k: d.get(k, 0.0) for k in all_keys} for d in X])
    return df, np.array(y), names, all_keys


def train_and_save(data_dir, models_dir):
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting dataset from {data_dir}...")
    X_df, y, names, feature_names = collect_dataset(data_dir)
    print(f"Extracted features shape: {X_df.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values)

    # Cross-validated evaluation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    rocs = []
    fold = 0
    for train_idx, test_idx in skf.split(X_scaled, y):
        fold += 1
        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        probas = model.predict_proba(X_te)[:, 1]
        acc = accuracy_score(y_te, preds)
        try:
            roc = roc_auc_score(y_te, probas)
        except Exception:
            roc = float('nan')
        accs.append(acc)
        rocs.append(roc)
        print(f"Fold {fold} - Acc: {acc:.4f}, ROC AUC: {roc:.4f}")

    metrics = {
        'cv_accuracy_mean': float(np.mean(accs)),
        'cv_accuracy_std': float(np.std(accs)),
        'cv_roc_mean': float(np.nanmean(rocs)),
        'cv_roc_std': float(np.nanstd(rocs)),
        'n_samples': int(len(y)),
        'n_features': int(X_df.shape[1])
    }

    # Train final model on all data
    final_model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss')
    final_model.fit(X_scaled, y)

    # Save artifacts with expected names used by the app
    joblib.dump(final_model, models_dir / "xgb_als_model.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(feature_names, models_dir / "feature_names.pkl")

    with open(models_dir / 'metrics.json', 'w') as fh:
        json.dump(metrics, fh, indent=2)

    print(f"Saved model, scaler, feature names, and metrics to {models_dir}")


def main():
    base = Path(__file__).resolve().parent
    data_dir = base / "voice_sample" / "VOC-ALS"
    models_dir = base / "models"

    train_and_save(str(data_dir), str(models_dir))


if __name__ == '__main__':
    main()
