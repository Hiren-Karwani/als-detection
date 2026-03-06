import os
import numpy as np
import librosa
from pathlib import Path
import joblib
try:
    import parselmouth
    from parselmouth.praat import call
except Exception:
    parselmouth = None


def load_artifacts(models_dir=None):
    base = Path(__file__).resolve().parent
    models_dir = Path(models_dir) if models_dir else base / 'models'
    model_p = models_dir / 'xgb_als_model.pkl'
    scaler_p = models_dir / 'scaler.pkl'
    feat_p = models_dir / 'feature_names.pkl'
    if not model_p.exists() or not scaler_p.exists() or not feat_p.exists():
        raise FileNotFoundError('Model artifacts not found in models/ — run retrain_combined.py')
    model = joblib.load(model_p)
    scaler = joblib.load(scaler_p)
    feature_names = joblib.load(feat_p)
    # metrics optional
    metrics = None
    mjson = models_dir / 'metrics.json'
    if mjson.exists():
        import json
        metrics = json.load(open(mjson))
    return model, scaler, feature_names, metrics


def compute_parselmouth_features(audio_path):
    feats = {}
    if parselmouth is None:
        return feats
    try:
        snd = parselmouth.Sound(audio_path)
        pitch = snd.to_pitch()
        f0 = pitch.selected_array['frequency']
        if len(f0) > 0:
            feats['f0_mean'] = float(np.mean(f0))
            feats['f0_std'] = float(np.std(f0))
        else:
            feats['f0_mean'] = 0.0; feats['f0_std'] = 0.0
        point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        feats['jitter_local'] = float(call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
        feats['shimmer_local'] = float(call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
        hnr = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        feats['hnr_mean'] = float(call(hnr, "Get mean", 0, 0))
    except Exception:
        pass
    return feats


def extract_audio_features(audio_path, sr=8000):
    try:
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
        feats = {}
        feats['Mean_Amplitude'] = float(np.mean(np.abs(y)))
        feats['Std_Amplitude'] = float(np.std(np.abs(y)))
        feats['Max_Amplitude'] = float(np.max(np.abs(y)))
        feats['Energy'] = float(np.sum(y ** 2))
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        feats['ZCR_Mean'] = float(np.mean(zcr))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(mfcc.shape[0]):
            feats[f'mfcc_{i+1}_mean'] = float(np.mean(mfcc[i]))
            feats[f'mfcc_{i+1}_std'] = float(np.std(mfcc[i]))
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        feats['Spectral_Centroid_Mean'] = float(np.mean(spec_centroid))
        rms = librosa.feature.rms(y=y)[0]
        feats['RMS_Mean'] = float(np.mean(rms))
        # parselmouth extras
        pm = compute_parselmouth_features(audio_path)
        feats.update(pm)
        return feats
    except Exception as e:
        return {}


def map_extracted_to_model_features(extracted: dict, feature_names: list):
    mapped = {fn: 0.0 for fn in feature_names}
    if not extracted:
        return mapped
    # simple exact or lowercase matching
    keys = {k.lower(): k for k in extracted.keys()}
    for fn in feature_names:
        low = fn.lower()
        if fn in extracted:
            mapped[fn] = float(extracted[fn])
        elif low in keys:
            mapped[fn] = float(extracted[keys[low]])
        else:
            # try partial match
            for k in extracted.keys():
                if low in k.lower() or k.lower() in low:
                    try:
                        mapped[fn] = float(extracted[k]); break
                    except Exception:
                        continue
    return mapped
