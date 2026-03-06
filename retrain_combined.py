import os
import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import lightgbm as lgb
import joblib
from utils import extract_audio_features
from tensorflow import keras

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
except Exception:
    Sequential = None


def collect_audio_features(data_dir):
    wav_files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
    rows = []
    labels = []
    names = []
    for fp in wav_files:
        feats = extract_audio_features(fp)
        if feats is None or len(feats) == 0:
            continue
        label = 1 if Path(fp).name.startswith('PZ') else 0
        rows.append(feats)
        labels.append(label)
        names.append(Path(fp).name)
    if not rows:
        return None, None, None
    all_keys = sorted({k for d in rows for k in d.keys()})
    df = pd.DataFrame([{k: d.get(k, 0.0) for k in all_keys} for d in rows])
    return df, labels, names


def load_tabular_for_training(csv_path):
    p = Path(csv_path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if 'label' not in df.columns:
        if 'Diagnosis (ALS)' in df.columns:
            df['label'] = df['Diagnosis (ALS)'].apply(lambda x: 1 if str(x).strip() == '1' else 0)
        else:
            raise ValueError('Tabular CSV missing label')
    X = df.drop(columns=['label']).select_dtypes(include=[np.number])
    y = df['label'].astype(int)
    return X, y


def prepare_combined_dataset(tabular_csv, voice_dir):
    # Tabular synthetic
    tab = load_tabular_for_training(tabular_csv)
    tab_X, tab_y = (pd.DataFrame(), pd.Series(dtype=int)) if tab is None else tab

    # Audio synthetic
    audio_df, audio_y, _ = collect_audio_features(voice_dir)

    if (tab is None or tab_X.empty) and (audio_df is None):
        raise RuntimeError('No training data found (tabular or audio)')

    if audio_df is None:
        X = tab_X
        y = tab_y
    elif tab is None or tab_X.empty:
        X = audio_df
        y = pd.Series(audio_y).astype(int)
    else:
        # concatenate by stacking rows and aligning columns
        tab_X = tab_X.reset_index(drop=True)
        audio_df = audio_df.reset_index(drop=True)
        combined = pd.concat([tab_X, audio_df], ignore_index=True, sort=False).fillna(0.0)
        labels = list(tab_y) + list(audio_y)
        X = combined.select_dtypes(include=[np.number])
        y = pd.Series(labels).astype(int)

    return X, y


def build_dnn(input_dim):
    if Sequential is None:
        return None
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def prepare_test_set(voice_test_dir, minsk_csv):
    # test audio
    audio_df, audio_y, _ = collect_audio_features(voice_test_dir)
    # test tabular
    tab = load_tabular_for_training(minsk_csv)
    if tab is None and audio_df is None:
        raise RuntimeError('No test data available')

    if tab is None or tab[0].empty:
        X_test = audio_df
        y_test = pd.Series(audio_y).astype(int)
    elif audio_df is None:
        X_test = tab[0]
        y_test = tab[1]
    else:
        combined = pd.concat([tab[0].reset_index(drop=True), audio_df.reset_index(drop=True)], ignore_index=True, sort=False).fillna(0.0)
        labels = list(tab[1]) + list(audio_y)
        X_test = combined.select_dtypes(include=[np.number])
        y_test = pd.Series(labels).astype(int)

    return X_test, y_test


def main(tabular_csv='synthetic_univariate_10000_ALS.csv', voice_dir='synthetic_voice_data',
         voice_test_dir='voice_sample/VOC-ALS', minsk_csv='Minsk2020_ALS_dataset.csv', models_dir='models',
         test_size=0.0, random_state=42,
         real_tabular_csv=None, real_voice_dir=None, synthetic_fraction=0.0,
         calibrate=False, calibration_method='logistic', ensemble_method='average', plot_dir=None):
    """Train classifiers on combined tabular and audio features.

    The training set can consist of synthetic data, real data, or a mixture of both.
    Specify ``synthetic_fraction`` to include a proportion of the synthetic set along
    with all available real examples.  Real inputs are supplied via
    ``real_tabular_csv`` and ``real_voice_dir``.  If none are provided, only the
    synthetic data is used.

    A validation split can still be taken from the resulting training dataset by
    setting ``test_size``.  This split is also used for probability calibration if
    ``calibrate`` is True and for training a stacking meta‑model when using
    ``ensemble_method='stack'``.
    """
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    # load synthetic dataset
    print('Loading synthetic dataset...')
    X_syn, y_syn = prepare_combined_dataset(tabular_csv, voice_dir)
    print(f'  synthetic samples: {len(y_syn)} rows, {X_syn.shape[1]} features')

    # optionally load real dataset
    X_real, y_real = None, None
    if real_tabular_csv or real_voice_dir:
        print('Loading real dataset...')
        # mimic prepare_combined_dataset logic but allow one side to be missing
        X_real, y_real = pd.DataFrame(), pd.Series(dtype=int)
        if real_tabular_csv:
            tab = load_tabular_for_training(real_tabular_csv)
            if tab is not None:
                X_real = tab[0]
                y_real = pd.Series(tab[1]).astype(int)
        if real_voice_dir:
            audio_df, audio_y, _ = collect_audio_features(real_voice_dir)
            if audio_df is not None and len(audio_y) > 0:
                if X_real.empty:
                    X_real = audio_df
                    y_real = pd.Series(audio_y).astype(int)
                else:
                    audio_df = audio_df.reset_index(drop=True)
                    X_real = pd.concat([X_real.reset_index(drop=True), audio_df],
                                        ignore_index=True, sort=False).fillna(0.0)
                    y_real = pd.Series(list(y_real) + list(audio_y)).astype(int)
        if X_real is None or len(y_real) == 0 or X_real.empty:
            print('  WARNING: real dataset did not contain any rows')
            X_real, y_real = None, None
        else:
            X_real = X_real.select_dtypes(include=[np.number])
            print(f'  real samples: {len(y_real)} rows, {X_real.shape[1]} features')

    # mix according to fraction
    if X_real is not None:
        # determine how much synthetic data to include
        if synthetic_fraction > 0 and X_syn is not None:
            n_take = int(len(X_syn) * synthetic_fraction)
            rng = np.random.RandomState(random_state)
            idx = rng.choice(len(X_syn), n_take, replace=False)
            X_syn_sub = X_syn.iloc[idx].reset_index(drop=True)
            y_syn_sub = pd.Series(y_syn).iloc[idx].reset_index(drop=True)
        else:
            X_syn_sub, y_syn_sub = X_syn, pd.Series(y_syn)

        if X_syn_sub is None or len(X_syn_sub) == 0:
            X, y = X_real, y_real
        else:
            X = pd.concat([X_real.reset_index(drop=True), X_syn_sub], ignore_index=True, sort=False).fillna(0.0)
            y = pd.Series(list(y_real) + list(y_syn_sub)).astype(int)
            X = X.select_dtypes(include=[np.number])
            print(f'  combined real+synthetic samples: {len(y)}')
    else:
        X, y = X_syn, pd.Series(y_syn).astype(int)
        print('  using synthetic data only')

    print(f'Training dataset: {len(y)} samples, {X.shape[1]} features')

    # create a train/validation split if requested
    X_train, y_train = X, y
    X_val, y_val = None, None
    if test_size > 0.0:
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        print(f'  split: {len(y_train)} train / {len(y_val)} val')

    # remove any constant or near-constant columns from training/validation
    const_cols = X_train.columns[X_train.nunique() <= 1].tolist()
    lowvar = X_train.columns[X_train.var() < 1e-8].tolist()
    drop_cols = sorted(set(const_cols + lowvar))
    if drop_cols:
        print(f'Removing {len(drop_cols)} low-variance feature(s): {drop_cols[:5]}{"..." if len(drop_cols)>5 else ""}')
        X_train = X_train.drop(columns=drop_cols)
        if X_val is not None:
            X_val = X_val.drop(columns=drop_cols, errors='ignore')
    # scaling
    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train.values)
    if X_val is not None:
        Xs_val = scaler.transform(X_val.values)

    # convenience helper to wrap arrays in DataFrame for prediction
    def dfify(arr):
        return pd.DataFrame(arr, columns=X.columns)

    # Train XGBoost
    print('Training XGBoost...')
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        eval_metric='logloss',
        random_state=random_state,
        n_jobs=-1
    )
    if X_val is not None:
        # some versions of XGBoost wrapper don't support early_stopping_rounds
        xgb.fit(Xs_train, y_train,
                eval_set=[(Xs_val, y_val)],
                verbose=False)
    else:
        xgb.fit(Xs_train, y_train)

    # Train LightGBM
    print('Training LightGBM...')
    lgbm = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=random_state,
        verbosity=-1  # suppress log messages and warnings about splits
    )
    # use all training data; validation is not supported by older versions
    lgbm.fit(Xs_train, y_train)

    # Train DNN
    dnn = None
    if Sequential is not None:
        print('Training DNN (TensorFlow)...')
        dnn = build_dnn(Xs_train.shape[1])
        callbacks = []
        if X_val is not None:
            from tensorflow.keras.callbacks import EarlyStopping
            callbacks.append(EarlyStopping(patience=5, restore_best_weights=True))
        dnn.fit(
            Xs_train, y_train,
            epochs=100,
            batch_size=64,
            validation_data=(Xs_val, y_val) if X_val is not None else None,
            callbacks=callbacks,
            verbose=1
        )
    else:
        print('TensorFlow not available; skipping DNN training')

    # probability calibration on validation set
    calibrators = {}
    if calibrate and X_val is not None:
        print(f'Calibrating probability outputs using {calibration_method} method on validation set')
        if calibration_method == 'logistic':
            from sklearn.linear_model import LogisticRegression
            for name, clf in [('xgb', xgb), ('lightgbm', lgbm)]:
                probs_val = clf.predict_proba(Xs_val)[:,1]
                lr = LogisticRegression(random_state=random_state, max_iter=1000)
                lr.fit(probs_val.reshape(-1,1), y_val)
                calibrators[name] = lr
            if dnn is not None:
                probs_val = dnn.predict(Xs_val).ravel()
                lr = LogisticRegression(random_state=random_state, max_iter=1000)
                lr.fit(probs_val.reshape(-1,1), y_val)
                calibrators['dnn'] = lr
        elif calibration_method == 'isotonic':
            from sklearn.isotonic import IsotonicRegression
            for name, clf in [('xgb', xgb), ('lightgbm', lgbm)]:
                probs_val = clf.predict_proba(Xs_val)[:,1]
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(probs_val, y_val)
                calibrators[name] = iso
            if dnn is not None:
                probs_val = dnn.predict(Xs_val).ravel()
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(probs_val, y_val)
                calibrators['dnn'] = iso
        else:
            raise ValueError(f'Unknown calibration method: {calibration_method}')

    # prepare meta-model for stacking if requested
    meta = None
    if ensemble_method == 'stack' and X_val is not None:
        print('Training stacking meta‑model (logistic regression) on validation set')
        from sklearn.linear_model import LogisticRegression
        # build validation-level feature matrix
        val_feats = []
        val_feats.append(xgb.predict_proba(Xs_val)[:,1])
        val_feats.append(lgbm.predict_proba(Xs_val)[:,1])
        if dnn is not None:
            val_feats.append(dnn.predict(Xs_val).ravel())
        meta_X = np.vstack(val_feats).T
        meta = LogisticRegression(random_state=random_state)
        meta.fit(meta_X, y_val)

    # Prepare test set from real data
    print('Preparing test dataset (voice_sample + Minsk)...')
    X_test, y_test = prepare_test_set(voice_test_dir, minsk_csv)
    X_test = X_test.reindex(columns=X.columns, fill_value=0.0)
    # drop any constant cols identified during training
    if 'const_cols' in locals() and const_cols:
        X_test = X_test.drop(columns=[c for c in const_cols if c in X_test.columns], errors='ignore')
    Xs_test = scaler.transform(X_test.values)

    # helper for ROC plotting and threshold tuning
    def find_best_threshold(y_true, probs):
        best_thr = 0.5
        best_score = -1
        from sklearn.metrics import f1_score
        for thr in np.linspace(0, 1, 101):
            score = f1_score(y_true, probs >= thr)
            if score > best_score:
                best_score = score
                best_thr = thr
        return best_thr

    # helper to apply calibrator (works with both LogisticRegression and IsotonicRegression)
    def apply_calibrator(calibrator, probs):
        if hasattr(calibrator, 'predict_proba'):
            # LogisticRegression - has predict_proba
            return calibrator.predict_proba(probs.reshape(-1,1))[:,1]
        else:
            # IsotonicRegression - has predict or transform
            return calibrator.predict(probs)

    thresholds = {}
    if X_val is not None:
        print('Finding optimal thresholds on validation set')
        # compute probabilities for each model on validation
        val_probs = {}
        val_probs['xgb'] = xgb.predict_proba(pd.DataFrame(Xs_val, columns=X_train.columns))[:,1]
        val_probs['lightgbm'] = lgbm.predict_proba(pd.DataFrame(Xs_val, columns=X_train.columns))[:,1]
        if dnn is not None:
            val_probs['dnn'] = dnn.predict(Xs_val).ravel()
        # apply calibrators if available
        if 'calibrators' in locals():
            for nm in list(val_probs.keys()):
                if nm in calibrators:
                    val_probs[nm] = apply_calibrator(calibrators[nm], val_probs[nm])
        for name, probs in val_probs.items():
            thr = find_best_threshold(y_val, probs)
            thresholds[name] = thr
        # ensemble threshold
        if ensemble_method == 'stack' and meta is not None:
            ens_val = meta.predict_proba(
                np.vstack([val_probs.get('xgb'), val_probs.get('lightgbm')] +
                          ([val_probs.get('dnn')] if 'dnn' in val_probs else [])
                         ).T
            )[:,1]
        else:
            pl = [val_probs.get('xgb'), val_probs.get('lightgbm')]
            if 'dnn' in val_probs:
                pl.append(val_probs.get('dnn'))
            ens_val = np.mean(np.vstack(pl), axis=0)
        thresholds['ensemble'] = find_best_threshold(y_val, ens_val)
        print('Validation thresholds:', thresholds)
    
    # if requested, plot ROC curve
    if plot_dir:
        import os
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
        os.makedirs(plot_dir, exist_ok=True)
        print('Plotting ROC curves to', plot_dir)
        for name, probs in [('xgb', xgb.predict_proba(pd.DataFrame(Xs_test, columns=X_train.columns))[:,1]),
                            ('lightgbm', lgbm.predict_proba(pd.DataFrame(Xs_test, columns=X_train.columns))[:,1])]:
            if name == 'dnn' and dnn is None:
                continue
            if name == 'dnn':
                probs = dnn.predict(Xs_test).ravel()
            fpr, tpr, _ = roc_curve(y_test, probs)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})')
            plt.plot([0,1],[0,1],'--',color='grey')
            plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC {name}')
            plt.legend(loc='lower right')
            plt.savefig(os.path.join(plot_dir, f'roc_{name}.png'))
            plt.close()

    # Evaluate
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

    print('Evaluating models on test set...')
    results = {}

    # helper to predict using DataFrame (keeps feature names for LGBM)
    Xs_test_df = pd.DataFrame(Xs_test, columns=X.columns)

    # XGBoost
    y_proba_x = xgb.predict_proba(Xs_test_df)[:, 1]
    if 'calibrators' in locals() and 'xgb' in calibrators:
        y_proba_x = apply_calibrator(calibrators['xgb'], y_proba_x)
    thr_x = thresholds.get('xgb', 0.5)
    y_pred_x = (y_proba_x >= thr_x).astype(int)
    results['xgb'] = {
        'accuracy': float(accuracy_score(y_test, y_pred_x)),
        'roc_auc': float(roc_auc_score(y_test, y_proba_x)) if len(np.unique(y_test))>1 else None,
        'threshold': thr_x,
        'confusion': confusion_matrix(y_test, y_pred_x).tolist(),
        'report': classification_report(y_test, y_pred_x, output_dict=True)
    }

    # LightGBM
    y_proba_l = lgbm.predict_proba(Xs_test_df)[:, 1]
    if 'calibrators' in locals() and 'lightgbm' in calibrators:
        y_proba_l = apply_calibrator(calibrators['lightgbm'], y_proba_l)
    thr_l = thresholds.get('lightgbm', 0.5)
    y_pred_l = (y_proba_l >= thr_l).astype(int)
    results['lightgbm'] = {
        'accuracy': float(accuracy_score(y_test, y_pred_l)),
        'roc_auc': float(roc_auc_score(y_test, y_proba_l)) if len(np.unique(y_test))>1 else None,
        'threshold': thr_l,
        'confusion': confusion_matrix(y_test, y_pred_l).tolist(),
        'report': classification_report(y_test, y_pred_l, output_dict=True)
    }

    # DNN
    if dnn is not None:
        y_proba_d = dnn.predict(Xs_test_df.values).ravel()
        if 'calibrators' in locals() and 'dnn' in calibrators:
            y_proba_d = apply_calibrator(calibrators['dnn'], y_proba_d)
        thr_d = thresholds.get('dnn', 0.5)
        y_pred_d = (y_proba_d >= thr_d).astype(int)
        results['dnn'] = {
            'accuracy': float(accuracy_score(y_test, y_pred_d)),
            'roc_auc': float(roc_auc_score(y_test, y_proba_d)) if len(np.unique(y_test))>1 else None,
            'threshold': thr_d,
            'confusion': confusion_matrix(y_test, y_pred_d).tolist(),
            'report': classification_report(y_test, y_pred_d, output_dict=True)
        }

    # Ensemble
    if ensemble_method == 'stack' and meta is not None:
        # construct meta features
        test_feats = [y_proba_x, y_proba_l]
        if dnn is not None:
            test_feats.append(y_proba_d)
        meta_X_test = np.vstack(test_feats).T
        ens_proba = meta.predict_proba(meta_X_test)[:,1]
    else:
        # average (unweighted) probabilities
        prob_list = [y_proba_x, y_proba_l]
        if dnn is not None:
            prob_list.append(y_proba_d)
        ens_proba = np.mean(np.vstack(prob_list), axis=0)
    thr_e = thresholds.get('ensemble', 0.5)
    y_pred_ens = (ens_proba >= thr_e).astype(int)
    results['ensemble'] = {
        'accuracy': float(accuracy_score(y_test, y_pred_ens)),
        'roc_auc': float(roc_auc_score(y_test, ens_proba)) if len(np.unique(y_test))>1 else None,
        'threshold': thr_e,
        'confusion': confusion_matrix(y_test, y_pred_ens).tolist(),
        'report': classification_report(y_test, y_pred_ens, output_dict=True)
    }

    print('Results:', json.dumps(results, indent=2))

    # Save artifacts
    print('Saving artifacts to', models_dir)
    joblib.dump(xgb, Path(models_dir)/'xgb_als_model.pkl')
    joblib.dump(lgbm, Path(models_dir)/'lgbm_als_model.pkl')
    if dnn is not None:
        # prefer new Keras format unless user explicitly wants HDF5
        dnn.save(Path(models_dir)/'dnn_als_model.keras')
    joblib.dump(scaler, Path(models_dir)/'scaler.pkl')
    feature_names = list(X.columns)
    joblib.dump(feature_names, Path(models_dir)/'feature_names.pkl')
    with open(Path(models_dir)/'metrics.json', 'w') as fh:
        json.dump(
            {'evaluation': results, 'n_samples': int(len(y)), 'n_features': int(X.shape[1])},
            fh,
            indent=2
        )

    print('Saved models and metrics')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Retrain multimodal ALS models')
    parser.add_argument('--tabular_csv', default='synthetic_univariate_10000_ALS.csv', help='Path to synthetic tabular training CSV')
    parser.add_argument('--voice_dir', default='synthetic_voice_data', help='Directory with synthetic training audio files')
    parser.add_argument('--voice_test_dir', default='voice_sample/VOC-ALS', help='Directory with test audio files')
    parser.add_argument('--minsk_csv', default='Minsk2020_ALS_dataset.csv', help='CSV with test tabular data')
    parser.add_argument('--models_dir', default='models', help='Output directory for models/metrics')
    parser.add_argument('--val_split', type=float, default=0.0, help='Fraction of training data to hold out for validation')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    parser.add_argument('--real_tabular_csv', default=None, help='Real-world tabular CSV to include in training')
    parser.add_argument('--real_voice_dir', default=None, help='Directory of real-world training audio samples')
    parser.add_argument('--synthetic_fraction', type=float, default=0.0, help='Fraction of synthetic data to mix with real data (e.g. 0.5)')
    parser.add_argument('--calibrate', action='store_true', help='Calibrate tree model probabilities using validation split')
    parser.add_argument('--calibration_method', choices=['logistic', 'isotonic'], default='logistic', help='Method for probability calibration (Platt=logistic, or isotonic)')
    parser.add_argument('--ensemble_method', choices=['average','stack'], default='average', help='How to combine model probabilities')
    parser.add_argument('--plot_dir', default=None, help='Directory where ROC curves will be saved if provided')
    args = parser.parse_args()

    main(
        tabular_csv=args.tabular_csv,
        voice_dir=args.voice_dir,
        voice_test_dir=args.voice_test_dir,
        minsk_csv=args.minsk_csv,
        models_dir=args.models_dir,
        test_size=args.val_split,
        random_state=args.random_state,
        real_tabular_csv=args.real_tabular_csv,
        real_voice_dir=args.real_voice_dir,
        synthetic_fraction=args.synthetic_fraction,
        calibrate=args.calibrate,
        calibration_method=args.calibration_method,
        ensemble_method=args.ensemble_method,
        plot_dir=args.plot_dir
    )
