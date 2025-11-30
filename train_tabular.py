
# train_tabular.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
import joblib
import json
from pathlib import Path

def main(features_csv="processed_features.csv", model_out_dir="models"):
    Path(model_out_dir).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(features_csv)
    if 'label' not in df.columns:
        raise ValueError("Label column missing in features CSV.")
    # keep numeric cols only (including label)
    X = df.drop(columns=['label']).select_dtypes(include=[np.number])
    y = df['label'].astype(int)
    feature_names = list(X.columns)
    print(f"Training with {X.shape[1]} features.")

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # XGBoost model
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_s, y_train)

    # Evaluate
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:,1]
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = float('nan')
    print("Accuracy:", acc)
    print("ROC AUC:", auc)
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))

    # Save model, scaler, feature names
    joblib.dump(model, Path(model_out_dir)/"xgb_als_model.pkl")
    joblib.dump(scaler, Path(model_out_dir)/"scaler.pkl")
    joblib.dump(feature_names, Path(model_out_dir)/"feature_names.pkl")
    with open(Path(model_out_dir)/"metrics.json","w") as f:
        json.dump({"accuracy": float(acc), "roc_auc": (None if np.isnan(auc) else float(auc))}, f, indent=2)
    print(f"Saved model+scaler+features to {model_out_dir}")

if __name__ == "__main__":
    main()
