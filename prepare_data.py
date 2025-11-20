# prepare_data.py
import pandas as pd
from pathlib import Path


def main(infile="synthetic_univariate_10000_ALS.csv", outfile="processed_features.csv"):
    p = Path(infile)
    if not p.exists():
        raise FileNotFoundError(f"{infile} not found. Place dataset in project folder.")

    df = pd.read_csv(p)
    # create binary label column
    if 'label' not in df.columns:
        if 'Diagnosis (ALS)' in df.columns:
            df['label'] = df['Diagnosis (ALS)'].apply(lambda x: 1 if str(x).strip() == '1' else 0)
        else:
            raise ValueError("No 'label' or 'Diagnosis (ALS)' column found.")
    # drop obviously irrelevant columns if present
    for c in ['ID', 'FileName', 'Path', 'Diagnosis (ALS)']:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
    # keep numeric features + label
    numeric = df.select_dtypes(include=['number']).copy()
    if 'label' not in numeric.columns:
        numeric['label'] = df['label'].astype(int)
    numeric.to_csv(outfile, index=False)
    print(f"Saved processed features to {outfile}. Shape: {numeric.shape}")

if __name__ == "__main__":
    main()
