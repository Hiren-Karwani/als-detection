# synthetic_data.py
import pandas as pd
import numpy as np
from pathlib import Path

def main(orig="synthetic_univariate_10000_ALS.csv", out="synthetic_ALS.csv", n_per_class=5000):
    p = Path(orig)
    if not p.exists():
        raise FileNotFoundError("Original dataset not found in working directory.")
    data = pd.read_csv(p)
    # label
    if 'Diagnosis (ALS)' in data.columns:
        data['label'] = data['Diagnosis (ALS)'].apply(lambda x: 1 if str(x).strip() == '1' else 0)
    elif 'label' not in data.columns:
        raise ValueError("No label column found.")
    # split classes
    pos = data[data['label']==1]
    neg = data[data['label']==0]
    def gen(class_df, n):
        rows = {}
        for col in class_df.columns:
            if np.issubdtype(class_df[col].dtype, np.number):
                vals = class_df[col].dropna().to_numpy()
                if vals.size == 0:
                    m = np.nan
                    s = 0.0
                else:
                    m = float(np.nanmean(vals))
                    # avoid computing std when there is <=1 observation to prevent
                    # runtime warnings and NaN std. Use ddof=0 and guard by count.
                    if vals.size <= 1:
                        s = 0.0
                    else:
                        s = float(np.nanstd(vals, ddof=0))
                if s == 0 or np.isnan(s):
                    rows[col] = np.full(n, m)
                else:
                    rows[col] = np.random.normal(m, s, n)
            else:
                vals = class_df[col].dropna().unique()
                if len(vals)==0:
                    rows[col] = np.array([None]*n)
                else:
                    probs = class_df[col].value_counts(normalize=True)
                    rows[col] = np.random.choice(probs.index, size=n, p=probs.values)
        return pd.DataFrame(rows)
    pos_synth = gen(pos, n_per_class)
    neg_synth = gen(neg, n_per_class)
    outdf = pd.concat([pos_synth, neg_synth], ignore_index=True)
    outdf.to_csv(out, index=False)
    print(f"Synthetic saved to {out} shape {outdf.shape}")

if __name__ == "__main__":
    main()