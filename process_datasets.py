import pandas as pd
import os
import re

from utils import build_preprocessor, preprocess_target

def process_datasets(
    data_folder: str = "./datasets", 
    summary_csv: str = "./uciml_summary.csv",
    processed_folder: str = "./processed_datasets"
):
    """
    For each CSV in data_folder:
      - extract UCI ID,
      - lookup target_col in summary,
      - load X and y,
      - apply preprocessing,
      - assemble a DataFrame with X_processed, append y_processed as 'target',
      - save to processed_folder.
    """
    summary = pd.read_csv(summary_csv, dtype={'uci_id': str, 'target_col': str})
    os.makedirs(processed_folder, exist_ok=True)
    pattern = re.compile(r"^(\d+)_")

    i = 0

    for fname in os.listdir(data_folder):
        if not fname.lower().endswith('.csv'):
            continue

        match = pattern.match(fname)
        if not match:
            print(f"Skipping unmatched file: {fname}")
            continue

        uci_id = match.group(1)
        row = summary.loc[summary['uci_id'] == uci_id]
        if row.empty:
            print(f"No summary entry for ID {uci_id}")
            continue

        target_col_str = row.iloc[0]['target_col']
        y_col = target_col_str.split(';')[0] if ';' in target_col_str else target_col_str

        df = pd.read_csv(os.path.join(data_folder, fname))
        if y_col not in df.columns:
            print(f"Target column {y_col} not in {fname}")
            continue

        X_df = df.drop(columns=[y_col])
        y_series = df[y_col]

        # 1. Preprocess features
        preprocessor = build_preprocessor(X_df)
        X_processed = preprocessor.fit_transform(X_df)

        # 2. Preprocess target
        y_processed, transformation_info, encoded_flag = preprocess_target(y_series)

        # 3. Build processed DataFrame
        try:
            feature_names = preprocessor.get_feature_names_out(X_df.columns)
        except Exception:
            feature_names = [f"f{i}" for i in range(X_processed.shape[1])]
        proc_df = pd.DataFrame(X_processed, columns=feature_names)
        proc_df['target'] = y_processed

        # 4. Dimension check
        if proc_df.shape[0] > 100000 and proc_df.shape[1] > 200:
            print(f"{fname}: too large after preprocessing, skipping.")
            continue

        # 5. Save processed file
        out_fname = os.path.join(processed_folder, fname)
        proc_df.to_csv(out_fname, index=False)
        print(f"Saved processed dataset to {out_fname} (shape: {proc_df.shape})")

        if i > 3: break

process_datasets()