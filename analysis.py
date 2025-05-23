import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "./OLD_DATASETS/NEW_DATASETS/"

results = []

for filename in os.listdir(DATA_PATH):
    if not filename.endswith(".csv"):
        continue

    file_path = os.path.join(DATA_PATH, filename)
    print(f"Processing {filename}...")

    try:
        df = pd.read_csv(file_path)

        # Heuristic: last column is the target
        target_col = df.columns[-1]

        # Drop rows with NA
        df = df.dropna()

        # # Encode target if categorical
        # if df[target_col].dtype == object or df[target_col].nunique() <= 10:
        #     le = LabelEncoder()
        #     df[target_col] = le.fit_transform(df[target_col])
        # else:
        #     raise ValueError("Target column does not look categorical.")

        # Prepare features and labels
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Convert categorical variables
        # X = pd.get_dummies(X, drop_first=True)

        # Must have at least 2 classes
        # if len(set(y)) < 2:
        #     raise ValueError("Less than 2 classes in target.")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        # Train classifier
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)

        # Predict probabilities
        y_proba = clf.predict_proba(X_test)

        # Handle binary and multiclass
        if y_proba.shape[1] == 2:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_test, y_proba, multi_class='ovo')

        results.append({"Dataset": filename, "AUC": auc})

    except Exception as e:
        results.append({"Dataset": filename, "Error": str(e)})

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("auc_summary_new.csv", index=False)

print("AUC calculation complete. Results saved to auc_summary.csv.")
