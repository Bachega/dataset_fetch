import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

def build_preprocessor(X: pd.DataFrame, high_cardinality_threshold: int = 20):
    """
    Builds a robust scikit-learn ColumnTransformer to preprocess mixed data types.

    This function automatically:
    1. Identifies numeric and categorical features.
    2. Creates a pipeline for numeric data to impute missing values (with median) and scale.
    3. Creates a pipeline for categorical data to impute missing values (with the mode)
       and apply one-hot encoding.
    4. Bundles these steps into a single, reusable preprocessor.

    Args:
        X (pd.DataFrame): The input features dataframe.
        high_cardinality_threshold (int): Numeric columns with unique values below this
                                          threshold will be treated as categorical.

    Returns:
        A scikit-learn ColumnTransformer object ready to be fitted to data.
    """
    # 1. Identify column types with a more robust heuristic
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Treat numeric columns with few unique values as categorical
    for col in numeric_features[:]:  # Iterate on a copy
        if X[col].nunique() <= high_cardinality_threshold:
            categorical_features.append(col)
            numeric_features.remove(col)

    print(f"Found {len(numeric_features)} numeric features: {numeric_features}")
    print(f"Found {len(categorical_features)} categorical features: {categorical_features}")

    # 2. Define the preprocessing pipelines for each data type
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
    ])

    # 3. Combine preprocessing steps with ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  # Keep other columns if any, or use 'drop'
    )

    return preprocessor

def preprocess_target(y: pd.Series):
    """Preprocesses the target variable by label encoding if it is not numeric."""
    if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
        print("Target variable is categorical. Applying LabelEncoder.")
        label_encoder = LabelEncoder()
        y_processed = label_encoder.fit_transform(y)
        return y_processed, label_encoder, True
    else:
        print("Target variable is numeric. No encoding needed.")
        return y.values, None, False

### How to use it in your automation script

# Let's say you've downloaded a dataset into X_df and y_series

# 1. Build the preprocessor tailored to this dataset
# preprocessor = build_preprocessor(X_df)

# 2. Fit the preprocessor on your data and transform it
# This learns the imputing/scaling stats and applies the transformation.
# X_processed = preprocessor.fit_transform(X_df)

# 3. Process the target variable separately
# y_processed, _, label_encoder_used = preprocess_target(y_series)

# 4. Check for excessive dimensions (optional, but good practice)
# if X_processed.shape[0] > 100000 and X_processed.shape[1] > 200:
#     print("Dataset is too large after preprocessing. Skipping.")
#     # return None
# else:
#     print("Preprocessing complete!")
#     # Now X_processed and y_processed are numpy arrays ready for a model.