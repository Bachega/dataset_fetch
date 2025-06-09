import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

def build_preprocessor(X: pd.DataFrame, high_cardinality_threshold: int = 20):
    """
    Builds a robust scikit-learn ColumnTransformer to preprocess mixed data types.
    
    (This function remains unchanged from the previous version)
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
        remainder='passthrough'
    )

    return preprocessor

def preprocess_target(y: pd.Series):
    """
    Preprocesses the target variable.

    - If numeric, it's returned as is.
    - If binary categorical, it's encoded to 0 and 1.
    - If multi-class categorical, it's binarized: the majority class becomes 1
      and all other classes become 0.
    """
    # Check if the target is categorical
    if not (y.dtype == 'object' or pd.api.types.is_categorical_dtype(y)):
        print("Target variable is numeric. No encoding needed.")
        return y.values, None, False

    num_classes = y.nunique()

    # Scenario 1: Binary Target (or less)
    if num_classes <= 2:
        print(f"Target is binary ({num_classes} classes). Applying standard LabelEncoder.")
        label_encoder = LabelEncoder()
        y_processed = label_encoder.fit_transform(y)
        return y_processed, label_encoder, True
    
    # Scenario 2: Multi-class Target
    else:
        # Find the majority class
        majority_class = y.value_counts().idxmax()
        print(f"Target is multi-class ({num_classes} classes). Binarizing with majority class '{majority_class}' as the positive class (1).")
        
        # Binarize the series: True (1) if it's the majority class, False (0) otherwise
        y_processed = (y == majority_class).astype(int)
        
        # Create an info object to store which class was the positive one
        transformation_info = {'positive_class': majority_class}
        
        return y_processed.values, transformation_info, True