import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def preprocess_data(X, y, categorical_indicator):
    one_hot_encoder_used = False
    label_encoder_used = False
    
    if any(categorical_indicator):
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        X_encoded = encoder.fit_transform(X.loc[:, categorical_indicator])
        X_non_categorical = X.loc[:, ~np.array(categorical_indicator)].values
        X_processed = np.hstack((X_non_categorical, X_encoded))
        one_hot_encoder_used = True
    else:
        X_processed = X.values
    
    if y.dtype == 'object' or y.dtype.name == 'category':
        label_encoder = LabelEncoder()
        y_processed = label_encoder.fit_transform(y)
        label_encoder_used = True
    else:
        y_processed = y
        
    return X_processed, y_processed, one_hot_encoder_used, label_encoder_used
