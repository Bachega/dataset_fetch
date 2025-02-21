from utils import test_mfe, test_scorer, preprocess_data

def run_tests(X, y, categorical_indicator):
    
    # import pdb;pdb.set_trace()
    # if X.shape[0]
    
    unique_counts = X.loc[:, categorical_indicator].nunique()
    total_features = unique_counts.sum() - len(unique_counts)
    if total_features > 200:
        # Vai crashar fudidamente, então vaza
        return None

    
    X_processed, y_processed, one_hot_encoder_used, label_encoder_used = preprocess_data(X, y, categorical_indicator)

    if X_processed.shape[0] > 100000 and X_processed.shape[1] > 200:
        return None

    # columns, features = test_mfe(X_processed, y_processed)
    # scores, tprfpr, clf, calib_clf = test_scorer(X_processed, y_processed, selected_scorer="LogisticRegression", selected_norm="Z-Score Scaling")
    # num_meta_features = len(features)

    # num_meta_features = 111
    # print(columns, features)
    # print(scores, tprfpr, clf, calib_clf)
    
    return X_processed, y_processed, X_processed.shape, 0, one_hot_encoder_used, label_encoder_used