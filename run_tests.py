from utils import test_mfe, test_scorer, preprocess_data

def run_tests(X, y, categorical_indicator):
    X_processed, y_processed, one_hot_encoder_used, label_encoder_used = preprocess_data(X, y, categorical_indicator)

    columns, features = test_mfe(X_processed, y_processed)
    scores, tprfpr, clf, calib_clf = test_scorer(X_processed, y_processed, selected_scorer="LogisticRegression", selected_norm="Z-Score Scaling")
    num_meta_features = len(features)

    # num_meta_features = 111
    # print(columns, features)
    # print(scores, tprfpr, clf, calib_clf)
    
    return X_processed, y_processed, X_processed.shape, num_meta_features, one_hot_encoder_used, label_encoder_used