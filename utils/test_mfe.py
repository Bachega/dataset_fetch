import pandas as pd
import numpy as np

from pymfe.mfe import MFE

def check_convert_data_type(data):
    if isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    return data

def extract_meta_features(mfe, X, y=None):
    X = check_convert_data_type(X)

    if y is None:
        mfe.fit(X, suppress_warnings=True)
    else:
        y = check_convert_data_type(y)
        mfe.fit(X, y, suppress_warnings=True)

    columns_and_features = mfe.extract(cat_cols="auto", suppress_warnings=True, verbose=0)
    columns = columns_and_features[0]
    features = columns_and_features[1]
    
    features = np.nan_to_num(features).tolist()
    for i in range(0, len(features)):
        if features[i] > np.finfo(np.float32).max:
            features[i] = np.finfo(np.float32).max

    return columns, features

def test_mfe(X, y):
    mfe = MFE()

    # y = df[class_col].to_numpy()
    # X = df.drop(columns=[class_col]).to_numpy()

    return extract_meta_features(mfe, X, y)

# def test_mfe(df, class_col):
#     with st.container(border=True):
#         st.title("Meta-Feature Extraction Test")
#         st.write("If you see a dataset below with a single instance, then everything is fine")
#         if st.button("Run Test"):
#             try:
#                 columns, features = run_test(df, class_col)
#                 mfe_df = pd.DataFrame(columns=columns, data=[features])

#                 st.success("OK")
#                 st.write(f"Number of features: {len(columns)}")
#                 st.dataframe(mfe_df)
#             except Exception as e:
#                 st.write(":red[Something went wrong...]")
#                 st.error(f"{e}")