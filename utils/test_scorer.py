from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from utils import getTrainingScores
from utils import getTPRFPR

def test_scorer(X, y, selected_scorer, selected_norm):  
    clf = None
    if selected_scorer == "LogisticRegression":
        clf = LogisticRegression(random_state=42, n_jobs=-1, max_iter=1000)
    
    scaler = None
    if selected_norm == "None":
        scaler = None
    if selected_norm == "Min-Max Scaling":
        scaler = MinMaxScaler()
    if selected_norm == "Z-Score Scaling":
        scaler = StandardScaler()

    # df_train, df_test = train_test_split(df, test_size=0.5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # X_train = df_train.drop(columns=[class_col]).to_numpy()
    # y_train = df_train[class_col].to_numpy()
    
    if not scaler is None:
        X_train = scaler.fit_transform(X_train)

    calib_clf = CalibratedClassifierCV(clf, cv=3, n_jobs=-1)
    calib_clf.fit(X_train, y_train)

    scores = getTrainingScores(X_train, y_train, 10, clf)[0]
    tprfpr = getTPRFPR(scores)
    clf.fit(X_train, y_train)

    return scores, tprfpr, clf, calib_clf
    
# def test_scorer(df, class_col):
#     with st.container(border=True):
#         st.title("Scorer Test")
#         st.write(
#             "This will test the scorer that will be used by the quantifiers. "
#             "This is important for the QuantifierRecommender project. "
#             "It will run a classifier (LogisticRegression) and will try to generate scores and "
#             "True Positive Rate and False Positive Rate for the dataset. "
#             "It will warn the user if it is successful (or not)."
#         )

#         col1, col2 = st.columns(2)
#         with col1:
#             selected_scorer = st.selectbox("Select the scorer", ['LogisticRegression'], index=0)
#         with col2:
#             selected_norm = st.selectbox("Select normalization method", ['None', 'Min-Max Scaling', 'Z-Score Scaling'], index=2)

#         if st.button("Run Test", key='run-scorer'):
#             try:
#                 scores, tprfpr, clf, calib_clf = run_scorer_test(df, class_col, selected_scorer, selected_norm)
#                 st.success("OK")
                
#                 col3, col4 = st.columns(2)
#                 with col3:
#                     with st.container():
#                         st.caption("Training Scores")
#                         st.dataframe(scores)
#                 with col4:
#                     with st.container():
#                         st.caption("True Positive Rate and False Positive Rate")
#                         st.dataframe(tprfpr)
#             except Exception as e:
#                 st.write(":red[Something went wrong...]")
#                 st.error(f"{e}")