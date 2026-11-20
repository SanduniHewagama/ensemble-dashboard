import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier

st.set_page_config(page_title="Ensemble Learning Dashboard", layout="wide")
st.title("Ensemble Learning Dashboard: Telecom Churn")
st.write("Bagging vs Boosting vs Stacking comparison")

results = pd.read_csv("results.csv")
model = XGBClassifier()
model.load_model("model.json")
cols = joblib.load("columns.pkl")
defaults = joblib.load("defaults.pkl")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Churn Comparison", "Feature Importance", "Predict Churn",
     "Patient Data", "Hybrid Models"])

with tab1:
    st.subheader("Results table")
    st.dataframe(results, use_container_width=True)
    metric = st.selectbox("Select metric", ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
    st.bar_chart(results.set_index("Model")[metric])

with tab2:
    imp = pd.read_csv("importance.csv", index_col=0)
    st.bar_chart(imp)

with tab3:
    st.subheader("Enter customer details")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.slider("Monthly Charges", 18, 120, 70)
    total = st.number_input("Total Charges", 0.0, 9000.0, float(tenure * monthly))

    if st.button("Predict"):
        row = defaults.copy()
        row["tenure"] = tenure
        row["MonthlyCharges"] = monthly
        row["TotalCharges"] = total
        data = pd.DataFrame([row])[cols]
        p = model.predict_proba(data)[0][1]
        st.metric("Churn probability", f"{p*100:.1f}%")
        if p > 0.5:
            st.error("High risk: customer is likely to leave")
        else:
            st.success("Low risk: customer is likely to stay")

with tab4:
    st.subheader("Patient (Diabetes) Data: Ensemble Comparison")
    pr = pd.read_csv("patient_results.csv")
    st.dataframe(pr, use_container_width=True)
    m2 = st.selectbox("Metric", ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], key="pm")
    st.bar_chart(pr.set_index("Model")[m2])

with tab5:
    st.subheader("Hybrid Models (RF + ANN)")
    hr = pd.read_csv("hybrid_results.csv")
    st.dataframe(hr, use_container_width=True)
    st.bar_chart(hr.set_index("Model")["RMSE"])
    st.caption("Innovation: SHAP explainability (see notebook Part 4)")

