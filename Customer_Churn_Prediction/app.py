import streamlit as st
import pandas as pd
import joblib


# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
    <style>
        .main-title {
            font-size: 2.4rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: #6b7280;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }
        div[data-testid="stMetric"] {
            background-color: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 12px;
            padding: 1rem;
        }
        .result-card {
            padding: 1.5rem;
            border-radius: 14px;
            text-align: center;
            margin-top: 1rem;
        }
        .churn-card {
            background-color: rgba(239, 68, 68, 0.12);
            border: 1px solid rgba(239, 68, 68, 0.4);
        }
        .stay-card {
            background-color: rgba(34, 197, 94, 0.12);
            border: 1px solid rgba(34, 197, 94, 0.4);
        }
        .result-title {
            font-size: 1.4rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
            color: inherit;
        }
        .result-card div {
            color: inherit;
        }
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            height: 3rem;
            font-weight: 600;
            font-size: 1.05rem;
        }
    </style>
""", unsafe_allow_html=True)


# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load(r"Joblib_File/Model.joblib")

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)


# ----------------------------
# Header
# ----------------------------
st.markdown('<div class="main-title">📊 Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Fill in the customer details below to predict the likelihood of churn.</div>',
    unsafe_allow_html=True
)

if not model_loaded:
    st.error(f"⚠️ Could not load the model. Please check the model path.\n\n{model_error}")
    st.stop()


# ----------------------------
# Sidebar - about / info
# ----------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This app uses a trained machine learning pipeline to predict "
        "whether a customer is likely to churn based on their profile "
        "and usage behavior."
    )
    st.divider()
    st.header("🧭 How to use")
    st.write(
        "1. Enter the customer's details in the form.\n"
        "2. Click **Predict Churn**.\n"
        "3. Review the prediction and churn probability."
    )
    st.divider()
    st.caption("Built with Streamlit • Model: scikit-learn Pipeline")


# ----------------------------
# Input form
# ----------------------------
with st.form("churn_form"):

    st.subheader("👤 Customer Profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=100, value=30)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col3:
        tenure = st.number_input("Tenure (months)", min_value=0, value=12)

    st.subheader("📈 Usage & Engagement")
    col4, col5, col6 = st.columns(3)
    with col4:
        usage_frequency = st.number_input("Usage Frequency", min_value=0, value=10)
    with col5:
        support_calls = st.number_input("Support Calls", min_value=0, value=2)
    with col6:
        last_interaction = st.number_input("Last Interaction (days ago)", min_value=0, value=10)

    st.subheader("💳 Subscription & Billing")
    col7, col8, col9 = st.columns(3)
    with col7:
        subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    with col8:
        contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
    with col9:
        payment_delay = st.number_input("Payment Delay (days)", min_value=0, value=5)

    total_spend = st.number_input("Total Spend ($)", min_value=0.0, value=1000.0, step=50.0)

    submitted = st.form_submit_button("🔮 Predict Churn")


# ----------------------------
# Prediction
# ----------------------------
if submitted:

    customer = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Tenure": [tenure],
        "Usage Frequency": [usage_frequency],
        "Support Calls": [support_calls],
        "Payment Delay": [payment_delay],
        "Subscription Type": [subscription_type],
        "Contract Length": [contract_length],
        "Total Spend": [total_spend],
        "Last Interaction": [last_interaction]
    })

    with st.spinner("Analyzing customer data..."):
        prediction = model.predict(customer)[0]
        probabilities = model.predict_proba(customer)[0][1]

    st.divider()
    st.subheader("🎯 Prediction Result")

    res_col, metric_col1, metric_col2 = st.columns([1.4, 1, 1])

    with res_col:
        if prediction == 1:
            st.markdown(
                f"""
                <div class="result-card churn-card">
                    <div class="result-title">⚠️ Likely to Churn</div>
                    <div>This customer shows signs of leaving.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="result-card stay-card">
                    <div class="result-title">✅ Likely to Stay</div>
                    <div>This customer appears loyal.</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with metric_col1:
        st.metric("Churn Probability", f"{probabilities:.1%}")

    with metric_col2:
        st.metric("Retention Probability", f"{(1 - probabilities):.1%}")

    st.write("")
    st.progress(float(probabilities), text=f"Churn risk: {probabilities:.1%}")

    with st.expander("📋 View submitted customer data"):
        st.dataframe(customer, use_container_width=True, hide_index=True)