import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# ---------------------------------------------------
# Custom CSS
# ---------------------------------------------------
st.markdown("""
<style>

.stApp{
    background-color:#111827;
    color:white;
}

h1{
    color:#00C4B4;
    text-align:center;
}

h3{
    color:white;
}

div[data-testid="stMetric"]{
    background-color:#1F2937;
    padding:15px;
    border-radius:12px;
    border:1px solid #374151;
}

.stButton>button{
    width:100%;
    background-color:#00C4B4;
    color:white;
    border:none;
    border-radius:8px;
    padding:12px;
    font-size:17px;
    font-weight:bold;
}

.stButton>button:hover{
    background-color:#009688;
    color:white;
}

.sidebar .sidebar-content{
    background:#1F2937;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Title
# ---------------------------------------------------
st.title("🚗 Used Car Price Prediction")

st.write(
    "Predict the estimated market value of a used car using a Machine Learning model."
)

# ---------------------------------------------------
# Load Model Files
# ---------------------------------------------------
@st.cache_resource
def load_files():
    model = joblib.load("joblib_files/car_price_model.pkl")
    brand_freq = joblib.load("joblib_files/brand_freq_map.pkl")
    brand_price = joblib.load("joblib_files/brand_price_map.pkl")
    clip_bounds = joblib.load("joblib_files/clip_bounds.pkl")
    return model, brand_freq, brand_price, clip_bounds


try:
    model, brand_freq_map, brand_price_map, clip_bounds = load_files()
except Exception as e:
    st.error(f"Unable to load model files.\n\n{e}")
    st.stop()

# ---------------------------------------------------
# Brand List — restricted to brands actually present in the
# training data, since the model has zero examples for anything
# else (e.g. no BMW/Audi/Mercedes/Jaguar/Volvo rows exist in the
# dataset, so predictions for those would just be guesses).
# Number of training rows per brand shown to signal confidence.
# ---------------------------------------------------
BRAND_ROW_COUNTS = {
    "Maruti": 921,
    "Hyundai": 571,
    "Ford": 153,
    "Tata": 130,
    "Volkswagen": 111,
    "Renault": 110,
    "Toyota": 51,
    "Mahindra": 30,
    "Chevrolet": 11,
    "Honda": 7,
}

COMMON_BRANDS = sorted(BRAND_ROW_COUNTS.keys())

# ---------------------------------------------------
# Sidebar — input controls only
# ---------------------------------------------------
st.sidebar.header("🔧 Vehicle Details")

with st.sidebar.form("car_form"):

    brand = st.selectbox("Brand", COMMON_BRANDS)

    year = st.slider(
        "Year of Manufacture",
        2000,
        datetime.now().year,
        2018
    )

    fuel = st.selectbox(
        "Fuel Type",
        ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
    )

    transmission = st.selectbox(
        "Transmission",
        ["Manual", "Automatic"]
    )

    seller_type = st.selectbox(
        "Seller Type",
        ["Dealer", "Individual", "Trustmark Dealer"]
    )

    km_driven = st.number_input(
        "Kilometers Driven",
        min_value=0,
        max_value=500000,
        value=45000,
        step=1000
    )

    engine_cc = st.number_input(
        "Engine (CC)",
        min_value=600,
        max_value=2500,
        value=1200,
        step=50,
        help="Typical mass-market range: 800–2000 CC"
    )

    mileage = st.number_input(
        "Mileage (km/l)",
        min_value=8.0,
        max_value=32.0,
        value=18.0,
        step=0.1
    )

    power = st.number_input(
        "Power (bhp)",
        min_value=30.0,
        max_value=200.0,
        value=85.0,
        step=1.0,
        help="Typical mass-market range: 40–140 bhp"
    )

    predict = st.form_submit_button("🚗 Predict Price")

# ---------------------------------------------------
# Feature Engineering
# ---------------------------------------------------
def prepare_input():

    current_year = datetime.now().year
    car_age = current_year - year
    km_per_year = km_driven / (car_age + 1)

    # Always 0 here — none of the offered brands are premium/luxury,
    # since the model has no training data for those. Kept as a column
    # because the trained model still expects this feature as input.
    premium_brand = 0

    is_automatic = 1 if transmission == "Automatic" else 0

    dealer_sale = 1 if seller_type in [
        "Dealer", "Trustmark Dealer"
    ] else 0

    # brand_freq_map / brand_price_map are expected to be dict-like or
    # pandas Series produced during training
    try:
        default_freq = brand_freq_map.median()
    except AttributeError:
        default_freq = np.median(list(brand_freq_map.values()))

    try:
        default_price = brand_price_map.mean()
    except AttributeError:
        default_price = np.mean(list(brand_price_map.values()))

    brand_freq = brand_freq_map.get(brand, default_freq)
    brand_price_mean = brand_price_map.get(brand, default_price)

    data = pd.DataFrame({
        "Mileage": [mileage],
        "Engine (CC)": [engine_cc],
        "Power": [power],
        "car_age": [car_age],
        "km_per_year": [km_per_year],
        "brand_freq": [brand_freq],
        "brand_price_mean": [brand_price_mean],
        "fuel": [fuel],
        "transmission": [transmission],
        "brand": [brand],
        "premium_brand": [premium_brand],
        "is_automatic": [is_automatic],
        "dealer_sale": [dealer_sale]
    })

    if clip_bounds is not None:
        for col, (low, high) in clip_bounds.items():
            if col in data.columns:
                data[col] = data[col].clip(lower=low, upper=high)

    return data

# ---------------------------------------------------
# Prediction (main panel — result only)
# ---------------------------------------------------
st.subheader("💰 Price Prediction")



if predict:

    
    try:

        with st.spinner("Predicting car price..."):

            input_df = prepare_input()

            log_price = model.predict(input_df)[0]

        
            predicted_price = float(np.expm1(log_price))

            low_price = predicted_price * 0.90
            high_price = predicted_price * 1.10

            st.success("Prediction Generated Successfully!")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="Estimated Price",
                    value=f"₹ {predicted_price:,.0f}"
                )

            with col2:
                st.metric(
                    label="Expected Range",
                    value=f"₹ {low_price:,.0f} - ₹ {high_price:,.0f}"
                )

            st.write("---")
            st.subheader("🚘 Vehicle Summary")

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Brand:** {brand}")
                st.write(f"**Year:** {year}")
                st.write(f"**Fuel:** {fuel}")
                st.write(f"**Transmission:** {transmission}")

            with col2:
                st.write(f"**Seller:** {seller_type}")
                st.write(f"**KM Driven:** {km_driven:,}")
                st.write(f"**Mileage:** {mileage} km/l")
                st.write(f"**Power:** {power} bhp")
                st.write(f"**Engine:** {engine_cc} CC")


    except Exception as e:
        st.error(f"Prediction Failed!\n\n{e}")

else:
    st.info("Fill in vehicle details in the sidebar and click **Predict Price**.")




# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.write("---")

with st.expander("🤖 Model Information"):

    st.write("""
**Algorithm:** Gradient Boosting Regressor

**Features Used**
- Brand
- Fuel
- Transmission
- Mileage
- Engine
- Power
- Car Age
- KM Per Year

Prediction is based on historical selling prices.
""")

st.caption(
    "🚗 This prediction is generated using a Machine Learning model. "
    "Actual market price may vary depending on the car's condition, "
    "service history, accident history, location and market demand."
)