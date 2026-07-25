import streamlit as st
import pandas as pd
import joblib
import time
from datetime import datetime

# ==================================================================
#  PAGE CONFIG & STATE
# ==================================================================
st.set_page_config(
    page_title="Nestly | Property Valuation",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "history" not in st.session_state:
    st.session_state.history = []

# ==================================================================
#  BRAND THEME — CSS injected for typography and specific highlights
# ==================================================================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #14162b;
    }
    [data-testid="stSidebar"] * {
        color: #e4e5f1 !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #8b8dc0 !important;
    }
    
    /* Brand Logo */
    .brand-container {
        display: flex; align-items: center; gap: 12px;
        padding-bottom: 20px; border-bottom: 1px solid #2a2c4a;
        margin-bottom: 20px;
    }
    .brand-logo {
        width: 38px; height: 38px;
        background: linear-gradient(135deg, #6366f1, #818cf8);
        border-radius: 10px; display: flex; align-items: center; 
        justify-content: center; font-size: 20px;
    }
    .brand-title { font-weight: 800; font-size: 1.25rem; color: #fff; margin:0; line-height: 1.2;}
    .brand-subtitle { font-size: 0.75rem; color: #8b8dc0; text-transform: uppercase; letter-spacing: 1px; margin:0;}

    /* Main Header */
    .app-header {
        display: flex; justify-content: space-between; align-items: flex-end;
        border-bottom: 1px solid #e7e8f2; padding-bottom: 20px; margin-bottom: 30px;
    }
    .app-header h1 { font-size: 1.8rem; font-weight: 800; color: #14162b; margin: 0; }
    .app-header p { font-size: 0.95rem; color: #6b6d8a; margin: 5px 0 0 0; }
    
    .status-badge {
        background: #ecfdf5; color: #059669; border: 1px solid #a7f3d0;
        padding: 6px 16px; border-radius: 50px; font-size: 0.85rem; font-weight: 600;
    }
    .status-badge.mock { background: #fffbeb; color: #d97706; border-color: #fde68a; }

    /* Prediction Result Box */
    .price-result {
        background: linear-gradient(135deg, #14162b 0%, #23255a 100%);
        color: #ffffff; padding: 40px 30px; border-radius: 16px;
        text-align: center; margin-top: 10px; box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    .price-result h2 { color: #a5a6f5; font-size: 0.9rem; letter-spacing: 0.1em; text-transform: uppercase; margin: 0; }
    .price-result .amount { font-size: 3.2rem; font-weight: 800; color: #ffffff; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)


# ==================================================================
#  LOAD ASSETS WITH MOCK FALLBACK (Failsafe)
# ==================================================================
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("joblib_files/rf.joblib")
        le_area = joblib.load("joblib_files/le_area.joblib")
        le_location = joblib.load("joblib_files/le_location.joblib")
        le_property = joblib.load("joblib_files/le_property.joblib")
        area_scaler = joblib.load("joblib_files/area_scaler.joblib")
        bhk_scaler = joblib.load("joblib_files/bhk_scaler.joblib")

        return True, model, le_area, le_location, le_property, area_scaler, bhk_scaler

    except Exception as e:
        st.error(f"❌ Model loading failed: {type(e).__name__}: {e}")
        raise
is_real_model, model, le_area, le_location, le_property, area_scaler, bhk_scaler = load_assets()

# ==================================================================
#  SIDEBAR
# ==================================================================
with st.sidebar:
    st.markdown("""
    <div class="brand-container">
        <div class="brand-logo">🏠</div>
        <div>
            <p class="brand-title">Nestly</p>
            <p class="brand-subtitle">Valuation Engine</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Property Details")
    property_type = st.selectbox("Property Type", le_property.classes_)
    area_type = st.selectbox("Area Type", le_area.classes_)
    location = st.selectbox("Location", le_location.classes_)

    st.markdown("#### Specifications")
    area = st.slider("Area (sq.ft)", 100, 10000, 1000, 50)
    rate = st.slider("Rate Per Sq.ft (₹)", 1000, 50000, 5000, 100)
    bhk = st.slider("BHK", 1, 10, 2, 1)

    st.divider()
    
    predict_clicked = st.button("Run Valuation →", use_container_width=True, type="primary")
    if st.button("Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ==================================================================
#  MAIN HEADER
# ==================================================================
badge_html = '<div class="status-badge">● Model Ready</div>' if is_real_model else '<div class="status-badge mock">⚠️ Mock Mode</div>'

st.markdown(f"""
<div class="app-header">
    <div>
        <h1>Property Valuation</h1>
        <p>Estimate market value using intelligent pricing models</p>
    </div>
    {badge_html}
</div>
""", unsafe_allow_html=True)

# ==================================================================
#  TABS
# ==================================================================
tab_valuation, tab_history, tab_insights = st.tabs(["📍 Valuation", "🕘 History", "📊 Insights"])

# ------------------------------------------------------------------
#  TAB 1 — VALUATION
# ------------------------------------------------------------------
with tab_valuation:
    if not predict_clicked and not st.session_state.history:
        # Empty State
        st.info("🏡 **Ready when you are!** Adjust the property details in the sidebar, then click **Run Valuation**.")
    
    if predict_clicked:
        with st.spinner("Analyzing market patterns..."):
            time.sleep(0.6) # Simulated delay for UI feel

            # Mock or Real Encoding
            try:
                area_scaled = area_scaler.transform([[area]])[0][0] if is_real_model else area
                bhk_scaled = bhk_scaler.transform([[bhk]])[0][0] if is_real_model else bhk
            except:
                area_scaled, bhk_scaled = area, bhk

            X = pd.DataFrame({
                "location": [le_location.transform([location])[0]],
                "rate_per_sqft": [rate],
                "area_in_sqft": [area_scaled],
                "area_type": [le_area.transform([area_type])[0]],
                "property_type": [le_property.transform([property_type])[0]],
                "bhk_type": [bhk_scaled]
            })

            prediction = float(model.predict(X)[0])
            naive_in_cr = (area * rate) / 1e7
            
            # Save to history
            st.session_state.history.append({
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Property Type": property_type,
                "Location": location,
                "Area (sq.ft)": area,
                "BHK": bhk,
                "Base (Cr.)": round(naive_in_cr, 2),
                "Predicted (Cr.)": round(prediction, 2),
            })
            
    if st.session_state.history:
        latest = st.session_state.history[-1]
        pred_val = latest['Predicted (Cr.)']
        base_val = latest['Base (Cr.)']
        
        # Calculate Delta
        delta_pct = ((pred_val - base_val) / base_val * 100) if base_val else 0

        # UI Layout for Results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Property Summary")
            with st.container(border=True):
                st.dataframe(
                    pd.DataFrame({
                        "Attribute": ["Property Type", "Location", "Area (sq.ft)", "Rate / sq.ft (₹)", "BHK"],
                        "Value": [latest['Property Type'], latest['Location'], f"{latest['Area (sq.ft)']:,}", f"{rate:,}", latest['BHK']]
                    }),
                    hide_index=True,
                    use_container_width=True
                )
                
            st.markdown("### Reference Baseline")
            with st.container(border=True):
                st.metric("Base Value (Area × Rate)", f"₹ {base_val:,.2f} Cr")
                st.caption("A simple calculation before ML adjustment.")

        with col2:
            st.markdown("### Valuation Model Output")
            st.markdown(f"""
            <div class="price-result">
                <h2>Estimated Market Value</h2>
                <div class="amount">{pred_val:,.2f} Cr</div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.container(border=True):
                st.metric(
                    label="Model adjustment vs. Baseline", 
                    value=f"{abs(pred_val - base_val):.2f} Cr difference", 
                    delta=f"{delta_pct:.1f}% {'Premium' if delta_pct > 0 else 'Discount'}"
                )

# ------------------------------------------------------------------
#  TAB 2 — HISTORY
# ------------------------------------------------------------------
with tab_history:
    st.markdown("### Session History")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        st.download_button(
            label="⬇️ Download CSV",
            data=hist_df.to_csv(index=False).encode("utf-8"),
            file_name="nestly_history.csv",
            mime="text/csv",
        )
    else:
        st.caption("No valuations run yet.")

# ------------------------------------------------------------------
#  TAB 3 — INSIGHTS
# ------------------------------------------------------------------
with tab_insights:
    st.markdown("### Analytics")
    if len(st.session_state.history) >= 2:
        hist_df = pd.DataFrame(st.session_state.history)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Runs", len(hist_df))
        c2.metric("Avg Estimate", f"{hist_df['Predicted (Cr.)'].mean():,.2f} Cr")
        c3.metric("Max Estimate", f"{hist_df['Predicted (Cr.)'].max():,.2f} Cr")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Valuation Trend**")
            st.line_chart(hist_df.set_index("Time")[["Predicted (Cr.)", "Base (Cr.)"]])
        with col2:
            st.markdown("**By Location**")
            st.bar_chart(hist_df.groupby("Location")["Predicted (Cr.)"].mean())
    else:
        st.caption("Run at least 2 valuations to generate insight charts.")