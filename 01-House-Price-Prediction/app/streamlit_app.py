
import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")
st.title("🏠 House Price Predictor")
st.caption("Enter property details to get an estimated price.")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        area      = st.number_input("Area (sq ft)", min_value=500, max_value=20000, value=5000, step=100)
        bedrooms  = st.slider("Bedrooms", 1, 6, 3)
        bathrooms = st.slider("Bathrooms", 1, 4, 2)
        stories   = st.slider("Stories", 1, 4, 2)
        parking   = st.slider("Parking spots", 0, 3, 1)

    with col2:
        mainroad        = st.selectbox("Main road access", ["yes", "no"])
        guestroom       = st.selectbox("Guest room", ["yes", "no"])
        basement        = st.selectbox("Basement", ["yes", "no"])
        hotwaterheating = st.selectbox("Hot water heating", ["yes", "no"])

    with col3:
        airconditioning = st.selectbox("Air conditioning", ["yes", "no"])
        prefarea        = st.selectbox("Preferred area", ["yes", "no"])
        furnishing      = st.selectbox("Furnishing", ["unfurnished", "semi-furnished", "furnished"])
        correct_bias    = st.checkbox("Apply mean calibration (~+2%)", value=False,
                                       help="Use when calibrated totals matter, not MAE")

    submitted = st.form_submit_button("Predict price")

if submitted:
    payload = {
        "area": float(area), "bedrooms": int(bedrooms), "bathrooms": int(bathrooms),
        "stories": int(stories), "mainroad": mainroad, "guestroom": guestroom,
        "basement": basement, "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning, "parking": int(parking),
        "prefarea": prefarea, "furnishingstatus": furnishing,
    }
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        r.raise_for_status()
        result = r.json()
        price = (result["price_inr_bias_corrected"] if correct_bias
                 else result["price_inr"])
        st.success(f"### Estimated price: ₹ {price:,.0f}")
        st.caption(f"Model v{result['model_version']}  •  raw: ₹{result['price_inr']:,.0f}"
                   + ("" if correct_bias else "  •  bias-corrected: "
                      f"₹{result['price_inr_bias_corrected']:,.0f}"))
    except requests.exceptions.RequestException as e:
        st.error(f"API error: {e}")