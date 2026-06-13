import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from PIL import Image
import numpy as np
import cv2
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from math import pi
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from streamlit_shap import st_shap

st.set_page_config(page_title="Advanced Ad Predictor", layout="wide")
# Copy and paste this right below your st.set_page_config() line
# Copy and paste this right below your st.set_page_config() line
st.markdown("""
    <style>
    /* Classic True Dark Background */
    .stApp {
        background-color: #121212;
    }
    /* Solid Dark Panels without glowing shadows */
    div[data-testid="stVerticalBlock"] > div {
        background: #1e1e1e;
        border-radius: 8px;
        padding: 20px;
        border: 1px solid #333333;
    }
    /* Clean, crisp off-white text to reduce eye strain */
    h1, h2, h3, p, span, label {
        color: #e0e0e0 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Clean dark input fields */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background-color: #2d2d2d !important;
        border: 1px solid #444444 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_data
def get_image_features(image_file):
    if image_file is None:
        return 0.5, 0.5 # Return default values
    pil_image = Image.open(image_file).convert("RGB")
    cv_image = np.array(pil_image)[:, :, ::-1]
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray_image) / 255.0
    (B, G, R) = cv2.split(cv_image.astype("float"))
    rg, yb = R - G, 0.5 * (R + G) - B
    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)
    colorfulness = np.sqrt((std_rg**2) + (std_yb**2)) + (0.3 * np.sqrt((mean_rg**2) + (mean_yb**2)))
    return brightness, min(1.0, colorfulness / 150.0)

CITY_AUDIENCE_SIZE = {
    'Chennai': 9447960, 'Coimbatore': 1772608, 'Madurai': 1067376,
    'Tiruchirappalli': 787971, 'Thoothukudi': 485654, 'Tirupur': 447736,
    'Salem': 673220, 'Vellore': 557218, 'Erode': 445885,
    'Dindigul': 291339, 'Kanniyakumari': 150448, 'Nagercoil': 104632,
    'Kanchipuram': 243001
}

@st.cache_data
def load_dropdown_data(path):
    try:
        data = pd.read_csv(path)
        return {
            'product_types': sorted(data['product_type'].unique()),
            'campaign_types': sorted(data['campaign_type'].unique()),
            'locations': sorted(data['location'].unique()),
            'currency': sorted(data['currency'].unique())[0] if 'currency' in data.columns else '₹'
        }
    except FileNotFoundError:
        return None

@st.cache_resource
def get_geolocator():
    return Nominatim(user_agent="ad_predictor_app")

def get_city_from_coords(lat, lon):
    try:
        geolocator = get_geolocator()
        location_data = geolocator.reverse((lat, lon), exactly_one=True, language="en")
        address = location_data.raw['address']
        city = address.get('city') or address.get('town') or address.get('county')
        for known_city in CITY_AUDIENCE_SIZE.keys():
            if city and known_city.lower() in city.lower():
                return known_city
        return None
    except Exception:
        return None

# --- Load Files ---
dropdown_data = load_dropdown_data('my_ads_data.csv')
try:
    model = joblib.load('lead_predictor_model_v4.pkl')
except FileNotFoundError:
    model = None

# --- App UI ---
st.title("🚀 V5.0: AI Ad Predictor with Explainable Insights")
st.write("Visually select your campaign's target area and get an instant, explained prediction.")

if dropdown_data and model:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("1. Campaign Details")
        product_type = st.selectbox("Product Type", dropdown_data['product_types'])
        campaign_type = st.selectbox("Campaign Type", dropdown_data['campaign_types'])
        budget = st.number_input(f"Budget (in {dropdown_data['currency']})", min_value=1000, value=40000, step=1000)
        
        st.subheader("Ad Creative")
        ad_text = st.text_area("Ad Text / Caption", "Discover amazing deals! Shop Now!", height=100)
        hashtags = st.text_input("Hashtags (comma-separated)", "#Trending #NewProduct")
        cta_strength = st.slider("Call-to-Action (CTA) Strength", 0.0, 1.0, 0.7, 0.05)

    with col2:
        st.header("2. Geo-Targeting & Image")
        st.write("Click on the map to pin your target city.")
        m = folium.Map(location=[10.7905, 78.7047], zoom_start=8)
        
        map_data = st_folium(m, width=700, height=350)
        
        selected_location = None
        audience_size = None

        if map_data and map_data.get("last_clicked"):
            lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
            city_name = get_city_from_coords(lat, lon)
            
            if city_name and city_name in dropdown_data['locations']:
                selected_location = city_name
                audience_size = CITY_AUDIENCE_SIZE.get(selected_location, 100000)
                st.success(f"✅ Target City Detected: **{selected_location}**")
                st.metric(label="Estimated Audience Size", value=f"{audience_size:,}")
            else:
                st.warning("Could not detect a supported city. Please pin a location in a known city.")
        else:
            st.info("Click on the map to define your target area.")

        uploaded_file = st.file_uploader("Upload Ad Image...", type=["jpg", "jpeg", "png"])
        image_brightness, image_colorfulness = get_image_features(uploaded_file)

        # *** THIS IS THE FIX: Center the image using columns ***
        _ , img_col, _ = st.columns([1, 6, 1]) # Create 3 columns
        with img_col: # Put the image in the middle column
             st.image(uploaded_file if uploaded_file else "https://i.imgur.com/2zz2b83.png", caption="Uploaded Ad Image", use_container_width=True)


    st.markdown("---")
    if st.button("✨ Predict & Explain My Leads!", use_container_width=True):
        if not selected_location or not audience_size:
            st.error("Please define a location on the map before predicting.")
        else:
            input_data = pd.DataFrame({
                'product_type': [product_type], 'campaign_type': [campaign_type],
                'location': [selected_location], 'audience_size': [audience_size],
                'budget': [budget], 'ad_text': [ad_text], 'hashtags': [hashtags],
                'image_brightness': [image_brightness],
                'image_colorfulness': [image_colorfulness], 'cta_strength': [cta_strength],
                'combined_text': [ad_text + ' ' + hashtags]
            })
            
            predicted_leads = int(model.predict(input_data)[0])
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=predicted_leads,
                title={'text': "Predicted Leads"},
                gauge={'axis': {'range': [None, max(200, predicted_leads + 50)]}, 'bar': {'color': "royalblue"}}
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader("💡 Why This Prediction?")
            st.write("This chart shows the factors that influenced the prediction. Red arrows push the prediction higher, and blue arrows push it lower.")

            preprocessor = model.named_steps['preprocessor']
            regressor = model.named_steps['regressor']
            
            features_transformed = preprocessor.transform(input_data)
            features_transformed_dense = features_transformed.toarray()
            
            explainer = shap.TreeExplainer(regressor)
            shap_values = explainer.shap_values(features_transformed_dense)
            
            feature_names = preprocessor.get_feature_names_out()

            plot = shap.force_plot(explainer.expected_value, shap_values[0,:], feature_names=feature_names)
            
            st_shap(plot)

            # ---> REPLACE THE TEXT BOX AT THE BOTTOM WITH THIS <---
            st.markdown("---")
            st.info("""
            💡 **How to read this chart:**
            * **Red arrows** show what is pushing your predicted leads higher. 
            * **Blue arrows** show what is pulling your predicted leads lower. 
            
            🎯 **Quick Fixes based on your Blue Arrows:**
            * **Budget:** If budget is blue, consider increasing your ad spend.
            * **Image Features:** If brightness or colorfulness is blue, try uploading a more vibrant, high-quality image.
            * **CTA Strength:** If CTA is blue, update your caption with stronger action words (e.g., 'Shop Now!').
            """)        

else:
    st.error("Critical files are missing. Please check your setup.")