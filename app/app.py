import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from PIL import Image
import easyocr
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="Meta Ad Predictor", layout="wide")

# --- Custom Styling ---
st.markdown("""
<style>
    /* Main App Styling */
    .main {
        background-color: #f0f2f6;
    }
    /* Sidebar Styling */
    .st-emotion-cache-16txtl3 {
        background-color: #FFFFFF;
    }
    /* Button Styling */
    .stButton>button {
        background-color: #0068c9;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 16px;
    }
    .stButton>button:hover {
        background-color: #0055a3;
    }
    /* Header Styling */
    h1, h2, h3 {
        color: #003b73;
    }
    /* Info Box Styling */
    .st-emotion-cache-1wivap2 {
        background-color: #e6f3ff;
    }
</style>
""", unsafe_allow_html=True)


# --- Load The Trained AI Brain ---
# Cache the model loading to improve performance
@st.cache_resource
def load_model():
    try:
        model = joblib.load('ad_robust_model.pkl')
        return model
    except FileNotFoundError:
        return None

model = load_model()
if model is None:
     st.error("Model file not found! Please run the `train_robust_model.py` script first.")

# --- OCR Model Loading ---
# Cache the OCR reader to load it only once
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['en'])

reader = load_ocr_reader()


# --- Helper Functions ---
def extract_text_from_image(image_bytes):
    """Reads text from an image using EasyOCR."""
    try:
        image_array = np.array(Image.open(image_bytes))
        result = reader.readtext(image_array, detail=0, paragraph=True)
        return " ".join(result)
    except Exception as e:
        return f"Could not read text from image. Error: {e}"

def analyze_image_text(text):
    """Provides simple suggestions based on the text found in an image."""
    suggestions = []
    text_lower = text.lower()
    if not text:
        suggestions.append("⚠️ **Suggestion:** No text was detected. Consider adding a clear headline or call to action on your image.")
    else:
        suggestions.append("✅ **Text Detected:** The AI found the following text on your image.")
        if len(text.split()) > 20:
            suggestions.append("⚠️ **Suggestion:** The text on the image is quite long. For social media, shorter, punchier text is often more effective.")
        if 'sale' in text_lower or 'offer' in text_lower or 'discount' in text_lower or '%' in text_lower:
            suggestions.append("👍 **Good Practice:** Including words like 'Sale' or 'Offer' directly on the image can effectively grab attention.")
    return suggestions

def get_caption_suggestions(caption):
    """Analyzes the ad caption and provides AI-driven suggestions."""
    suggestions = []
    caption_lower = caption.lower()
    
    # Define keywords the AI has learned are powerful
    power_words = ['sale', 'giveaway', 'offer', 'discount', '%', 'new', 'win', 'free', 'limited', 'shop now', 'link in bio']
    
    # Check for missing power words
    found_words = [word for word in power_words if word in caption_lower]
    if not found_words:
        suggestions.append("💡 **AI Suggestion:** Your caption could be stronger. Try adding powerful keywords like **'Sale', 'Giveaway', 'New',** or a clear call to action like **'Shop Now'** to create urgency.")
    
    # Check for a question to drive engagement
    if '?' not in caption:
        suggestions.append("💡 **AI Suggestion:** Engage your audience by asking a question! For example, 'What's your favorite color?' or 'Who would you share this with?'. Questions encourage comments.")
        
    return suggestions


# --- Application Layout ---
st.title("🚀 Meta Ad Performance Predictor (Final Version)")
st.write("Predict engagement for your Instagram or Facebook ad campaign with our most advanced AI.")

# --- Two-Column Layout ---
col1, col2 = st.columns([1, 2])

# --- COLUMN 1: User Inputs ---
with col1:
    st.header("Campaign & Account Details")
    
    platform = st.selectbox("Platform", ["Instagram", "Facebook"])
    category = st.selectbox("Product Category", ["Cosmetic", "Clothing", "Food", "Electronic", "Books"])
    target_age = st.slider("Target Audience Age", 13, 65, 25)
    budget = st.slider("Budget (₹)", min_value=800, max_value=100000, value=5000, step=100)
    followers = st.select_slider(
        "Follower Count (Approx.)",
        options=["1k", "5k", "10k", "50k", "100k", "500k", "1M+"],
        value="50k"
    )
    posting_time = st.selectbox("Posting Time", ["Morning", "Afternoon", "Evening", "Night"])
    
    # Ad Creative Section
    st.header("Your Ad Creative")
    ad_copy = st.text_area("Enter the caption and hashtags", "The ultimate glow duo. ✨ Shop our new collection for that summer glow! #Cosmetic #Beauty #Skincare", height=150)
    ad_type = st.selectbox("Ad Type", ["Image", "Video", "Carousel"])

    uploaded_file = st.file_uploader("Upload Ad Creative", type=["png", "jpg", "jpeg"])
    
    # --- New Image Analysis Section in Column 1 ---
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Ad Creative", use_column_width=True)
        with st.spinner('Analyzing image text...'):
            extracted_text = extract_text_from_image(uploaded_file)
            st.subheader("Image Text Analysis")
            suggestions = analyze_image_text(extracted_text)
            for suggestion in suggestions:
                st.markdown(suggestion)
            if extracted_text:
                st.text_area("Detected Text:", extracted_text, height=100, disabled=True)


# --- COLUMN 2: Prediction Output ---
with col2:
    st.header("Prediction & Insights")
    
    if st.button("✨ Predict Engagement"):
        if model is not None:
            # 1. Create a DataFrame from user inputs
            input_data = pd.DataFrame({
                'Platform': [platform],
                'Category': [category],
                'Target_Age': [target_age],
                'Budget': [budget],
                'Followers': [followers],
                'Posting_Time': [posting_time],
                'Caption': [ad_copy],
                'Ad_Type': [ad_type]
            })

            # 2. Make the prediction
            try:
                prediction = model.predict(input_data)
                predicted_score = int(prediction[0])

                # --- 3. Display the Gauge Chart ---
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = predicted_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Predicted Engagement Score", 'font': {'size': 24}},
                    gauge = {
                        'axis': {'range': [0, 50000], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "#0068c9"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "#cccccc",
                        'steps': [
                            {'range': [0, 5000], 'color': '#d1e0ff'},
                            {'range': [5000, 20000], 'color': '#a3c2ff'}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 45000}}))
                
                fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig, use_container_width=True)

                # --- 4. Display Summary & Suggestions ---
                st.subheader("Analysis & Suggestions")
                if predicted_score < 5000:
                    st.warning(f"**Low Engagement ({predicted_score})**: This score is on the lower end. Consider revising your caption with stronger calls to action, or increasing your budget for better reach.")
                elif predicted_score < 20000:
                    st.info(f"**Average Engagement ({predicted_score})**: This is a solid score. To improve, you could experiment with posting at a different time of day or testing a different ad copy.")
                else:
                    st.success(f"**High Engagement ({predicted_score})**: Excellent score! This campaign has high potential based on the provided data. This combination of category, budget, and caption is effective.")
                
                # --- 5. NEW: AI-Powered Caption Suggestions ---
                st.subheader("AI-Powered Caption Advice")
                caption_suggestions = get_caption_suggestions(ad_copy)
                if caption_suggestions:
                    for suggestion in caption_suggestions:
                        st.info(suggestion)
                else:
                    st.success("👍 **Great Caption!** Your ad copy already includes powerful keywords and engagement techniques.")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            st.error("The AI model is not loaded. Please check for errors.")

# Sidebar with future feature placeholder
st.sidebar.header("Account Insights")
st.sidebar.file_uploader("Upload Insights Screenshot", type=["png", "jpg", "jpeg"], disabled=True)
st.sidebar.info("💡 **Coming Soon!** In a future version, you'll be able to upload your account's performance insights for even more personalized predictions.")

