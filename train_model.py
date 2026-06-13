import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

print("--- Starting V4.0 AI Model Training ---")

# --- 1. Load Your New Dataset ---
try:
    data = pd.read_csv('my_ads_data.csv')
    print("✅ Step 1: New dataset 'my_ads_data.csv' loaded successfully!")
except FileNotFoundError:
    print("❌ Error: 'my_ads_data.csv' not found. Please save the new data and try again.")
    exit()

# --- 2. Prepare Features and Target ---
# We'll create a combined text column for the TfidfVectorizer
data['combined_text'] = data['ad_text'].fillna('') + ' ' + data['hashtags'].fillna('')

# We're dropping the currency column as it's for display in the app.
# We also drop campaign_id and expected_engagement_rate as they are not input features for the model.
features = data.drop(['campaign_id', 'actual_leads', 'expected_engagement_rate', 'currency'], axis=1)
target = data['actual_leads']
print("✅ Step 2: Features and target variable defined.")

# --- 3. Create the Advanced Preprocessing Pipeline ---
# Define column types
categorical_features = ['product_type', 'campaign_type', 'location']
numerical_features = ['budget', 'image_brightness', 'image_colorfulness', 'cta_strength', 'audience_size']
text_feature = 'combined_text'

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('text', TfidfVectorizer(), text_feature)
    ],
    remainder='drop'
)

# --- 4. Define the Full Training Pipeline ---
training_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])
print("✅ Step 3: Advanced training pipeline created.")

# --- 5. Train the New AI Model ---
print("⏳ Step 4: Training the AI on your new, richer dataset... (This might take a moment)")
training_pipeline.fit(features, target)
print("✅ AI training complete!")

# --- 6. Save the Upgraded AI Brain ---
joblib.dump(training_pipeline, 'lead_predictor_model_v4.pkl')
print("✅ Upgraded AI Brain saved successfully as 'lead_predictor_model_v4.pkl'!")