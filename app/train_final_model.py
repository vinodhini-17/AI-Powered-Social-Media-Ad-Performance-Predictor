import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import re

print("--- Starting the training process for the final model ---")

# --- Step 1: Load the final dataset ---
try:
    data = pd.read_csv('ads_data.csv')
    print("✅ Step 1: Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ ERROR: 'ads_data.csv' not found. Please make sure the file is in the correct folder.")
    exit()

# --- Step 2: Define features and target ---
# We are dropping 'Image_File' for now as we are not using it in this model version
features = data.drop(['Engagement', 'Image_File'], axis=1)
target = data['Engagement']

# --- Step 3: Set up the processing pipeline for all data types ---
# This is where we tell the AI how to handle each type of column

# Identify different types of columns
categorical_features = ['Ad_Type', 'Platform', 'Posting_Time', 'Category', 'Followers']
numerical_features = ['Target_Age', 'Budget']
text_features = 'Caption'

# Create a pre-processing pipeline
# It will apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('text', TfidfVectorizer(stop_words='english'), text_features)
    ],
    remainder='drop' # Drop any columns we haven't specified
)

print("✅ Step 2: Data processing pipeline created successfully!")

# --- Step 4: Create and train the final AI model ---
# We combine the preprocessor with our AI model (RandomForest)
final_model = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

print("⏳ Step 3: Training the final AI model... (This may take a moment)")
# Train the model on the entire dataset
final_model.fit(features, target)
print("✅ Step 3: Model training complete!")


# --- Step 5: Save the final trained model ---
joblib.dump(final_model, 'ad_final_model.pkl')
print("✅ Step 4: Final AI brain saved as 'ad_final_model.pkl'")
print("\n--- Process complete! You are ready to connect this to your app. ---")

