import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import numpy as np

print("--- Starting the training process for the FINAL ROBUST model ---")

# --- 1. Load the Dataset ---
try:
    data = pd.read_csv('ads_data.csv')
    print("✅ Step 1: Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ Error: ads_data.csv not found. Please ensure it's in the correct folder.")
    exit()

# --- 2. Definitive Data Cleaning ---
# This ensures all data is in the correct format before any processing.
data['Caption'] = data['Caption'].fillna('')
data['Engagement'] = pd.to_numeric(data['Engagement'], errors='coerce').fillna(0)
# We will not use the Image_File column in this model
data = data.drop(columns=['Image_File'])
print("✅ Step 2: Data has been cleaned and prepared!")

# --- 3. Define Features and Target ---
features = data.drop('Engagement', axis=1)
target = data['Engagement']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
print("✅ Step 3: Data has been split for training!")

# --- 4. Create the Ultimate Preprocessing Pipeline ---
# This pipeline is an "expert system" for preparing all your different types of data.

# Define which columns need which treatment
numerical_features = ['Target_Age', 'Budget']
# 'Followers' is now treated as a special categorical feature
categorical_features = ['Ad_Type', 'Platform', 'Posting_Time', 'Category', 'Followers'] 
text_feature = 'Caption'

# Create the preprocessor using ColumnTransformer
# It applies a different "tool" to each type of column.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('text', TfidfVectorizer(max_features=500), text_feature) # Text expert
    ])
print("✅ Step 4: Data processing pipeline created!")


# --- 5. Create the Final AI Model ---
# We combine the preprocessor and the "brain" (RandomForest) into a single, powerful pipeline.
final_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=150, random_state=42, max_depth=20))
])
print("✅ Step 5: Final robust AI model created!")


# --- 6. Train the Model ---
print("⏳ Step 6: Training the robust AI model... (This will be fast)")
final_model.fit(X_train, y_train)
print("✅ Model training complete!")

# --- 7. Save the Final Model ---
joblib.dump(final_model, 'ad_robust_model.pkl')
print("✅ All done! You now have a powerful, robust AI brain saved as 'ad_robust_model.pkl'.")
