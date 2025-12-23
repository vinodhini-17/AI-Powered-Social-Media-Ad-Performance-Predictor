import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

print("Starting the training process for the new, smarter AI...")

# Step 1: Load the upgraded dataset
print("--> Loading the dataset with captions...")
data = pd.read_csv('ads_data.csv')
# Simple cleanup: remove any rows that might have an empty caption
data = data.dropna(subset=['Caption'])
print("--> Dataset loaded successfully.")

# Step 2: Define the different types of data we have
# The AI needs to know which columns are numbers, which are categories, and which is text.
X = data.drop('Engagement', axis=1)
y = data['Engagement']

categorical_features = ['Ad_Type', 'Platform', 'Posting_Time']
numerical_features = ['Target_Age', 'Budget']
text_feature = 'Caption'

# Step 3: Create the "Language Expert" and other preprocessing tools
# This ColumnTransformer is like a manager that tells different tools how to handle different columns.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        # The TfidfVectorizer is our powerful "language expert" tool.
        ('text', TfidfVectorizer(stop_words='english', max_features=500), text_feature)
    ])

# Step 4: Build the full AI pipeline
# A pipeline chains all the steps together: first preprocess the data, then train the model.
# This makes our AI brain much more organized and efficient.
text_model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

# Step 5: Train the new, powerful AI brain!
print("--> Training the AI brain... This may take a moment.")
text_model_pipeline.fit(X, y)
print("--> Training complete!")

# Step 6: Save the entire pipeline (the brain and all its tools) to a single file
joblib.dump(text_model_pipeline, 'ad_text_model.pkl')
print("\n✅ All done! The new AI brain that understands text is saved as 'ad_text_model.pkl'.")
