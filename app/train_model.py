import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# This script will read your CSV data, learn from it, and save the "brain".

# 1. Load the "experience book" (your CSV file)
print("Step 1: Loading the dataset...")
data = pd.read_csv('ads_data.csv')
print("Dataset loaded successfully!")

# 2. Prepare the data for the AI
# The AI brain only understands numbers, so we convert text like "Instagram"
# into a number format. This is called 'one-hot encoding'.
print("Step 2: Preparing the data...")
features = pd.get_dummies(data.drop('Engagement', axis=1))
target = data['Engagement']
print("Data is ready for training!")

# 3. Time to learn!
# This is where the AI studies the data to find patterns.
print("Step 3: Training the AI model... (This will be very fast)")
# We are using a 'RandomForestRegressor', which is like an experienced committee
# that votes on the final prediction.
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(features, target)
print("Model training complete!")

# 4. Save the brain!
# We save the learned knowledge into a file so we can use it later in your app.
print("Step 4: Saving the 'AI Brain' to a file...")
joblib.dump(model, 'ad_model.pkl')

# We also save the column names so your app knows exactly what
# information to send to the brain.
model_columns = list(features.columns)
joblib.dump(model_columns, 'model_columns.pkl')

print("\nAll done! You now have a trained AI brain saved as 'ad_model.pkl'.")

