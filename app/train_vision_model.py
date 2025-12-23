import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate, TextVectorization
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageDraw, ImageFont # Using Pillow for robust image handling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

print("--- Starting the training process for the ULTIMATE VISION model (v7 - Final) ---")

# --- Configuration ---
IMG_WIDTH, IMG_HEIGHT = 224, 224
IMAGE_FOLDER = 'ad_images'

# --- 1. Load the Dataset ---
try:
    data = pd.read_csv('ads_data.csv')
    print("✅ Step 1: Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ Error: ads_data.csv not found. Please make sure it's in the correct folder.")
    exit()

# --- 2. Data Cleaning (DEFINITIVE FIX) ---
# Clean the data BEFORE any other processing to prevent all errors.
data['Caption'] = data['Caption'].fillna('')
# Force the Engagement column to be a clean numerical type.
# This is the most critical fix.
data['Engagement'] = pd.to_numeric(data['Engagement'], errors='coerce').fillna(0)
print("✅ Step 2: Data has been cleaned successfully!")


# --- 3. Image Preprocessing & Generation Functions ---
def create_placeholder_image(text="No Image Found"):
    """Creates a simple, blank placeholder image with text."""
    try:
        img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), color = (211, 211, 211))
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()
        d.text((10,10), text, fill=(0,0,0), font=font)
        return img
    except Exception as e:
        return Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), color = (211, 211, 211))

def preprocess_image(file_path):
    """Loads a real image or creates a placeholder if not found."""
    try:
        full_path = os.path.join(IMAGE_FOLDER, file_path)
        img = Image.open(full_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
    except (FileNotFoundError, IOError):
        img = create_placeholder_image(file_path)
        
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

print("✅ Step 3: Image handling functions are ready!")

# --- 4. Prepare Data for the Model ---
print("⏳ Step 4: Processing all images listed in the dataset (this will be fast)...")
data['image_features'] = data['Image_File'].apply(preprocess_image)
print("✅ All images processed!")

features = data.drop('Engagement', axis=1)
target = data['Engagement']

# The target is now already clean, so we just split it.
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Convert the clean target data to the final NumPy format
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

X_train_images = np.array(X_train['image_features'].tolist())
X_test_images = np.array(X_test['image_features'].tolist())

# Reshape the text data to be (number_of_samples, 1) to match the model's input layer.
X_train_text = X_train['Caption'].astype(str).to_numpy().reshape(-1, 1)
X_test_text = X_test['Caption'].astype(str).to_numpy().reshape(-1, 1)

tabular_features_train = X_train.drop(columns=['Caption', 'Image_File', 'image_features'])
tabular_features_test = X_test.drop(columns=['Caption', 'Image_File', 'image_features'])

# --- 5. Create Preprocessing Pipeline for Tabular Data ---
numerical_features = ['Target_Age', 'Budget']
categorical_features = ['Ad_Type', 'Platform', 'Posting_Time', 'Category', 'Followers']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

X_train_tabular_processed = preprocessor.fit_transform(tabular_features_train)
X_test_tabular_processed = preprocessor.transform(tabular_features_test)

joblib.dump(preprocessor, 'final_preprocessor.pkl')
print("✅ Step 5: Tabular data preprocessor created and saved!")

# --- 6. Build the Ultimate AI Brain (Neural Network) ---
base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False
image_input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3), name='image_input')
x_image = base_model(image_input, training=False)
x_image = tf.keras.layers.GlobalAveragePooling2D()(x_image)
x_image = Dense(32, activation='relu')(x_image)

vectorizer = TextVectorization(max_tokens=2000, output_sequence_length=100)
vectorizer.adapt(X_train_text.flatten())
text_input = Input(shape=(1,), dtype=tf.string, name='text_input')
x_text = vectorizer(text_input)
x_text = tf.keras.layers.Embedding(input_dim=len(vectorizer.get_vocabulary()), output_dim=128)(x_text)
x_text = tf.keras.layers.GlobalAveragePooling1D()(x_text)
x_text = Dense(32, activation='relu')(x_text)

joblib.dump({'config': vectorizer.get_config(),
             'weights': vectorizer.get_weights()}, 'text_vectorizer.pkl')

tabular_input = Input(shape=(X_train_tabular_processed.shape[1],), name='tabular_input')
x_tabular = Dense(32, activation='relu')(tabular_input)

combined = concatenate([x_image, x_text, x_tabular])
z = Dense(64, activation='relu')(combined)
z = Dense(32, activation='relu')(z)
output = Dense(1, name='output')(z)

vision_model = Model(inputs=[image_input, text_input, tabular_input], outputs=output)

vision_model.compile(optimizer='adam', loss='mean_squared_error')
print("✅ Step 6: Ultimate AI Brain constructed successfully!")

# --- 7. Train the Model ---
print("⏳ Step 7: Training the Ultimate AI Brain... (This will take a few minutes)")
vision_model.fit(
    {'image_input': X_train_images, 'text_input': X_train_text, 'tabular_input': X_train_tabular_processed},
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
print("✅ Model training complete!")

# --- 8. Save the Final Model ---
vision_model.save('ad_vision_model.h5')
print("✅ All done! You now have an all-seeing AI brain saved as 'ad_vision_model.h5'.")

