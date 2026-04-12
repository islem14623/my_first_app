# cnn_model_improved_pfe.py
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("="*80)
print("EDGE-IIOTSET - DEEP LEARNING MODEL (CNN) FOR INTRUSION DETECTION")
print("="*80)

# ==============================
# 1. Load and Prepare Data
# ==============================
print("\n[1] Loading dataset...")
path = "/home/islem/Documents/IIot_project/archive/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"

data = pd.read_csv(path, low_memory=False)
data.columns = data.columns.str.strip()

data = data.dropna()
data = data.sample(n=50000, random_state=42)

print(f"Dataset shape after sampling: {data.shape}")

# Correct target column
class_names = ['Normal', 'Attack']
le = LabelEncoder()
y = le.fit_transform(data['Attack_label'])

X = data.drop(columns=['Attack_label'])
X = X.select_dtypes(include=[np.number])

print(f"Number of features: {X.shape[1]}")

# ==============================
# 2. Feature Selection (from your GA/PSO)
# ==============================
# Best 22 features indices (you can update this list)
selected_indices = [0, 4, 7, 8, 9, 10, 11, 12, 13, 15, 16, 20, 21, 22, 
                   23, 24, 25, 26, 27, 30, 34, 36]

X_selected = X.iloc[:, selected_indices].values
print(f"Using {len(selected_indices)} selected features")

# ==============================
# 3. Scaling + Train/Test Split
# ==============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Reshape for CNN (samples, timesteps, features)
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn  = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(f"Train shape: {X_train_cnn.shape} | Test shape: {X_test_cnn.shape}")

# ==============================
# 4. Build Improved CNN Model
# ==============================
print("\n[2] Building CNN Model...")

model = keras.Sequential([
    keras.layers.Conv1D(64, kernel_size=3, activation='relu', 
                        input_shape=(X_train_cnn.shape[1], 1), padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(pool_size=2),
    
    keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(pool_size=2),
    
    keras.layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.GlobalAveragePooling1D(),
    
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint('best_cnn_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
]

# ==============================
# 5. Train the Model
# ==============================
print("\n[3] Training CNN Model...")

start_time = time.time()

history = model.fit(
    X_train_cnn, y_train,
    epochs=30,
    batch_size=128,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time

# ==============================
# 6. Evaluation
# ==============================
print("\n[4] Evaluating Model...")

y_pred_prob = model.predict(X_test_cnn, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*80)
print("CNN MODEL RESULTS FOR PFE")
print("="*80)
print(f"Accuracy           : {accuracy*100:.2f}%")
print(f"Training Time      : {training_time:.1f} seconds")
print(f"Features Used      : {len(selected_indices)}")
print(f"Best Epoch         : {len(history.history['loss'])}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==============================
# 7. Plot Training History (Optional but good for PFE)
# ==============================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('cnn_training_history.png', dpi=300)
print("\nTraining history plot saved as 'cnn_training_history.png'")
