# FINAL_PFE_ACO_CNN_FIXED.py
"""
FINAL PFE SUBMISSION
Ant Colony Optimization (ACO) Feature Selection + CNN
Corrected Version
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

# Import your ACO function
from aco_feature_selection import aco_feature_selection, load_data

print("="*90)
print("FINAL PFE SUBMISSION - ACO + CNN")
print("="*90)

# 1. Load Dataset
print("\n[1] Loading dataset...")
X, y, class_names = load_data(sample_size=100000)

print(f"Total features before selection: {X.shape[1]}")

# 2. ACO Feature Selection
print("\n[2] Running ACO Feature Selection...")

# Call ACO and handle the return value safely
result = aco_feature_selection(X, y, n_ants=20, iterations=10, n_features_select=22)

# Debug: Show what ACO returns
print(f"Return type from ACO: {type(result)}")

# Safely extract selected_features
if isinstance(result, tuple):
    # If it returns multiple values, take the first one that looks like indices
    for item in result:
        if isinstance(item, (list, np.ndarray)) and len(item) > 0:
            selected_features = np.array(item).flatten()
            break
    else:
        selected_features = np.array([])
else:
    selected_features = np.array(result).flatten()

print(f"ACO selected {len(selected_features)} features: {selected_features.tolist()}")

# 3. Prepare Data
print("\n[3] Preparing data with ACO-selected features...")

X_selected = X.iloc[:, selected_features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn  = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(f"Input shape for CNN: {X_train_cnn.shape}")

# 4. Build CNN Model
print("\n[4] Building CNN Model...")

model = keras.Sequential([
    keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same',
                        input_shape=(X_train_cnn.shape[1], 1)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(2),
    
    keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(2),
    
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid')
])

class_weight = {0: 1.0, 1: 3.0}

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)]

print("\nTraining CNN...")
history = model.fit(
    X_train_cnn, y_train,
    epochs=25,
    batch_size=128,
    validation_split=0.2,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

# 5. Evaluation
y_pred = (model.predict(X_test_cnn, verbose=0) > 0.5).astype(int).flatten()

print("\n" + "="*90)
print("FINAL RESULTS - ACO + CNN")
print("="*90)
print(f"Accuracy                    : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Features Selected by ACO    : {len(selected_features)} / {X.shape[1]}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

model.save("final_aco_cnn_model.keras")
print("\nModel saved successfully!")