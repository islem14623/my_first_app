# FINAL_PFE_PYSWARMS_CNN.py
"""
FINAL PFE SUBMISSION
Binary Particle Swarm Optimization (PySwarms) Feature Selection + CNN Classifier
"""

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

import pyswarms as ps
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

print("="*90)
print("FINAL PFE - PySwarms Binary PSO Feature Selection + CNN")
print("="*90)

# ==============================
# 1. Load Dataset
# ==============================
print("\nLoading dataset...")
path = "/home/islem/Documents/IIot_project/archive/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"

data = pd.read_csv(path, low_memory=False)
data.columns = data.columns.str.strip()

data = data.dropna()
data = data.sample(n=100000, random_state=42)

le = LabelEncoder()
y = le.fit_transform(data['Attack_label'])
class_names = ['Normal', 'Attack']

X = data.drop(columns=['Attack_label'])
X = X.select_dtypes(include=[np.number])

print(f"Total features: {X.shape[1]}")

# ==============================
# 2. Define Fitness Function for PySwarms
# ==============================
def fitness_function(position, X, y):
    # position is binary (0 or 1) for each feature
    selected = np.where(position == 1)[0]
    if len(selected) == 0:
        return 0.0
    
    X_sel = X.iloc[:, selected]
    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y, test_size=0.2, random_state=42, stratify=y
    )
    
    rf = RandomForestClassifier(n_estimators=60, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    score = rf.score(X_test, y_test)
    return -score  # Negative because PySwarms minimizes

# ==============================
# 3. Binary PSO using PySwarms
# ==============================
print("\nRunning Binary PSO Feature Selection using PySwarms...")

options = {'c1': 0.5, 'c2': 0.5, 'w': 0.9, 'k': 30, 'p': 2}

optimizer = ps.discrete.BinaryPSO(
    n_particles=30,
    dimensions=X.shape[1],
    options=options
)

cost, pos = optimizer.optimize(
    fitness_function, 
    iters=20, 
    verbose=True,
    X=X, 
    y=y
)

# Get selected features
selected_features = np.where(pos == 1)[0]
print(f"\nPySwarms Binary PSO selected {len(selected_features)} features")

# ==============================
# 4. Prepare Data for CNN
# ==============================
X_selected = X.iloc[:, selected_features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn  = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# ==============================
# 5. Build & Train CNN
# ==============================
print("\nBuilding and Training CNN Model...")

model = keras.Sequential([
    keras.layers.Conv1D(64, 3, activation='relu', padding='same', input_shape=(X_train_cnn.shape[1], 1)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(2),
    keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
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

early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

history = model.fit(
    X_train_cnn, y_train,
    epochs=25,
    batch_size=128,
    validation_split=0.2,
    class_weight=class_weight,
    callbacks=[early_stop],
    verbose=1
)

# ==============================
# 6. Evaluation
# ==============================
y_pred = (model.predict(X_test_cnn, verbose=0) > 0.5).astype(int).flatten()

print("\n" + "="*90)
print("FINAL RESULTS - PySwarms Binary PSO + CNN")
print("="*90)
print(f"Accuracy                    : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Features Selected by PSO    : {len(selected_features)} / {X.shape[1]}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

model.save("final_pyswarms_cnn_model.keras")
print("\nModel saved as 'final_pyswarms_cnn_model.keras'")