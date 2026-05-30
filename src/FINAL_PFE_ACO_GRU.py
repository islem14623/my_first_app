"""
ACO + GRU Pipeline - Final PFE Version
Complete pipeline with NO data leakage
Uses pre-computed ACO feature selection results from CNN run
Author: Islem Chenafi
University: Université Ferhat Abbas Sétif-1
"""

import numpy as np
import pandas as pd
import joblib
import json
import time
import warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Import load_data from ACO module (feature selection skipped - using saved results)
from aco_feature_selection import load_data

warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def build_gru_model(n_features, learning_rate=0.001):
    """Build GRU model for binary classification"""
    model = keras.Sequential([
        keras.layers.GRU(64, return_sequences=True,
                        input_shape=(n_features, 1)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),

        keras.layers.GRU(32, return_sequences=False),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def plot_training_history(history, save_path):
    """Plot training and validation loss/accuracy curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(history.history['loss'], label='Training Loss',
                 color='blue', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss',
                 color='red', linewidth=2)
    axes[0].set_title('Model Loss During Training',
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(history.history['accuracy'], label='Training Accuracy',
                 color='blue', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy',
                 color='red', linewidth=2)
    axes[1].set_title('Model Accuracy During Training',
                      fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved: {save_path}")


def main():
    pipeline_start = time.time()

    print("="*90)
    print("FINAL PFE PIPELINE: ACO FEATURE SELECTION + GRU")
    print("="*90)
    print("Student: Islem Chenafi")
    print("Topic: Feature Selection for IIoT Intrusion Detection")
    print("="*90)

    print("="*90)
    print("NOTE: Feature selection results loaded from previous CNN run")
    print("This ensures fair comparison between classifiers on same features")
    print("="*90)

    # ========================================
    # STEP 1: Load Data
    # ========================================
    print("\n[STEP 1] Loading dataset...")
    X, y, feature_names, class_names = load_data(sample_size=None)

    # ========================================
    # STEP 2: Split data FIRST
    # ========================================
    print("\n[STEP 2] Splitting data (before any preprocessing)...")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    print(f"✓ Train set      : {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"✓ Validation set : {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"✓ Test set       : {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

    # ========================================
    # STEP 3: Scale AFTER splitting
    # ========================================
    print("\n[STEP 3] Scaling features (fit on train only)...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"✓ Scaler fit on {X_train_scaled.shape[0]} training samples")
    print(f"✓ Test set NEVER seen by scaler (no data leakage!)")

    # ========================================
    # STEP 4: Load pre-selected features from saved file
    # ========================================
    print("\n[STEP 4] Loading pre-selected features from saved file...")
    selected_indices = joblib.load("models/aco_selected_features.pkl")
    print(f"✓ Loaded {len(selected_indices)} features (from CNN run)")
    print(f"✓ Skipping FS algorithm - using saved features for fair comparison")

    selected_indices = np.sort(selected_indices)

    # Apply feature selection
    X_train_selected = X_train_scaled[:, selected_indices]
    X_val_selected = X_val_scaled[:, selected_indices]
    X_test_selected = X_test_scaled[:, selected_indices]

    print(f"\n✓ Applied {len(selected_indices)} features out of {X_train_scaled.shape[1]}")

    # ========================================
    # STEP 5: Reshape for GRU
    # ========================================
    print("\n[STEP 5] Preparing data for GRU...")

    X_train_gru = X_train_selected.reshape(X_train_selected.shape[0], X_train_selected.shape[1], 1)
    X_val_gru = X_val_selected.reshape(X_val_selected.shape[0], X_val_selected.shape[1], 1)
    X_test_gru = X_test_selected.reshape(X_test_selected.shape[0], X_test_selected.shape[1], 1)

    print(f"✓ GRU input shape: {X_train_gru.shape}")

    # ========================================
    # STEP 6: Build GRU
    # ========================================
    print("\n[STEP 6] Building GRU model...")
    model = build_gru_model(n_features=len(selected_indices))
    print(model.summary())

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )

    class_weight = {0: 1.0, 1: 2.5}

    # ========================================
    # STEP 7: Train GRU
    # ========================================
    print("\n[STEP 7] Training GRU...")
    print(f"Class weights: {class_weight}")

    history = model.fit(
        X_train_gru, y_train,
        validation_data=(X_val_gru, y_val),
        epochs=20,
        batch_size=128,
        class_weight=class_weight,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Plot training history
    plot_training_history(history, "aco_gru_training_history.png")

    # ========================================
    # STEP 8: Final Evaluation
    # ========================================
    print("\n" + "="*90)
    print("FINAL RESULTS - TEST SET (NEVER SEEN BEFORE)")
    print("="*90)

    y_pred_prob = model.predict(X_test_gru, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    test_accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")
    print(f"Features Used (loaded from ACO CNN run): {len(selected_indices)} / {len(feature_names)}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    print(f"\nTrue Negatives (Normal correctly identified) : {cm[0,0]}")
    print(f"False Positives (Normal wrongly as Attack)   : {cm[0,1]}")
    print(f"False Negatives (Attack wrongly as Normal)   : {cm[1,0]}")
    print(f"True Positives (Attack correctly identified) : {cm[1,1]}")

    attack_recall = cm[1,1] / (cm[1,0] + cm[1,1])
    attack_precision = cm[1,1] / (cm[0,1] + cm[1,1])
    attack_f1 = 2 * (attack_precision * attack_recall) / (attack_precision + attack_recall)

    print(f"\nAttack Detection Metrics:")
    print(f"  Recall (Detection Rate)    : {attack_recall:.4f}")
    print(f"  Precision                  : {attack_precision:.4f}")
    print(f"  F1-Score                   : {attack_f1:.4f}")

    # ========================================
    # STEP 9: Save Everything
    # ========================================
    print("\n[STEP 9] Saving model and artifacts...")

    model.save("final_aco_gru_model.keras")
    joblib.dump(scaler, "aco_gru_scaler.pkl")

    # Save metrics
    final_metrics = {
        "algorithm": "ACO + GRU",
        "features_loaded_from": "aco_selected_features.pkl",
        "feature_selection_reused": True,
        "test_accuracy": float(test_accuracy),
        "attack_recall": float(attack_recall),
        "attack_precision": float(attack_precision),
        "attack_f1": float(attack_f1),
        "features_selected": int(len(selected_indices)),
        "total_features": int(len(feature_names)),
        "feature_reduction_percent": float(100 - (len(selected_indices)/len(feature_names)*100)),
        "model_params": int(model.count_params()),
        "training_time_seconds": float(time.time() - pipeline_start),
        "confusion_matrix": cm.tolist(),
        "selected_feature_indices": selected_indices.tolist(),
    }

    with open("aco_gru_final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    print("✓ Model saved as 'final_aco_gru_model.keras'")
    print("✓ Scaler saved as 'aco_gru_scaler.pkl'")
    print("✓ Metrics saved as 'aco_gru_final_metrics.json'")
    print("✓ Training history plot saved as 'aco_gru_training_history.png'")

    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*90)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*90)
    print(f"Final Test Accuracy : {test_accuracy*100:.2f}%")
    print(f"Features Reduced    : {len(feature_names)} → {len(selected_indices)} ({len(selected_indices)/len(feature_names)*100:.1f}% retained)")
    print(f"Model Parameters    : {model.count_params():,}")
    print(f"Total Time          : {(time.time()-pipeline_start)/60:.1f} minutes")
    print("="*90)


if __name__ == "__main__":
    main()
