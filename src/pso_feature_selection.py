# pso_feature_selection_fixed.py
import numpy as np
import pandas as pd
import time
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings('ignore')


def load_data(sample_size=None):
    """Load and prepare the Edge-IIoTset dataset"""
    print("Loading dataset...")
    
    path = "/home/islem/Documents/IIot_project/archive/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"
    
    data = pd.read_csv(path, low_memory=False)
    data.columns = data.columns.str.strip()
    
    # Keep only numeric columns
    data = data.select_dtypes(include=[np.number])
    data = data.dropna()
    
    if sample_size is not None:
        data = data.sample(n=min(sample_size, len(data)), random_state=42)
    
    print(f"Dataset shape after cleaning: {data.shape}")
    
    # Target
    if 'Attack_label' not in data.columns:
        raise KeyError("Column 'Attack_label' not found in dataset!")
    
    # FIX: Get original class names as strings BEFORE encoding
    class_names = [str(cls) for cls in sorted(data['Attack_label'].unique())]
    
    le = LabelEncoder()
    y = le.fit_transform(data['Attack_label'])
    
    X = data.drop(columns=['Attack_label'])
    
    # Scaling
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    print(f"Number of features: {X.shape[1]} | Number of samples: {X.shape[0]}")
    print(f"Classes: {class_names}")
    
    return X, y, class_names


def evaluate(solution, X, y, n_estimators=60):
    """Evaluate a single solution (feature subset)"""
    selected = np.where(solution >= 0.5)[0]
    
    if len(selected) == 0:
        return 0.0
    
    X_selected = X.iloc[:, selected]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=15,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


def pso_feature_selection(X, y, n_particles=15, iterations=8):
    """Improved Particle Swarm Optimization for feature selection"""
    start_time = time.time()
    n_features = X.shape[1]
    
    print(f"\nStarting PSO with {n_particles} particles and {iterations} iterations...")
    
    w = 0.729
    c1 = 1.49445
    c2 = 1.49445
    
    particles = np.random.uniform(0, 1, (n_particles, n_features))
    velocities = np.random.uniform(-0.5, 0.5, (n_particles, n_features))
    
    p_best_pos = particles.copy()
    p_best_scores = np.array([evaluate(p, X, y) for p in particles])
    
    g_best_idx = np.argmax(p_best_scores)
    g_best_pos = p_best_pos[g_best_idx].copy()
    g_best_score = p_best_scores[g_best_idx]
    
    print(f"Initial best accuracy: {g_best_score:.4f}")
    
    for it in range(iterations):
        print(f"\nIteration {it+1}/{iterations} | Current best: {g_best_score:.4f}")
        
        for i in range(n_particles):
            r1 = np.random.random(n_features)
            r2 = np.random.random(n_features)
            
            velocities[i] = (w * velocities[i] +
                            c1 * r1 * (p_best_pos[i] - particles[i]) +
                            c2 * r2 * (g_best_pos - particles[i]))
            
            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], 0, 1)
            
            score = evaluate(particles[i], X, y)
            
            if score > p_best_scores[i]:
                p_best_scores[i] = score
                p_best_pos[i] = particles[i].copy()
            
            if score > g_best_score:
                g_best_score = score
                g_best_pos = particles[i].copy()
                print(f"   → New best found! Accuracy = {g_best_score:.4f} (particle {i})")
    
    best_solution = (g_best_pos >= 0.5).astype(int)
    selected_features = np.where(best_solution == 1)[0]
    
    elapsed = time.time() - start_time
    print("\n" + "="*65)
    print("PSO Feature Selection Completed!")
    print(f"Best Accuracy Achieved : {g_best_score:.4f}")
    print(f"Features Selected      : {len(selected_features)} / {n_features}")
    print(f"Total Time             : {elapsed:.1f} seconds ({elapsed/60:.1f} min)")
    print(f"Selected feature indices: {selected_features.tolist()}")
    
    return best_solution, g_best_score, selected_features


# ==============================
# Main
# ==============================
if __name__ == "__main__":
    # Use 80000 for speed during testing. Change to None for full dataset later.
    X, y, class_names = load_data(sample_size=80000)
    
    best_solution, best_accuracy, selected_features = pso_feature_selection(
        X, y,
        n_particles=15,
        iterations=8
    )
    
    # Final Evaluation
    print("\nFinal Evaluation on Best Feature Subset:")
    X_final = X.iloc[:, selected_features]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42, stratify=y
    )
    
    final_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=class_names, 
                                digits=4))
    
    print("\nConfusion Matrix:")
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_test, y_pred))