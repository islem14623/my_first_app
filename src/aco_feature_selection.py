# aco_feature_selection_fixed.py
import numpy as np
import pandas as pd
import time
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score

warnings.filterwarnings('ignore')


def load_data(sample_size=50000):
    print("Loading dataset...")
    path = "/home/islem/Documents/IIot_project/archive/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"
    
    data = pd.read_csv(path, low_memory=False)
    data.columns = data.columns.str.strip()
    
    data = data.select_dtypes(include=[np.number])
    data = data.dropna()
    
    if sample_size:
        data = data.sample(n=min(sample_size, len(data)), random_state=42)
    
    print(f"Dataset shape: {data.shape}")
    
    # FIXED: Use string class names
    class_names = ['Normal', 'Attack']
    
    le = LabelEncoder()
    y = le.fit_transform(data['Attack_label'])
    
    X = data.drop(columns=['Attack_label'])
    
    # Scaling
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    print(f"Features: {X.shape[1]} | Classes: {class_names}")
    return X, y, class_names


def evaluate(selected_features, X, y):
    if len(selected_features) == 0:
        return 0.0
    
    X_selected = X.iloc[:, selected_features]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(
        n_estimators=50, 
        max_depth=12, 
        n_jobs=-1, 
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return f1_score(y_test, y_pred, average='weighted')


def aco_feature_selection(X, y, n_ants=20, iterations=10, n_features_select=20):
    start_time = time.time()
    n_features = X.shape[1]
    
    print(f"\nStarting Improved ACO with {n_ants} ants and {iterations} iterations...")
    
    alpha = 1.0
    beta = 2.0
    rho = 0.2
    Q = 1.0
    
    pheromones = np.ones(n_features)
    
    # Heuristic (feature importance)
    quick_model = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)
    quick_model.fit(X, y)
    heuristic = quick_model.feature_importances_ + 1e-6
    
    best_features = None
    best_score = 0.0
    
    for it in range(iterations):
        print(f"\nIteration {it+1}/{iterations}")
        iteration_best_score = 0.0
        iteration_best_sol = None
        
        for ant in range(n_ants):
            probs = (pheromones ** alpha) * (heuristic ** beta)
            probs = probs / probs.sum()
            
            selected = np.random.choice(n_features, size=n_features_select, replace=False, p=probs)
            
            score = evaluate(selected, X, y)
            
            if score > iteration_best_score:
                iteration_best_score = score
                iteration_best_sol = selected.copy()
            
            if score > best_score:
                best_score = score
                best_features = selected.copy()
        
        print(f"Best in this iteration: {iteration_best_score:.4f}")
        
        # Update pheromones
        pheromones = (1 - rho) * pheromones
        if iteration_best_sol is not None:
            for f in iteration_best_sol:
                pheromones[f] += Q * iteration_best_score
    
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("ACO Feature Selection Finished!")
    print(f"Best F1-Score     : {best_score:.4f}")
    print(f"Features Selected : {len(best_features)}")
    print(f"Time Taken        : {elapsed:.1f} seconds")
    print(f"Selected indices  : {np.sort(best_features)}")
    
    return best_features, best_score


# ===================== MAIN =====================
if __name__ == "__main__":
    X, y, class_names = load_data(sample_size=50000)
    
    selected_features, best_score = aco_feature_selection(
        X, y, 
        n_ants=20, 
        iterations=10,
        n_features_select=20
    )
    
    # Final Evaluation
    print("\nFinal Evaluation with Selected Features:")
    X_final = X.iloc[:, selected_features]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42, stratify=y
    )
    
    final_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))