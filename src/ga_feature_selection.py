# improved_genetic_algorithm.py
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score

def load_data(sample_size=50000):
    print("Loading dataset...")
    path = "/home/islem/Documents/IIot_project/archive/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"
    
    data = pd.read_csv(path, low_memory=False)
    data.columns = data.columns.str.strip()
    
    # Keep only numeric columns
    data = data.select_dtypes(include=[np.number])
    data = data.dropna()
    
    if sample_size:
        data = data.sample(n=min(sample_size, len(data)), random_state=42)
    
    # FIXED: Use correct column name
    class_names = ['Normal', 'Attack']
    
    le = LabelEncoder()
    y = le.fit_transform(data['Attack_label'])
    
    X = data.drop(columns=['Attack_label'])
    
    # Scaling (Very important)
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    print(f"Dataset shape: {X.shape} | Classes: {class_names}")
    return X, y, class_names


def evaluate(solution, X, y):
    selected = np.where(solution == 1)[0]
    if len(selected) == 0:
        return 0.0
    
    X_selected = X.iloc[:, selected]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(
        n_estimators=60,      # Increased
        max_depth=15,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return f1_score(y_test, y_pred, average='weighted')


def genetic_algorithm(X, y, n_features, pop_size=20, generations=10):
    print(f"\nStarting Genetic Algorithm with population={pop_size}, generations={generations}...")
    start_time = time.time()
    
    # Initialize population (0 or 1 for each feature)
    population = np.random.randint(0, 2, (pop_size, n_features))
    
    best_solution = None
    best_score = 0.0
    
    for gen in range(generations):
        print(f"\nGeneration {gen+1}/{generations}")
        
        scores = []
        for solution in population:
            score = evaluate(solution, X, y)
            scores.append(score)
        
        scores = np.array(scores)
        
        # Track best
        best_idx = np.argmax(scores)
        if scores[best_idx] > best_score:
            best_score = scores[best_idx]
            best_solution = population[best_idx].copy()
        
        print(f"Best F1-score so far: {best_score:.4f}")
        
        # Elitism: Keep top 2 best solutions
        elite_idx = np.argsort(scores)[-2:]
        elite = population[elite_idx]
        
        # Create new population
        new_population = list(elite)
        
        while len(new_population) < pop_size:
            # Selection (tournament)
            p1 = population[np.random.randint(pop_size)]
            p2 = population[np.random.randint(pop_size)]
            
            # Crossover
            point = np.random.randint(1, n_features)
            child = np.concatenate([p1[:point], p2[point:]])
            
            # Mutation
            if np.random.random() < 0.15:   # mutation rate
                mutation_point = np.random.randint(n_features)
                child[mutation_point] = 1 - child[mutation_point]
            
            new_population.append(child)
        
        population = np.array(new_population)
    
    elapsed = time.time() - start_time
    selected_features = np.where(best_solution == 1)[0]
    
    print("\n" + "="*65)
    print("Genetic Algorithm Finished!")
    print(f"Best F1-Score     : {best_score:.4f}")
    print(f"Features Selected : {len(selected_features)} / {n_features}")
    print(f"Time Taken        : {elapsed:.1f} seconds")
    print(f"Selected indices  : {selected_features}")
    
    return best_solution, best_score, selected_features


# ===================== MAIN =====================
if __name__ == "__main__":
    X, y, class_names = load_data(sample_size=50000)   # Change to None for full data
    
    best_solution, best_score, selected_features = genetic_algorithm(
        X, y, 
        n_features=X.shape[1],
        pop_size=20,
        generations=10
    )
    
    # Final Evaluation
    print("\nFinal Evaluation on Best Features:")
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