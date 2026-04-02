import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# ==============================
# 1 Load and prepare data
# ==============================
def load_data():
    data = pd.read_csv("/home/islem/Documents/IIot_project/archive/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv",
                       low_memory=False)
    data = data.dropna()
    
    le = LabelEncoder()
    data['Attack_type'] = le.fit_transform(data['Attack_type'])
    
    # Remove non-numeric columns !
    data = data.select_dtypes(include=[np.number])
    
    X = data.drop(columns=['Attack_type'], errors='ignore')
    y = le.fit_transform(data['Attack_type']) if 'Attack_type' in data.columns else None
    
    return X, y

# ==============================
# 2 Evaluate a solution
# ==============================
def evaluate(solution, X, y):
    # Get selected features
    selected = np.where(solution == 1)[0]
    
    # Need at least 1 feature
    if len(selected) == 0:
        return 0
    
    X_selected = X.iloc[:, selected]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# ==============================
# 3 Genetic Algorithm
# ==============================
def genetic_algorithm(X, y, n_features, pop_size=10, generations=5):
    
    print("Starting Genetic Algorithm...")
    
    # Step 1 - Create random population
    population = np.random.randint(0, 2, (pop_size, n_features))
    
    best_solution = None
    best_accuracy = 0
    
    for gen in range(generations):
        print(f"Generation {gen+1}/{generations}")
        
        # Step 2 - Evaluate each solution
        scores = []
        for solution in population:
            acc = evaluate(solution, X, y)
            scores.append(acc)
        
        scores = np.array(scores)
        
        # Track best solution
        if scores.max() > best_accuracy:
            best_accuracy = scores.max()
            best_solution = population[scores.argmax()]
        
        print(f"Best accuracy so far: {best_accuracy:.4f}")
        
        # Step 3 - Select best solutions
        idx = np.argsort(scores)[::-1]
        population = population[idx]
        top = population[:pop_size//2]
        
        # Step 4 - Crossover
        new_population = list(top)
        while len(new_population) < pop_size:
            p1 = top[np.random.randint(len(top))]
            p2 = top[np.random.randint(len(top))]
            point = np.random.randint(1, n_features)
            child = np.concatenate([p1[:point], p2[point:]])
            new_population.append(child)
        
        population = np.array(new_population)
        
        # Step 5 - Mutation
        for i in range(len(population)):
            if np.random.random() < 0.1:
                point = np.random.randint(n_features)
                population[i][point] = 1 - population[i][point]
    
    selected_features = np.where(best_solution == 1)[0]
    print(f"\nBest accuracy : {best_accuracy:.4f}")
    print(f"Features selected : {len(selected_features)}")
    print(f"Feature indices : {selected_features}")
    
    return best_solution, best_accuracy

# ==============================
# 4 Main
# ==============================
if __name__ == "__main__":
    print("Loading data...")
    X, y = load_data()
    print(f"Dataset shape: {X.shape}")
    
    n_features = X.shape[1]
    
    best_solution, best_accuracy = genetic_algorithm(
        X, y, n_features, pop_size=10, generations=5)