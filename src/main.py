import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==============================
# 1 Load Dataset
# ==============================

data = pd.read_csv("Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv")

print("Dataset shape:", data.shape)

# ==============================
# 2 Data Cleaning
# ==============================

data = data.dropna()

# ==============================
# 3 Encode Labels
# ==============================

label_encoder = LabelEncoder()
data['Attack_type'] = label_encoder.fit_transform(data['Attack_type'])

# ==============================
# 4 Split Features / Labels
# ==============================

X = data.drop(columns=['Attack_type'])
y = data['Attack_type']

# ==============================
# 5 Feature Selection
# ==============================

selector = SelectKBest(score_func=chi2, k=20)
X_selected = selector.fit_transform(X, y)

print("Selected Features:", X_selected.shape)

# ==============================
# 6 Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

# ==============================
# 7 Machine Learning Model
# ==============================

model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)

# ==============================
# 8 Prediction
# ==============================

y_pred = model.predict(X_test)

# ==============================
# 9 Evaluation
# ==============================

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
