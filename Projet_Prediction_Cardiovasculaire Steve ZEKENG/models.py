import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import joblib
import os

# Load dataset
data = pd.read_csv("heart_cleveland_upload.csv")

# Split features and target
X = data.drop("target", axis=1)
y = data["target"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

# Train and save models
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, f"models/{name.replace(' ', '_').lower()}.pkl")
