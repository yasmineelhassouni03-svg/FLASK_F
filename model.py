import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import joblib

# Charger les données Iris
iris = load_iris()
X = iris.data
y = iris.target

# Entraînement du modèle
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Sauvegarde du modèle
joblib.dump(model, "model.pkl")

print("Modèle entraîné et sauvegardé avec succès")