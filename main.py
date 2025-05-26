# Import des bibliothèques nécessaires
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd

# Chargement du dataset California Housing
df = fetch_california_housing(as_frame=True)
# Séparation des features (X) et de la target (y)
X,y = df.data, df.target

# Division des données en sets d'entraînement et de test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création des modèles
dtm = DecisionTreeRegressor(random_state=42)  # Modèle d'arbre de décision
rfm = RandomForestRegressor(random_state=42)   # Modèle de forêt aléatoire

# Entraînement des modèles
dtm.fit(X_train, y_train)
rfm.fit(X_train, y_train)

# Prédictions sur le set de test
dtm_pred = dtm.predict(X_test)
rfm_pred = rfm.predict(X_test)

# Évaluation des modèles avec différentes métriques
# Erreur absolue moyenne
print("Decision Tree Mean Absolute Error:", round(mean_absolute_error(y_test, dtm_pred), 2))
print("Random Forest Mean Absolute Error:", round(mean_absolute_error(y_test, rfm_pred), 2))

# Importance des features pour chaque modèle
print("Decision Tree Feature Importances:", dtm.feature_importances_)
print("Random Forest Feature Importances:", rfm.feature_importances_)

# Score R2 (coefficient de détermination)
print("Decision Tree R2 Score:", round(r2_score(y_test, dtm_pred), 2))
print("Random Forest R2 Score:", round(r2_score(y_test, rfm_pred), 2))

# Erreur absolue moyenne en pourcentage
print("Decision Tree Mean Absolute Percentage Error:", round(mean_absolute_percentage_error(y_test, dtm_pred), 2))
print("Random Forest Mean Absolute Percentage Error:", round(mean_absolute_percentage_error(y_test, rfm_pred), 2))
