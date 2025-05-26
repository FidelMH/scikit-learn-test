# Import des bibliothèques nécessaires
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# Import de pandas pour la manipulation de données
import pandas as pd

# Chargement du dataset California Housing
df = fetch_california_housing(as_frame=True)
# Séparation des features (X) et de la target (y)
X,y = df.data, df.target

# Division des données en sets d'entraînement et de test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création d'un pipeline pour la normalisation des données
pipeline_rf = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", RandomForestRegressor(random_state=42))  # Modèle de régression
])

pipeline_dt = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", DecisionTreeRegressor(random_state=42))  # Modèle de régression
])



# Entraînement des modèles

pipeline_rf.fit(X_train, y_train)
pipeline_dt.fit(X_train, y_train)

# Prédictions sur le set de test
dtm_pred = pipeline_dt.predict(X_test)
rfm_pred = pipeline_rf.predict(X_test)

# Évaluation des modèles avec différentes métriques
# Erreur absolue moyenne
print("Decision Tree Mean Absolute Error:", round(mean_absolute_error(y_test, dtm_pred), 2))
print("Random Forest Mean Absolute Error:", round(mean_absolute_error(y_test, rfm_pred), 2))


# Affichage des noms des features
feature_names = df.feature_names
print("Feature Names:", feature_names)
# Affichage des importances des features avec leurs noms triées par ordre décroissant
feature_importances_dt = pd.Series(pipeline_dt.named_steps['regressor'].feature_importances_, index=feature_names).sort_values(ascending=False)
feature_importances_rf = pd.Series(pipeline_rf.named_steps['regressor'].feature_importances_, index=feature_names).sort_values(ascending=False)
print("Decision Tree Feature Importances with Names:\n", feature_importances_dt)
print("Random Forest Feature Importances with Names:\n", feature_importances_rf)

# Score R2 (coefficient de détermination)
print("Decision Tree R2 Score:", round(r2_score(y_test, dtm_pred), 2))
print("Random Forest R2 Score:", round(r2_score(y_test, rfm_pred), 2))

# Erreur absolue moyenne en pourcentage
print("Decision Tree Mean Absolute Percentage Error:", round(mean_absolute_percentage_error(y_test, dtm_pred), 2))
print("Random Forest Mean Absolute Percentage Error:", round(mean_absolute_percentage_error(y_test, rfm_pred), 2))


feature_importances_rf.plot(kind='barh', figsize=(8,6))
plt.title("Importances des features - Random Forest")
plt.xlabel("Importance")
plt.show()

