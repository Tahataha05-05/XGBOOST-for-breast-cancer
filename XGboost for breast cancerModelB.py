import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

# ============================================================
# 1. CHARGEMENT ET NETTOYAGE
# ============================================================

data = pd.read_csv("C:/Users/lenovo/Desktop/Breastcancer/breastcancerdataset.txt")
print("\nValeurs manquantes :")
print(data.isnull().sum().sort_values(ascending=False).head(10))
print("\nNombre de doublons :", data.duplicated().sum())
print("\nRépartition des diagnostics :")
print(data["diagnosis"].value_counts())

data = data.drop(columns=["Unnamed: 32", "id"], errors="ignore")
data["diagnosis"] = data["diagnosis"].map({"B": 0, "M": 1})

# Corrélations entre les variables explicatives
X = data.drop(columns="diagnosis")
y = data["diagnosis"]

corr = X.corr()
plt.figure(figsize=(18, 15))
sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1)
plt.title("Heatmap des corrélations entre variables")
plt.tight_layout()
plt.show()

# Corrélation de chaque variable avec le diagnostic
corr_target = (
    data.corr()["diagnosis"]
    .drop("diagnosis")
    .sort_values(key=abs, ascending=False)
)

plt.figure(figsize=(9, 10))
sns.barplot(x=corr_target.values, y=corr_target.index)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Corrélation avec le diagnostic")
plt.xlabel("Corrélation de Pearson")
plt.tight_layout()
plt.show()

# ============================================================
# 2. SPLIT TRAIN/TEST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

# ============================================================
# MODELE B : REDUCTION DE LA REDONDANCE
# ============================================================

# Repérer les variables fortement corrélées entre elles
corr_train = X_train.corr().abs()
upper = corr_train.where(np.triu(np.ones(corr_train.shape), k=1).astype(bool))

# Variables candidates à supprimer
to_drop_B = [column for column in upper.columns if any(upper[column] > 0.8)]

# Conserver les mêmes colonnes dans train et test
X_train_B = X_train.drop(columns=to_drop_B)
X_test_B = X_test.drop(columns=to_drop_B)

print("Nombre initial de variables :", X_train.shape[1])
print("Variables supprimées :", len(to_drop_B))
print("Variables restantes :", X_train_B.shape[1])
if to_drop_B:
    print(to_drop_B)

# Entraînement du modèle avec les mêmes hyperparamètres que le modèle A
model_B = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)
model_B.fit(X_train_B, y_train)

# Prédictions sur le test
y_pred_B = model_B.predict(X_test_B)
y_proba_B = model_B.predict_proba(X_test_B)[:, 1]

results_B = {
    "Accuracy": accuracy_score(y_test, y_pred_B),
    "Precision": precision_score(y_test, y_pred_B, zero_division=0),
    "Recall": recall_score(y_test, y_pred_B, zero_division=0),
    "F1-score": f1_score(y_test, y_pred_B, zero_division=0),
    "ROC-AUC": roc_auc_score(y_test, y_proba_B),
}

print("\n===== MODELE B : TEST =====")
for metric, value in results_B.items():
    print(f"{metric:12s}: {value:.4f}")

print("\n===== MATRICE DE CONFUSION B =====")
print(confusion_matrix(y_test, y_pred_B, labels=[0, 1]))

print("\n===== RAPPORT B =====")
print(
    classification_report(
        y_test,
        y_pred_B,
        labels=[0, 1],
        target_names=["Benigne", "Maligne"],
        zero_division=0,
    )
)

# Matrice de confusion
cm_B = confusion_matrix(y_test, y_pred_B, labels=[0, 1])
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_B,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Benigne", "Maligne"],
    yticklabels=["Benigne", "Maligne"],
)
plt.xlabel("Classe prédite")
plt.ylabel("Classe réelle")
plt.title("Matrice de confusion - modèle B")
plt.tight_layout()
plt.show()

# Courbe ROC
fig, ax = plt.subplots(figsize=(7, 6))
RocCurveDisplay.from_predictions(y_test, y_proba_B, name="XGBoost B", ax=ax)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Classifieur aléatoire")
ax.set_title("Courbe ROC - modèle B")
ax.legend()
plt.tight_layout()
plt.show()

# Courbe Precision-Recall
fig, ax = plt.subplots(figsize=(7, 6))
PrecisionRecallDisplay.from_predictions(y_test, y_proba_B, name="XGBoost B", ax=ax)
ax.set_title("Courbe Precision-Recall - modèle B")
plt.tight_layout()
plt.show()

# Importance des variables du modèle B
importance_B = pd.Series(
    model_B.feature_importances_,
    index=X_train_B.columns,
).sort_values(ascending=False)
plt.figure(figsize=(10, 8))
sns.barplot(x=importance_B.values, y=importance_B.index)
plt.xlabel("Importance relative")
plt.ylabel("Variable")
plt.title("Importance des variables - modèle B")
plt.tight_layout()
plt.show()

# Validation croisée stratifiée à 5 folds
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

cv_results = cross_validate(
    model_B,
    X_train_B,
    y_train,
    cv=cv,
    scoring=scoring,
    n_jobs=1,
    return_train_score=False,
)

print("\n===== VALIDATION CROISÉE (5 FOLDS) =====")
for metric in scoring:
    scores = cv_results[f"test_{metric}"]
    print(f"{metric.upper():<12}: {scores.mean():.4f} +/- {scores.std():.4f}")