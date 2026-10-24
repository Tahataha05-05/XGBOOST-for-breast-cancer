import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
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
    PrecisionRecallDisplay
)
data = pd.read_csv("C:/Users/lenovo/Desktop/Breastcancer/breastcancerdataset.txt")
print("\nValeurs manquantes :")
print(data.isnull().sum().sort_values(ascending=False).head(10))
print("\nNombre de doublons :", data.duplicated().sum())
print("\nRépartition des diagnostics :")
print(data["diagnosis"].value_counts())
data = data.drop(columns=["Unnamed: 32", "id"], errors="ignore")

# Encodage de la cible : bénigne = 0, maligne = 1
data["diagnosis"] = data["diagnosis"].map({
    "B": 0,
    "M": 1
})

# Corrélations entre les variables explicatives
X = data.drop(columns="diagnosis")
y = data["diagnosis"]

corr = X.corr()

plt.figure(figsize=(18, 15))
sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1
)
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
#________________________________split (data pour le test et pour l'entrainement)____________________________________

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


# ============================================================
#  MODELE XGBOOST A : 30 VARIABLES
# ============================================================

model_A = XGBClassifier(
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
    n_jobs=-1
)

# Entraînement
model_A.fit(X_train, y_train)

print("Entraînement terminé.")

# ============================================================
#  VALIDATION CROISEE
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

cv_results = cross_validate(
    model_A,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring,
    n_jobs=1,
    return_train_score=False
)

print("\n===== VALIDATION CROISEE (5 PLIS) =====")

for metric in scoring:
    scores = cv_results[f"test_{metric}"]

    print(
        f"{metric.upper():10s} : "
        f"{scores.mean():.4f} "
        f"+/- {scores.std():.4f}"
    )

# ============================================================
# PREDICTIONS SUR LE TEST
# ============================================================

# Classes prédites avec le seuil par défaut de 0.5
y_pred = model_A.predict(X_test)

# Probabilité prédite pour la classe maligne (1)
y_proba = model_A.predict_proba(X_test)[:, 1]

# ============================================================
#  METRIQUES
# ============================================================

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall (sensibilite)": recall_score(y_test, y_pred, zero_division=0),
    "F1-score": f1_score(y_test, y_pred, zero_division=0),
    "ROC-AUC": roc_auc_score(y_test, y_proba)
}

print("\n===== RESULTATS SUR LE TEST =====")

for name, value in metrics.items():
    print(f"{name:25s}: {value:.4f}")

print("\n===== RAPPORT DE CLASSIFICATION =====")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Benigne (B)", "Maligne (M)"],
        digits=4,
        zero_division=0
    )
)

# ============================================================
# MATRICE DE CONFUSION
# ============================================================

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Benigne", "Maligne"],
    yticklabels=["Benigne", "Maligne"]
)

plt.xlabel("Classe prédite")
plt.ylabel("Classe réelle")
plt.title("Matrice de confusion - XGBoost A")
plt.tight_layout()
plt.show()


# ============================================================
# COURBE ROC
# ============================================================

fig, ax = plt.subplots(figsize=(7, 6))

RocCurveDisplay.from_predictions(
    y_test,
    y_proba,
    name="XGBoost A",
    ax=ax
)

ax.plot(
    [0, 1], [0, 1],
    linestyle="--",
    color="gray",
    label="Classifieur aléatoire"
)

ax.set_title("Courbe ROC - XGBoost A")
ax.legend()
plt.tight_layout()
plt.show()


# ============================================================
# COURBE PRECISION-RECALL
# ============================================================

fig, ax = plt.subplots(figsize=(7, 6))

PrecisionRecallDisplay.from_predictions(
    y_test,
    y_proba,
    name="XGBoost A",
    ax=ax
)

ax.set_title("Courbe Precision-Recall - XGBoost A")
plt.tight_layout()
plt.show()


# ============================================================
#  IMPORTANCE DES VARIABLES
# ============================================================

importance = pd.Series(
    model_A.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

plt.figure(figsize=(10, 9))

sns.barplot(
    x=importance.head(15).values,
    y=importance.head(15).index
)

plt.xlabel("Importance (gain relatif par défaut de XGBoost)")
plt.ylabel("Variable")
plt.title("Top 15 des variables importantes - XGBoost A")
plt.tight_layout()
plt.show()

print("\n===== TOP 15 DES VARIABLES =====")
print(importance.head(15))
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

# Validation croisée stratifiée à 5 folds
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1,
    random_state=42,
    n_jobs=-1
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

results = cross_validate(
    model,
    X_train,  # Pour B ou C, utiliser leurs variables sélectionnées
    y_train,
    cv=cv,
    scoring=scoring,
    n_jobs=-1
)

print("\n===== VALIDATION CROISÉE (5 FOLDS) =====")

for metric in scoring:
    scores = results[f"test_{metric}"]
    print(
        f"{metric.upper():<12}: "
        f"{scores.mean():.4f} +/- {scores.std():.4f}"
    )
