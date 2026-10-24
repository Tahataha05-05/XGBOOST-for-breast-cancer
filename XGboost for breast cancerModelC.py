import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.inspection import permutation_importance
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

data = data.drop(columns=["id", "Unnamed: 32"], errors="ignore")

if data["diagnosis"].dtype == "object":
    data["diagnosis"] = data["diagnosis"].map({"B": 0, "M": 1})

if data["diagnosis"].isnull().any():
    raise ValueError("La cible contient des valeurs inconnues.")

X = data.drop(columns="diagnosis")
y = data["diagnosis"].astype(int)

if X.isnull().any().any():
    raise ValueError("Le dataset contient des valeurs manquantes.")

if not np.isfinite(X.to_numpy(dtype=float)).all():
    raise ValueError("Le dataset contient des valeurs infinies.")

print("Dimensions :", data.shape)
print("Nombre de variables initiales :", X.shape[1])
print("\nRépartition des diagnostics :")
print(y.value_counts())

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

print("\nTrain :", X_train.shape)
print("Test :", X_test.shape)

# ============================================================
# 3. MODELE DE REFERENCE POUR LA SELECTION
# ============================================================

model_reference = XGBClassifier(
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
model_reference.fit(X_train, y_train)

# ============================================================
# 4. IMPORTANCE PAR PERMUTATION
# ============================================================

perm = permutation_importance(
    model_reference,
    X_train,
    y_train,
    scoring="roc_auc",
    n_repeats=20,
    random_state=42,
    n_jobs=-1,
)

importance = pd.DataFrame(
    {
        "variable": X_train.columns,
        "importance_moyenne": perm.importances_mean,
        "importance_std": perm.importances_std,
    }
).sort_values("importance_moyenne", ascending=False)

print("\n===== IMPORTANCE PAR PERMUTATION =====")
print(importance.to_string(index=False))

plt.figure(figsize=(10, 9))
sns.barplot(data=importance.head(20), x="importance_moyenne", y="variable")
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Baisse moyenne de ROC-AUC après permutation")
plt.ylabel("Variable")
plt.title("Importance des variables - modèle de référence")
plt.tight_layout()
plt.show()

# ============================================================
# 5. SÉLECTION DES VARIABLES
# ============================================================

selected_features = importance.loc[
    importance["importance_moyenne"] > 0,
    "variable",
].tolist()

if len(selected_features) == 0:
    raise ValueError("Aucune variable n'a une importance positive. Il faut revoir le critère de sélection.")

print("\n===== VARIABLES SÉLECTIONNÉES =====")
print("Nombre de variables :", len(selected_features))
print(selected_features)

X_train_C = X_train[selected_features].copy()
X_test_C = X_test[selected_features].copy()

# ============================================================
# 6. ENTRAÎNEMENT DU MODÈLE C
# ============================================================

model_C = XGBClassifier(
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
model_C.fit(X_train_C, y_train)

# ============================================================
# 7. PRÉDICTIONS ET MÉTRIQUES
# ============================================================

y_pred_C = model_C.predict(X_test_C)
y_proba_C = model_C.predict_proba(X_test_C)[:, 1]

metrics_C = {
    "Accuracy": accuracy_score(y_test, y_pred_C),
    "Precision": precision_score(y_test, y_pred_C, zero_division=0),
    "Recall": recall_score(y_test, y_pred_C, zero_division=0),
    "F1-score": f1_score(y_test, y_pred_C, zero_division=0),
    "ROC-AUC": roc_auc_score(y_test, y_proba_C),
}

print("\n===== RESULTATS DU MODÈLE C =====")
for name, value in metrics_C.items():
    print(f"{name:12s}: {value:.4f}")

print("\n===== RAPPORT DE CLASSIFICATION =====")
print(
    classification_report(
        y_test,
        y_pred_C,
        labels=[0, 1],
        target_names=["Benigne", "Maligne"],
        digits=4,
        zero_division=0,
    )
)

# Matrice de confusion
cm_C = confusion_matrix(y_test, y_pred_C, labels=[0, 1])
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_C,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Benigne", "Maligne"],
    yticklabels=["Benigne", "Maligne"],
)
plt.xlabel("Classe prédite")
plt.ylabel("Classe réelle")
plt.title("Matrice de confusion - modèle C")
plt.tight_layout()
plt.show()

# Courbe ROC
fig, ax = plt.subplots(figsize=(7, 6))
RocCurveDisplay.from_predictions(y_test, y_proba_C, name="XGBoost C", ax=ax)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Classifieur aléatoire")
ax.set_title("Courbe ROC - modèle C")
ax.legend()
plt.tight_layout()
plt.show()

# Courbe Precision-Recall
fig, ax = plt.subplots(figsize=(7, 6))
PrecisionRecallDisplay.from_predictions(y_test, y_proba_C, name="XGBoost C", ax=ax)
ax.set_title("Courbe Precision-Recall - modèle C")
plt.tight_layout()
plt.show()

# Importance des variables du modèle final
importance_finale = pd.Series(
    model_C.feature_importances_,
    index=X_train_C.columns,
).sort_values(ascending=False)
plt.figure(figsize=(10, 8))
sns.barplot(x=importance_finale.values, y=importance_finale.index)
plt.xlabel("Importance relative")
plt.ylabel("Variable")
plt.title("Importance des variables - modèle C final")
plt.tight_layout()
plt.show()

print("\n===== IMPORTANCE DU MODÈLE C =====")
print(importance_finale)

# ============================================================
# 8. VALIDATION CROISÉE
# ============================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

cv_results = cross_validate(
    model_C,
    X_train_C,
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