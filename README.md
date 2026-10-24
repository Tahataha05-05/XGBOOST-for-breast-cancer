# XGBoost pour la classification des tumeurs mammaires

## 1. Introduction et problématique

### 1.1 Objectif
Le projet vise à construire un modèle de machine learning capable de distinguer deux catégories de tumeurs à partir de données numériques issues d’analyses et d’images de masses mammaires :

- Bénigne : absence de caractère malin
- Maligne : présence d’un caractère malin

L’objectif principal est d’évaluer l’impact de la sélection des variables explicatives sur la performance d’un modèle XGBoost, tout en cherchant à réduire sa complexité sans dégrader significativement ses résultats de classification.

### 1.2 Algorithme XGBoost
XGBoost est un algorithme de machine learning très performant, largement utilisé pour les tâches de classification et de régression. Il repose sur l’assemblage de nombreux arbres de décision, chacun corrigeant les erreurs des arbres précédents afin d’améliorer progressivement la précision du modèle.

### 1.3 Problématique
Dans quelle mesure la sélection des variables explicatives permet-elle de réduire la complexité d’un modèle XGBoost tout en maintenant des performances de classification comparables ?

---

## 2. Présentation du jeu de données

Le projet utilise le jeu de données Breast Cancer Wisconsin (Diagnostic), composé de 569 observations. Le fichier initial comporte 32 colonnes, dont deux colonnes non explicatives :

- id
- Unnamed: 32

En excluant ces deux colonnes, l’analyse porte sur 30 variables explicatives.

### 2.1 Répartition des classes
La distribution des observations entre les deux catégories de tumeurs est la suivante :

| Catégorie | Nombre d’observations | Pourcentage |
|----------|----------------------|-------------|
| Bénigne  | 357                  | 62,74 %     |
| Maligne  | 212                  | 37,26 %     |

### 2.2 Description des variables
Les variables numériques décrivent plusieurs caractéristiques des cellules et des noyaux, notamment :

- taille
- texture
- périmètre
- surface
- compacité
- concavité
- symétrie
- dimension fractale

Pour chaque famille de caractéristiques, le jeu de données fournit plusieurs mesures :

- mean : moyenne
- se : erreur standard
- worst : valeur extrême

---

## 3. Méthodologie

Pour analyser l’effet de la sélection des variables, nous comparons trois versions du même modèle XGBoost :

- Modèle A : référence, utilisant les 30 variables explicatives
- Modèle B : réduction des variables fortement corrélées entre elles pour limiter la redondance
- Modèle C : sélection des sept variables les plus importantes selon l’importance par permutation

### 3.1 Critères d’évaluation
Les performances sont évaluées à l’aide de plusieurs métriques :

- Accuracy : proportion de prédictions correctes
- Precision : capacité du modèle à prédire correctement une classe
- Recall : capacité du modèle à retrouver les observations d’une classe
- F1-score : moyenne harmonique entre précision et rappel
- ROC-AUC : capacité du modèle à distinguer les deux classes selon différents seuils de décision

Nous comparons en priorité les résultats moyens obtenus par validation croisée, puis nous les confrontons aux performances observées sur le jeu de test.

### 3.2 Objectif de la comparaison
L’objectif est de vérifier si un modèle plus compact, utilisant moins de variables, peut conserver des performances de classification proches de celles du modèle de référence.

---

## 4. Comparaison des modèles

Le tableau ci-dessous présente les résultats moyens obtenus lors de la validation croisée.

| Métrique | Modèle A | Modèle B | Modèle C |
|----------|----------|----------|----------|
| Accuracy | 96,92 % | 94,73 % | 96,92 % |
| Precision | 96,48 % | 94,57 % | 95,88 % |
| Recall | 95,29 % | 91,18 % | 95,88 % |
| F1-score | 95,85 % | 92,81 % | 95,87 % |
| ROC-AUC | 99,34 % | 98,88 % | 99,36 % |
| Nombre de variables | 30 | Réduit | 7 |

### 4.1 Interprétation des résultats
Les résultats montrent que le Modèle B obtient des performances inférieures à celles du Modèle A sur toutes les métriques. La suppression des variables fortement corrélées n’a donc pas amélioré la classification dans cette expérience.

Le Modèle C, quant à lui, conserve des performances très proches de celles du Modèle A :

- Accuracy presque identique
- F1-score quasiment équivalent
- Recall et ROC-AUC très proches
- Réduction considérable du nombre de variables : 30 à 7

Ce modèle apparaît donc comme une alternative plus compacte, tout en maintenant une qualité de classification comparable au modèle de référence.

Cependant, les écarts observés restent modestes et ne permettent pas, à eux seuls, de conclure à une supériorité statistique du Modèle C. Ils suggèrent néanmoins qu’une sélection ciblée des variables peut réduire la complexité du modèle sans dégrader sensiblement ses performances.

---

## 5. Limites méthodologiques

Plusieurs limites doivent être prises en compte :

1. Sélection des variables et validation croisée  
   Pour le Modèle C, les variables ont été sélectionnées à partir de l’ensemble d’entraînement avant la validation croisée. Cela signifie que les folds ont indirectement participé au choix des variables. Pour une estimation plus rigoureuse, la sélection doit être répétée à l’intérieur de chaque fold, en utilisant uniquement le sous-ensemble d’entraînement de ce fold.

2. Comparaison statistique  
   Les différences entre les modèles A et C sont modestes. Il ne faut donc pas les présenter comme une amélioration statistiquement démontrée. Une comparaison sur les mêmes partitions et une procédure de validation adaptée sont nécessaires pour tirer une conclusion plus solide.

3. Jeu de test  
   Le jeu de test doit rester réservé à l’évaluation finale. Il ne doit ni servir à choisir les variables, ni à ajuster les hyperparamètres, ni à sélectionner le meilleur modèle.

4. Portée médicale  
   Cette étude est réalisée dans un cadre pédagogique. Les résultats obtenus ne suffisent pas à affirmer qu’un modèle serait fiable pour une utilisation clinique réelle.

---

## 6. Conclusion

Dans cette expérience, la suppression des variables fortement corrélées n’a pas amélioré les performances de XGBoost. À l’inverse, l’importance par permutation a permis de retenir sept variables et de construire un modèle plus compact, dont les performances restent proches de celles du modèle de référence sur le jeu de test.

Ce résultat reste à confirmer par une validation sans fuite d’information, dans laquelle la sélection des variables est répétée à l’intérieur de chaque fold. Cette approche permettrait d’obtenir une estimation plus fiable de l’impact réel de la sélection des variables sur les performances du modèle.
