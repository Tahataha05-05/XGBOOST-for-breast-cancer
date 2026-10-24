1 Introduction et probl´ematique
1.1 Objectif
Le but de ce projet est de construire un mod`ele de machine learning capable de distinguer deux
cat´egories de tumeurs `a partir de donn´ees num´eriques extraites d’analyses et d’images de masses
mammaires :
• B´enigne : absence de caract`ere malin.
• Maligne : pr´esence d’un caract`ere malin.
1.2 Algorithme XGBoost
XGBoost est un algorithme d’intelligence artificielle tr`es puissant, utilis´e pour pr´edire des r´esultats
`a partir de tableaux de donn´ees. Il fonctionne en combinant plusieurs arbres de d´ecision simples,
qui se corrigent les uns apr`es les autres afin de devenir de plus en plus pr´ecis. C’est l’un des outils
les plus populaires et les plus rapides pour r´esoudre des probl`emes de classification ou de pr´ediction
de valeurs num´eriques.
1.3 Probl´ematique
Dans quelle mesure la s´election des variables explicatives permet-elle de r´eduire la complexit´e d’un
mod`ele XGBoost tout en pr´eservant ses performances de classification ?
2
2 Pr´esentation du jeu de donn´ees
Le projet s’appuie sur le jeu de donn´ees Breast Cancer Wisconsin (Diagnostic). Il contient 569
observations. Le fichier initial comporte 32 colonnes, dont les colonnes non explicatives id et
Unnamed: 32. En excluant ces deux colonnes, l’analyse porte sur 30 variables explicatives.
La r´epartition des observations entre les deux cat´egories de tumeurs est pr´esent´ee dans le tableau
ci-dessous.
Table 1: Caract´eristiques principales du jeu de donn´ees
´ El´ement
Valeur
Observations
Colonnes initiales
569
32
Colonnes non explicatives id, Unnamed: 32
Variables explicatives
30
Tumeurs b´enignes
Tumeurs malignes
357 (62,74%)
212 (37,26%)
Les variables num´eriques d´ecrivent plusieurs caract´eristiques des cellules et de leurs noyaux : leur
taille et leur texture, leur p´erim`etre et leur surface, ainsi que leur compacit´e, leur concavit´e, leur
sym´etrie et leur dimension fractale. Pour chaque famille de caract´eristiques, le jeu fournit diff´erentes
mesures : une moyenne (mean), une erreur standard (se) et une mesure des valeurs extrˆemes
(worst).
3 M´ethodologie
Pour ´etudier l’effet de la s´election des variables, nous comparerons trois versions du mˆeme mod`ele
XGBoost. Le mod`ele A servira de r´ef´erence et utilisera les 30 variables explicatives. Dans le mod`ele
B, nous retirerons les variables fortement corr´el´ees entre elles afin de r´eduire la redondance. Pour
le mod`ele C, nous conserverons les sept variables les plus importantes, s´electionn´ees `a partir de leur
importance par permutation.
La comparaison s’appuiera sur plusieurs m´etriques compl´ementaires. L’accuracy indique la part des
pr´edictions correctes parmi toutes les observations. La pr´ecision mesure la fiabilit´e des pr´edictions
d’une classe, tandis que le rappel indique dans quelle mesure les observations de cette classe sont
retrouv´ees. Le F1-score combine pr´ecision et rappel en un seul indicateur. Enfin, le ROC-AUC
´evalue la capacit´e du mod`ele `a distinguer les deux classes lorsque le seuil de d´ecision varie.
Nous examinerons en priorit´e les r´esultats moyens de validation crois´ee, qui permettent de comparer
les mod`eles sur plusieurs partitions des donn´ees d’entraˆınement, puis les confronterons aux r´esultats
obtenus sur le jeu de test. Nous tiendrons ´egalement compte du nombre de variables utilis´ees :
l’objectif est de v´erifier si un mod`ele plus compact peut conserver des performances de classification
proches de celles du mod`ele de r´ef´erence.
Comparaison des mod`eles
Le tableau ci-dessous rassemble les r´esultats moyens de validation crois´ee. Cette comparaison
permet d’observer `a la fois la qualit´e de classification et le nombre de variables conserv´ees par
chaque mod`ele.
Table 2: R´esultats de validation crois´ee et nombre de variables
M´etrique CV
Mod`ele A Mod`ele B Mod`ele C
Accuracy
Precision
Recall
F1-score
ROC-AUC
Nombre de variables
96,92%
96,48%
95,29%
95,85%
99,34%
30
94,73%
94,57%
91,18%
92,81%
98,88%
R´eduit
96,92%
95,88%
95,88%
95,87%
99,36%
7
Les r´esultats montrent que le mod`ele B obtient des scores plus faibles que le mod`ele A pour
l’ensemble des m´etriques. La suppression des variables corr´el´ees n’a donc pas am´elior´e la classi
fication dans cette exp´erience. Le mod`ele C, en revanche, conserve une accuracy et un F1-score
presque identiques `a ceux du mod`ele A ; son recall et son ROC-AUC sont ´egalement tr`es proches.
Il atteint ces r´esultats avec seulement sept variables au lieu de trente, ce qui en fait une option plus
compacte.
Ces ´ecarts restent modestes et ne suffisent pas, `a eux seuls, `a d´emontrer une sup´eriorit´e statistique
du mod`ele C. Ils sugg`erent toutefois qu’une s´election cibl´ee des variables peut r´eduire la complexit´e
sans d´egrader sensiblement les performances observ´ees.
6 Limites m´ethodologiques
1. S´election des variables et validation crois´ee. Pour le mod`ele C, les variables ont ´et´e
s´electionn´ees `a partir de l’ensemble d’entraˆınement avant la validation crois´ee. Les folds de
validation ont donc indirectement particip´e au choix des variables. Pour obtenir une estima
tion plus rigoureuse, la s´election doit ˆetre r´ep´et´ee `a l’int´erieur de chaque fold, en utilisant
uniquement son sous-ensemble d’entraˆınement.
2. Comparaison statistique. Les diff´erences de scores entre les mod`eles A et C sont mod
estes. Il convient donc de ne pas les pr´esenter comme une am´elioration statistiquement
d´emontr´ee. Une comparaison sur les mˆemes partitions et une proc´edure de validation adapt´ee
sont n´ecessaires pour tirer une conclusion plus solide.
3. Jeu de test. Le jeu de test doit rester r´eserv´e `a l’´evaluation finale. Il ne doit servir ni `a
choisir les variables, ni `a r´egler les hyperparam`etres, ni `a s´electionner le meilleur mod`ele.
4. Port´ee m´edicale. Cette ´etude utilise le jeu de donn´ees dans un cadre p´edagogique. Les
r´esultats obtenus ne suffisent pas `a ´etablir qu’un mod`ele serait fiable pour une utilisation
clinique r´eelle.
9
7 Conclusion
Dans cette exp´erience, retirer les variables fortement corr´el´ees n’a pas am´elior´e les performances
de XGBoost. ` A l’inverse, l’importance par permutation a permis de retenir sept variables et de
construire un mod`ele compact, aux r´esultats comparables `a ceux du mod`ele A sur le jeu de test.
Ce r´esultat reste `a confirmer par une validation sans fuite d’information, dans laquelle la s´election
des variables est r´ep´et´ee `a l’int´erieur de chaque fold
