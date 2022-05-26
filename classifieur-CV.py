#------
# Utilisation : python3 classifieur-CV.py data
# argv[1] = Dossier principal, avec un dossier pour chaque classe avec les fichiers texte.
#------
import sys
# On ne peut utiliser le paramètre "zero_division" avec cross_validate. Donc, on élimine les warnings. 
import warnings
warnings.filterwarnings('ignore')
import numpy as np

# Pour la visualisation des résultats
import seaborn as sns
import matplotlib.pyplot as plt

import sklearn.datasets
# Pour vectoriser les texts et les transformer en sacs de mots.
import sklearn.feature_extraction.text
# Modèles
import sklearn.model_selection
# Importation des classifieurs
import sklearn.neighbors, sklearn.tree, sklearn.svm, sklearn.naive_bayes
from sklearn.ensemble import RandomForestClassifier


# Validation croisée
# Indices pour diviser les corpus en test/train corpus. StratifiedKFold est similaire à KFold, mais il maintient  la proportion de classes dans chaque division
from sklearn.model_selection import cross_validate, cross_val_predict, StratifiedKFold, KFold
# Mesures
from sklearn.metrics import confusion_matrix

# Fonction principal qui prend comme argument le classifier (déjà chargé), le dataset (corpus vectorisé et transformé) et les classes.
# Validation croisée : 10 blocs (folds) pour évaluer les résultats avec les mesures d'exactitude, précision, rappel et F-mesure.
# On utilise StratifiedKFold pour mantenier les classes équilibrées.
def classification(classifier, data_set, data_target):
    mesures = ['accuracy','precision_macro', 'recall_macro', 'f1_macro','precision_weighted','recall_weighted','f1_weighted']
    scores = cross_validate(classifier, data_set, data_target, cv=StratifiedKFold(n_splits=10), scoring=mesures) #StratifiedKFold
    # Pour générer une matrice de confusion, nous avons besoin des scores de prediction.
    y_pred = cross_val_predict(classifier, data_set, data_target, cv=StratifiedKFold(n_splits=10))

    matrice_confusion(data_target,y_pred)

    # On calcul la moyenne de chaque mesure et son écart type
    print("%0.3f exactitude avec un écart type de %0.2f" % (scores['test_accuracy'].mean(), scores['test_accuracy'].std()))
    print("------------------------------------------")
    print("%0.3f précision avec un écart type de %0.2f" % (scores['test_precision_macro'].mean(), scores['test_precision_macro'].std()))
    print("%0.3f rappel avec un écart type de %0.2f" % (scores['test_recall_macro'].mean(), scores['test_recall_macro'].std()))
    print("%0.3f f-mesure avec un écart type de %0.2f" % (scores['test_f1_macro'].mean(), scores['test_f1_macro'].std()))
    print("------------------------------------------")
    print("%0.3f précision pondérée avec un écart type de %0.2f" % (scores['test_precision_weighted'].mean(), scores['test_precision_macro'].std()))
    print("%0.3f rappel pondérée avec un écart type de %0.2f" % (scores['test_recall_weighted'].mean(), scores['test_recall_macro'].std()))
    print("%0.3f f-mesure pondérée avec un écart type de %0.2f" % (scores['test_f1_weighted'].mean(), scores['test_f1_macro'].std()))
    print("------------------------------------------")

# Fonction pour visualiser la matrice de confusion avec ses étiquettes. 
def matrice_confusion(data_target,y_pred):
    cm = confusion_matrix(data_target, y_pred)
    plt.title("Matrice de confusion")
    x_axis_labels = ["action","comedy","horror","sci-fi"]
    y_axis_labels = ["action","comedy","horror","sci-fi"]
    sns.heatmap(cm, fmt='g', annot=True,center=90,xticklabels=x_axis_labels, yticklabels=y_axis_labels)
    plt.show()

# ------------------------------ Programme principal -------------------------------------------
# Chargement de données
dataset = sklearn.datasets.load_files(f"{sys.argv[1]}", encoding="utf-8", shuffle=True)

# Chargement de vectorisateur
# Trois options à tester : representation binaire, par la fréquence absolu et du type TF-IDF
# On utilise TfidVectorizer car il combine la vectorization avec TfidfTransformer, une représentation selon la fréquences des items 

#vectorizer = sklearn.feature_extraction.text.CountVectorizer(binary=True)
#vectorizer = sklearn.feature_extraction.text.CountVectorizer(min_df=2)
vectorizer =  sklearn.feature_extraction.text.TfidfVectorizer(min_df=2, ngram_range=(1,3))

# Vectorisation et son résultat : sac de mots
counts = vectorizer.fit_transform(dataset.data)

# Boucle pour lancer les classifieur et pour avoir un bon affichage
for classifier in (sklearn.neighbors.KNeighborsClassifier(n_neighbors=15, weights='distance'),
                   sklearn.ensemble.RandomForestClassifier(max_depth=5),
                   sklearn.naive_bayes.MultinomialNB(),
                   sklearn.svm.LinearSVC(C=0.1)):
    print(type(classifier).__name__)
    classification(classifier, counts, dataset.target)