# Fouille de textes
This is the final project for the text mining course (M1) made by me and Santiago Herrera Yanez. Objective is to classify movie synopses using Scikit learn and Weka, collected using Beautiful Soup on Rotten Tomatoes. 

Professor: Loïc Grobol

Introduction
---
La fouille de textes et l’apprentissage automatique occupent une place essentielle dans le domaine du traitement automatique des langues depuis qu’il est possible d'accéder à de grandes quantités de données. Il s’agit principalement de la construction de programmes qui permettent de traiter des données textuelles de façon numérique, automatique et quantitative, en visant des objectifs concrets, notamment, la classification de textes.

Dans le cadre de ce projet, à partir des démarches de l’apprentissage automatique supervisé, nous avons mis en place différents algorithmes de classification pour les tester. Plus spécifiquement, nous voulions classer binairement des synopsis de films en fonction de leur genre en utilisant diverses ressources. Dans ce travail, nous décrirons  les différents algorithmes de classification via le logiciel Weka et à travers les fonctions et algorithmes disponibles dans la bibliothèque scikit-learn en Python. Notre but, enfin, est de pouvoir montrer et expliquer le fonctionnement de ces différents classifieurs et les évaluer de manière qualitative et quantitative.

Construction du corpus et formatage
---
Notre corpus final est constitué de 818 synopsis de films en anglais, récoltés sur le site web Rotten Tomatoes (ex. synopsis d’un film d’horreur) de façon semi-automatique. Les synopsis sont classées en quatre genres, selon la même distribution que le site Rotten Tomatoes propose. Nous avons choisi de travailler avec 4 classes ou genres : action, horreur, sci-fi et comédie.
Nous avons extrait chaque synopsis avec notre script Python “crawling_tomatoes.py”, qui utilise comme outils principaux le parseur HTML de la bibliothèque “Beautifulsoup” et des expressions régulières. En choisissant des sites avec plusieurs films listés (ex. Top 100 horror movies), notre méthode consistait à récupérer les liens associés à chaque synopsis pour en extraire les contenus textuels et les sauvegarder dans un seul fichier texte, avec des balises comme délimiteurs.
Pour avoir une distribution égale de genre, nous avons extrait, au début, environ  220 synopsis par genre. Néanmoins, nous nous sommes retrouvés face à notre premier problème. Bien que nous ayons suivi les classifications déterminées par le site Rotten Tomatoes, nous nous sommes rendu compte que les films pouvaient être assignés à plusieurs genres. Notamment, les films d'action sont difficilement séparables des films de science fiction. Pour éviter ce problème, étant donné que nous ne réalisions pas de la classification multi-class, nous avons éviter les doublons, en plaçant manuellement les synopsis répétés dans le genre qui nous semblait le mieux convenir.
Pour obtenir de meilleurs résultats, au moment de mettre en place les classifieurs, nous avons écrit un deuxième script pour rendre notre corpus plus propre. Sachant que nous représentons nos données de manière tabulaire, en différents types de fréquences, il était nécessaire de réduire le bruit dans nos données, éliminer les mots plus répétés, notamment, les mots vides, et normaliser le contenu textuel. Pour cela, nous avons écrit un deuxième script appelé “preprocessing.py. À partir de ce celui-ci, nous avons, premièrement, lemmatiser le corpus et filtrer la ponctuation et les mots vides en utilisant le modèle “en_core_web_md” de l’anglais de spaCy. Nous avons modifié la liste de mots vides de spaCy, pour ajouter quelques mots clés comme “action”, “comedy”,  “horror”, etc. afin d'éviter de biaiser la classification. Malgré ça, nous avons décidé de filtrer de nouveau les mots vides à partir d’un fichier texte parce que la liste de spaCy n'était pas exhaustive. Enfin, avec le même script,  les résultats, c’est-à-dire, chaque synopsis prétraitée, a été sauvegardé dans un seul fichier, dans le dossier du genre. Cette disposition de fichiers est nécessaire pour lancer sur notre corpus le script “vectorisation.py” afin de travailler avec Weka ou avec les classifieurs de scikit-learn.

Finalement notre corpus, composé de 818 synopsis, est d’une taille totale de 31,241 tokens, avec une moyenne de 38 mots par synopsis (calculs faits à partir du simple script “count_fq.py”). Voici la distribution finale de notre corpus:

<img width="404" alt="pic0" src="https://user-images.githubusercontent.com/77155381/170557454-a5464c5b-b43e-40a6-ab63-5eb52f1374be.PNG">

En ce qui concerne le statut juridique des données que nous avons récoltées, tout le contenu de Rotten Tomatoes est protégé par copyright et appartient à l'entreprise Fandango. Néanmoins, les données sont exploitables si elles sont destinées à des fins non-commerciales et personnelles (voir, Terms of use). Les informations et données récoltées tout au long du projet n’ont pas d’autre fin que de servir à notre éducation et, en effet, leur usage est strictement personnel. 

Méthodologie
---
Une fois notre corpus constitué et que nous ayons réglé ses problèmes les plus saillants, nous avons décidé d'établir deux façons différentes de mettre en œuvre les divers classifieurs.

D’une part, nous avons travaillé depuis le logiciel Weka, une fois vectorisé notre corpus avec le script donné á telle fin par le Prof. Loïc Grobol. Comme résultat, on obtient des données tabulaires selon la fréquence absolue de chaque terme. Nous avons aussi créé une représentation booléenne de notre corpus, qui finalement n'a pas été utilisée. 

D’autre part, nous avons écrit un script Python qui utilise les différents classifieurs et fonctions de la bibliothèque scikit-learn. Ce script a aussi été écrit sur la base d’un exemple donné par notre professeur. L’utilisation de cette bibliothèque nous donne plus de liberté au moment de normaliser et représenter les données sous forme de sac de mots. Ainsi, par exemple, dans le script Python nous avons en général privilégié une représentation du type TF-IDF, parce qu'elle donne souvent de meilleurs résultats. En revanche, Weka à partir de la visualisation claire des données et, par exemple, grâce à l’affichage des arbres de décisions, nous a aidé à mieux comprendre notre corpus et à le rendre encore plus propre (ex. élimination de ponctuation). La combinaison de deux méthodes, Weka et scikit-learn, nous a permis d’avoir une compréhension globale de notre travail et d’obtenir de meilleurs résultats. Pour apprendre à utiliser les deux méthodes, surtout le fonctionnement de la bibliothèque scikit-learn, nous avons dû lire de la documentation concernant les algorithmes et les fonctions.

À propos de nos choix, nous avons essayé de tester à chaque fois les classifieurs à travers les deux méthodes, en testant différentes options et paramètres. Dans certains cas (ZeroR, RandomForest, etc.), les classifieurs ont été testé par une seule méthode. Nous avons toujours choisi les paramètres et les modes de représentation qui nous donnaient de meilleurs résultats. Notamment, sur le script Python, nous avons parfois privilégié l’utilisation de bigrammes ou trigrammes, ou  l'établissement d’un seuil minimum de la fréquence de chaque token pour obtenir des résultats plus favorables. Tout au long de ce travail, nous détaillerons les options utilisées. Si les paramètres des algorithmes ou de la vectorisation/représentation ne sont pas expressément indiqués pour un classifieur, c'est parce que nous avons utilisé les paramètres par défaut.

En ce qui concerne l'évaluation, elle a été réalisée de la même façon via Weka et à travers scikit-learn. Nous avons mis en place la validation croisée avec 10 blocs (folds). Sur scikit-learn, nous avons utilisé spécialement la fonction StratifiedKFold() pour garantir la meilleure distribution possible de données. Entre les mesures d'évaluation, nous présentons dans ce travail la précision, le rappel et la F-mesure (macro et pondérées), l’exactitude et même des matrices de confusion pour mieux comprendre le comportement de chaque classifieur. Nous comparerons aussi les résultats entre les différents classifieurs selon ces mesures en pondérant aussi les avantages ou les inconvénients de chacun, au moment de sa mise en place.

Enfin, nous avons réalisé le travail en deux parts égales. 

Expériences réalisées
---
1)Classe Majoritaire (ZeroR)


L'algorithme de “classe majoritaire”, qui s'appelle ZeroR sur Weka, fournit une méthode de classification simple. Il est appelé “simple” parce qu'il prédit la classe qui a la plus grande fréquence. Il est utile pour déterminer une base de référence pour d'autres méthodes de classification. Sur Weka c’est la méthode de classification par défaut

<img width="469" alt="fdt1" src="https://user-images.githubusercontent.com/77155381/170557703-90b2aa48-7a16-4c8c-89bd-7727e5bb7a3f.PNG">

Comme prévu, la classe majoritaire ne donne pas de bons résultats. Lorsque nous examinons la précision détaillée par classe, nous constatons que le plus grand nombre de fichiers se trouvent en sci-fi.  Cependant, la “PRC Area” montre que le corpus est en grande partie équilibré puisque chaque classe a environ 25% d'occurrences. Il peut être pris comme résultat de base pour utiliser d'autres algorithmes. 

<img width="444" alt="fdt2" src="https://user-images.githubusercontent.com/77155381/170557802-be69f426-35ce-4237-a799-9cbc273d4f5c.PNG">

2)K-plus proches voisins

Cet algorithme, qualifié de "paresseu", associe des nouvelles données à la classe majoritaire du sous-ensemble le plus proche, composé par les k données voisines. Les k-plus proches voisins sont, donc, les k valeurs que se trouvent le plus proches l'une de l'autre.

Le valeur de k ne doit pas être un multiple du nombre de classes parce que l’on a besoin de forcer l'existence d’une classe majoritaire. En outre, le paramétrage de la valeur de k dépend aussi de la quantité des données disponibles. Par exemple, pour des raisons assez claires, elle ne peut pas être plus élevée que la quantité de données.

L’obtention d’assez bons résultats en utilisant une représentation TF-IDF, en comparaison aux autres algorithmes que nous avons essayé tout au long de ce travail,  éveille en nous une question. Nous savons que l’utilisation de k-plus proches voisins est fréquent dans le service de recommandation, en combinant des données à recommander avec d’information personnelle de l’utilisateur. Bien que ça ne soit pas le cas ici, étant donné que nos données sont des textes qui décrivent des films (extraits d’un site de critiques) et qu’ils sont très fortement liés à la tâche de recommandation. Est-ce que le format de nos données privilégie l’utilisation de cet algorithme? 

Enfin, Il s’agit d’un modèle facile à développer et d'une interprétation simple, qui malgré les bons résultats que nous avons obtenus, n’est ni plus efficace pour une grande quantité de données, ni pour une taille de données assez limitée comme la nôtre. 

- <img width="509" alt="fdt3" src="https://user-images.githubusercontent.com/77155381/170558175-ccd32c5d-91cf-4f33-8ff0-79567babde08.PNG">
<img width="301" alt="fdt4" src="https://user-images.githubusercontent.com/77155381/170558246-88b2e5dd-b23e-4103-8d46-2f0210d24e5f.PNG">

- Résultats et appréciations
À partir de fonctions de scikit-learn, nous avons choisi de présenter les données selon une fréquence pondérée en utilisant une représentation en sac de mots de type TF-IDF. 
Les résultats sont assez intéressants surtout si on les compare avec ceux obtenus avec Weka, à travers des représentations booléennes ou à partir des fréquences absolues. On observe que les résultats de notre script Python sont très supérieurs. 
Tandis que l'algorithme peut trouver des sous-ensembles de k-voisins si la fréquence est pondérée et la représentation inclut l’utilisation de bigrammes et trigrammes, quand la représentation de données est plus plate, les résultats ressemblent à ceux donnés par l'algorithme ZeroR.

Arbres de décisions 
---
Nous avons appliqué deux algorithmes différents de la famille des arbres de décisions. Premièrement, à travers Weka, nous avons utilisé le J48 (similaire à C4.5). Deuxièmement, via scikit-learn, le classifieur Forest Random.

Les arbres de décision fonctionnent par des méthodes symboliques et cherchent à classer les données à partir de celles qui sont le plus informatives ou le plus discriminantes pour ensuite pouvoir évaluer quelles sont celles qui diffèrent le moins dans notre corpus. Pour trouver quels sont les éléments le plus discriminant, l'algorithme évalue la différence ou l'homogénéité d’une donnée en calculant la fonction Gini ou l’entropie. Cette évaluation se réalise aussi grâce à l’utilisation des seuils dans le cas où un attribut ou un trait parmi nos données n’est pas définitoire pour classer notre corpus. Voici un exemple extrait de l’évaluation que nous avons fait :
<img width="278" alt="fdt5" src="https://user-images.githubusercontent.com/77155381/170558446-b3b76b7c-0d57-4b19-a327-e321789cdd17.PNG">

Le mot le plus discriminant dans notre corpus est le mot “alien”. Un texte avec ce mot sera classé comme appartenant à la classe “sci-fi”. Un peu plus haut, on peut voir que si les mots “victime” et “future” sont cooccurrents, le texte sera classé en tant que “sci-fi”.

En ce qui concerne les algorithmes utilisés, le J48 privilégie la construction des arbres “imparfaits”, qui maximisent dans la limite du possible la recherche des traits qui permettent de classer nos données. De cette façon, on évite le surapprentissage et on maximise nos résultats. Le RandomForest construit, en revanche, plusieurs arbres dont les résultats sont combinés pour prendre la meilleure décision de classification.

- Résultats et appréciations

L’utilisation de l’algorithme J48, malgré son instabilité, nous a permis de bien comprendre la composition de notre corpus et de nettoyer certaines imperfections.
    
En ce qui concerne les résultats, ils sont meilleurs que l'algorithme de classe majoritaire et que la version sans la représentation du type TF-IDF de notre classifieur K-plus proches voisins. Avec RandomForest on arrive à une précision de plus de 50%. En regardant de plus près la matrice de confusion de ce deuxième algorithme, on voit comment la classe plus problématique au moment de classer est “action”, confirmant l’impression que nous avons eu au moment de récolter le corpus. Il s’agit d’une classe peu définie, qui se chevauche facilement avec “sci-fi” ou même avec “comedy”.


<img width="503" alt="fdt6" src="https://user-images.githubusercontent.com/77155381/170558571-3f556b88-bf66-4bee-9fa2-c5be1f82dcb9.PNG">
<img width="284" alt="fdt7" src="https://user-images.githubusercontent.com/77155381/170558684-f7a7c974-a51d-4afb-bb16-e3cfd6196502.PNG">

Naïve Bayes 
---
Les méthodes de Naïve Bayes sont un groupe d’algorithmes d’apprentissage automatique basés sur des probabilités conditionnelles. Ils sont appelés “naïve”, car ils sont construits sur la présomption que chaque variable d'entrée est indépendante. Il faut souligner que la conversion d’un texte en un sac de mot est fondée aussi sur cette présomption.

On a utilisé la version classique Multinomial de Naïve Bayes, qui décrit la probabilité d'observer des occurrences parmi un certain nombre de catégories. La méthode multinomiale est la plus appropriée pour les caractéristiques qui représentent des fréquences.

En outre, les classifieurs du type Naïve Bayes sont faciles à implémenter et à mettre à jour. Il s’agit donc d’un algorithme assez pratique pour des tâches comme la nôtre, pour sa potentialité et sa petite complexité.

- Résultats et appréciations

Avec cet algorithme depuis Weka, en utilisant les paramètres par défaut, nous obtenons des résultats relativement satisfaisants en termes de précision et d’autres mesures. Si l'on compare le résultat obtenu avec ZeroR, on peut dire que le base a été triplé avec Naïve Bayes. L'application de cet algorithme donne encore de meilleurs résultats s’il est appliqué sur des données représentées selon une fréquence pondérée du type TF-IDF. 

<img width="345" alt="fdt8" src="https://user-images.githubusercontent.com/77155381/170558811-e3f392f1-c587-4916-b059-3f8ddf982292.PNG">


SVM : Support Vector Machines ou Séparateurs à vaste marges
---

Ce classifieur de haute complexité, qui a donné pendant plusieurs années les meilleurs résultats parmi les algorithmes existant, travaille à partir des notions de hyperplan et de marge. Pour bien classer les données, le classifieur essaie de trouver le meilleur hyperplans pour diviser binairement et successivement les données. Le meilleur hyperplan est celui qui peut tracer une division nette entre les éléments de deux classes, avec le plus large marge possible entre les vecteurs pris en compte pour tracer l’hyperplan. Maintenir une marge vaste garantit la généralisation de la classification et évite le sur-apprentissage. 

Néanmoins, comme dans notre cas, les données ne sont pas toujours faciles à séparer de façon nette. C’est pour cela que le classifieur incorpore deux “astuces” : 1) des pénalités pour chaque élément qui n’a pas pu être bien classé et 2) la possibilité de représenter les données dans un dimension plus large pour trouver une hyperplan capable de bien séparer les données. 
<img width="236" alt="fdt9" src="https://user-images.githubusercontent.com/77155381/170558947-ee3dcb28-6306-4d65-afbe-0a9ad75a55a3.PNG">

- Résultats et appréciations

Le classifieur SVM linear de la bibliothèque scikit-learn, comme nous l’attendions, nous donne de très bons résultats, avec plus de 70% de précision et avec des mesures situées autour de ce nombre. On obtient des résultats proches à partir de Weka et son l’algorithme SMO. Nous croyons que la différence se doit à la représentation des données, sachant que la représentation TF-IDF permet plus de finesse dans le sac de mots, en prenant en compte les fréquences qui sont plus discriminantes. 

En regardant la matrice de confusion, on reconstate que les synopsis de la classe “action” sont toujours les plus problématiques à classer. Mais elle n’est pas la seule classe à présenter des résultats peu satisfaisants. “Sci-fi”, comme nous l'avons déjà dit, se chevauche souvent à la classe “action”. “Comedy” est la catégorie la mieux classée.

Bien que nous ayons obtenu de bons résultats en comparaison à d'autres essais, le classifieur SVM n’est pas un algorithme qui se caractérise pour être incrémental. À cause de la petite différence à la faveur du classifieur Naïve Bayes, nous nous demandons si le classifieur SVM est une bonne option.


Conclusions
---
Dans ce projet, nous avons mis en place différents algorithmes de classification et nous avons fait une comparaison de leur fonctionnement et de leur performance. Tout au long de ce travail, nous les avons comparés en utilisant le logiciel Weka, et les fonctions et algorithmes disponibles dans la bibliothèque scikit-learn.

En termes de corpus, la similarité entre les genres a affecté considérablement la  performance des algorithmes. Par exemple, les catégories “action” et la “sci-fi” partagent beaucoup de similarités, il est donc difficile de les distinguer, ce qui cause plus de bruit entre les deux genres. Ce problème s’est posé avec les autres genres mais en moindre quantité. En revanche, nous croyons avoir obtenu un corpus assez propre grâce aux pré-traitements effectués. 

Si nous comparons le logiciel Weka et les possibilités que la bibliothèque scikit-learn offrent, nous pouvons remarquer rapidement quelques différences et similarités. Weka est plus pratique à utiliser et à prendre en main. Il y a beaucoup de classifieurs disponibles et son interface amicale facilite le changement de paramètres. Toutefois, scikit-learn offre également de nombreuses possibilités et une plus grande liberté en termes de traitement de données. L’utilisation du python est, bien sûr, un gros avantage. Par exemple, l’incorporation des paquets seaborn nous permet de créer des graphes plus facilement interprétables. 

Dans le cadre de ce modeste travail, la plus grande différence entre Weka et scikit-learn est que la bibliothèque du python donne accès à différentes manières de représenter notre corpus en tant que données tabulaires. Nous avons observé, par exemple, que les classifieurs fonctionnent mieux si l’on utilise une représentation TF-IDF. En outre, en général, l’utilisation de bigrammes et trigrammes améliore aussi leur performance, en les fonctions de vectorisation et représentation des données de scikit-learn sont assez faciles de paramétrées. 

Les résultats obtenus tout au long de ce travail, à partir de l'implémentation de différents classifieurs, nous permettent de réfléchir non seulement aux performances de chaque algorithme mais aussi au classifieur le plus approprié pour notre tâche. En effet, parmi tous les classifieurs utilisés, certains sont particulièrement intéressants pour nous. Nous croyons que, étant donné la taille de notre corpus, le classifieur Naïve Bayes est la meilleure option à développer pour diverses raisons : c’est celui qui a donné des résultats comparables et meilleures aux classifieurs de type SVM, ayant la capacité d'être facilement incrémental, même s’il s’agit d’un algorithme moins complexe. En outre, les algorithmes des arbres de décisions ont été une surprise pour nous, par leur lisibilité, car ils nous ont aidés à mieux comprendre notre corpus, et aussi pour ses résultats dans sa version RandomForest. 

Enfin, il y a encore beaucoup à améliorer et différents classifieurs encore à tester. Évidemment, il est nécessaire d'élargir nos corpus pour atteindre de meilleurs résultats. Nous devons aussi améliorer la normalisation de données via scikit-learn. Nous croyons que ce sont deux chemins possibles, entre autres, pour continuer à explorer la tâche de la classification. 
