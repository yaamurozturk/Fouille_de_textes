#!/usr/bin/python

#----------------------------------------------------------------
# Utilisation : python3 preprocessing.py
# On a besoin obligatoirement d'avoir recolté les synopsis dans un seul fichier avec le script "crawling_tomatoes.py" dans le dossier "synopsis"
# Il est necessaire d'avoir un fichier avec des mots vides dans chaque ligne
# Enfin, Il faut aussi avoir créé la structure de répertoire suivante avec un dossier par genre :
# Structure nécessaire avec les fichiers : 

# ├── data
# │   ├── action
# │   ├── comedy
# │   ├── horror
# │   └── sci-fi
# ├── synopsis
# │   ├── action_synopsis.txt
# │   ├── comedy_synopsis.txt
# │   ├── horror_synopsis.txt
# │   └── sci-fi_synopsis.txt

# Aclaration : le script est pensé pour travailler sur l'anglais
#----------------------------------------------------------------
import re
import spacy


# Fonction pour écrire dans un nouveau fichier
def text_to_file(filename, string):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(string)

# Fonction pour lire un fichier ligne par ligne      
def file_to_list(filename):
    with open(filename, encoding="UTF-8") as temp:
        tokens_list = []
        for line in temp.readlines():
            tokens_list.append(line.strip("\n"))
        return tokens_list

# Fonction de pré-traitement : tokenisation, lemmatisation et élimination de mots vides et de la ponctuation.
def preprocessing(text):
    text_lemmas = ""
    doc = nlp(text)
    for token in doc:
        if token.is_stop is False and token.pos_ != "PUNCT":
            clean_token = re.sub(r"\W$", r"", token.lemma_)
            clean_token = re.sub(r"^\W", r"", clean_token)
            text_lemmas = text_lemmas+ f" {clean_token}"
    return text_lemmas

# ------------- PROGRAMME PRINCIPAL -----------------------------------

# Chargement du modèle de l'anglais de spaCy, sans le parser et le NER.  
nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])

genres = ["sci-fi", "horror","comedy","action"]

# On ajoute à la liste de stop_words de spaCy nos mots clés.
nlp.Defaults.stop_words |= {"horror","sci-fi","comedy","action","science","fiction"}
# On peut aussi enlever des mots_vides. 
nlp.Defaults.stop_words -= {"former"}

stopwords = '|'.join(map(re.escape, file_to_list("stopwords.txt")))

# Pour chaque classe/genre, on sépare chaque synopsis dans un fichier independant. 
for genre in genres:
    path = f"synopsis/{genre}_synopsis.txt"
    # Lecture du fichier avec les synopsis
    with open (path, encoding="utf-8") as f:
        # Élimination des espaces et des retours à la ligne. 
        text = f.read().strip()
        text = re.sub(r"\n","", text)
        # Compter pour nommer les fichier
        i = 0
        # Extraction du contenu textuel d'une seul synopsis
        for synopsis in re.findall(r"<.+?>([^<]+)", text):
            # Pré-traitement
            new_synopsis = preprocessing(synopsis)
            new_synopsis = re.sub(r"'","",new_synopsis)
            new_synopsis = re.sub(rf" ({stopwords}) ","",new_synopsis)
            # Écriture du fichier
            text_to_file(f"data/{genre}/text{i}.txt", new_synopsis.strip())
            i+=1