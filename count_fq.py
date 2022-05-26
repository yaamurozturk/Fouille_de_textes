#!/usr/bin/python

# Utilisation : python3 count_f1.py
import os

list = ["action", "comedy", "horror", "sci-fi"]

c = 0
t = 0

# Petit script pour compter les mots et les fichiers textes situés dans un dossier par genre, dans le dossier père "data".

for genre in list:
    for dirpath, dirnames, files in os.walk(f"./data/{genre}"):
        for file in files:
            t += 1
            path = f"./data/{genre}/{file}"
            with open(path, encoding="utf-8") as f:
                texte = f.read()
                words = texte.split()
                c = c + len(words)

print(f"Nombre total de synopsis : {t}")
print(f"Nombre total de mots : {c}")
print(f"Moyenne de mots par synopsis : {c/t}")
