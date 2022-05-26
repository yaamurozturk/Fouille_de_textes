#!/usr/bin/python

from bs4 import BeautifulSoup
import re
import requests

#---------------------------------------------
# Exemple pour extraire des synopsis de Rotten Tomatoes. 
# Utilisation : python3 crawling_tomatoes.py
# Il faut changer les liens un par un. Le script a été fait quand nous n'avions pas encore tous les lien.
#---------------------------------------------

# Fonction pour extraire un site web à partir de la bibliothèque request.
def get_textURL(url):
    URL = requests.get(url)
    return URL.text

output = open("horror_synopsis.txt", "w", encoding="utf-8")
# Récuperation du site web principal
html_text = get_textURL("https://www.rottentomatoes.com/top/bestofrt/top_100_horror_movies/")
# Avec BiutifulSoup, on crée un objet parseable
soup = BeautifulSoup(html_text, 'html.parser')
# Pour chaque ligne du fichier html qui contient la balise <td> (le site est structuré comme un tableaux)

films_set = set()

# Récuperation des URLS de chaque film
for tag in soup.find_all('a', class_='unstyled articleLink'):
    film = re.search(r"href=\"(\/m\/.+)\"", str(tag))
    if film is not None:
        films_set.add(film.group(1))

i = 0

# Extraction du synopsis du site de chaque film et écriture dans un nouveau fichier.
for film in films_set:
    html_text = get_textURL("https://www.rottentomatoes.com{}".format(film))
    soup = BeautifulSoup(html_text, 'html.parser')
    synopsis = soup.find(id='movieSynopsis')
    if synopsis is not None:
        synopsis = synopsis.string
        synopsis = re.sub(r"^\s+","", synopsis)
        synopsis = re.sub(r".\s+$",".\n", synopsis)
        output.write("<synopsis class=\"horror\" no=\"{}\">\n{}".format(i,synopsis))
        print("Extracting synopsis no. {}".format(i))
        i += 1

output.close()