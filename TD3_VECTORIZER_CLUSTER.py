#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 11:16:20 2022

@author: antonomaz
"""

# on import les outils dont on à besoin
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import DistanceMetric
from sklearn.feature_extraction.text import CountVectorizer
import sklearn
import json
import glob
import re
from collections import OrderedDict
import matplotlib.pyplot as plt


# on crée des fonctions, ce sont des outils qu'on réalise pour nous faciliter la tâche et raccourcir la page du programme
def nomfichier(chemin):  # ici on définit le nom de notre fichier pour éviter d'avoir un long chemin comme nom d'usage. Le chemin est une chaîne de caractères qui reprèsentent les dossiers qu'on ouvre pour accéder au fichier.
    nomfich = chemin.split("/")[-1]
    nomfich = nomfich.split(".")
    nomfich = ("_").join([nomfich[0], nomfich[1]])
    return nomfich


# une fonction qui permet d'ouvrir un fichier en indiquant le chemin où se trouve ce fichier.
def lire_json(chemin_fichier_json):
    with open(chemin_fichier_json, "r", encoding="utf-8") as r:
        fich_json = json.load(r)
    return fich_json


# ecrire un fichier json + mettre extension dans le nom sinn enregistre comme txt
def write_json(liste_ent, nom_json):
    with open(nom_json, "w", encoding="utf-8") as w:
        w.write(json.dumps(liste_ent, indent=2))

# MAIN (le corps de notre programmation)


chemin_entree = "CARRAUD_petite-Jeanne_TesseractFra-PNG.txt_SEM_WiNER.ann_SEM.json-concat.json"


# for subcorpus in glob.glob(path_copora):
#    print("SUBCORPUS***",subcorpus)
#liste_nom_fichier =[]
for path in glob.glob(chemin_entree):
    #        print("PATH*****",path)
    liste_nom_fichier = []
    # nom_fichier = nomfichier(path)#on utilise les fonctions crée précedemment
#        print(nom_fichier)
    # on utilise les fonctions crée précedemment. c'est une chaine de caractères
    liste = lire_json(path)


#### FREQUENCE ########

    dic_mots = {}  # un dictionnaire vide, un dico est composé d'une clé et valeur {cle:valeur, cle:valeur...}
    i = 0  # on compte les fréquences de chaque mot à partir du début indice 0

    for mot in liste:  # pour tout les mot qui sont dans la liste

        if mot not in dic_mots:  # si le mot ne se trouve pas dans le dictionnaire crée précedemment
            dic_mots[mot] = 1  # on ajoute le mot en cle et 1 en valeur
        else:  # si le mot est déjà dans le dico on ajoute 1 à sa valeur
            dic_mots[mot] += 1

    i += 1  # on compte la fréquence de chaque mot en avancant pas à pas

    # on ordonne le dictionnaire en fonction des cle, le dictionnaire sera donc rangé par ordre alphabétique
    new_d = OrderedDict(sorted(dic_mots.items(), key=lambda t: t[0]))

    # la variable freq contient la longueur des cle (nbr de caractères)
    freq = len(dic_mots.keys())

    # set permet de retirer les elements répétés dans une liste, chaque element est unique dans la liste. un set contient des elt entre {}
    Set_00 = set(liste)
    # on transforme le set en format liste, les element sont plus maniable que les set. les element d'une liste sont entre []
    Liste_00 = list(Set_00)
    dic_output = {}  # un dico vide
    liste_words = []  # une liste vide
    matrice = []

    for l in Liste_00:  # pour les element dans Liste_00
        # si la longueur de l'element est differente de 1 e que l n'est pas vide
        if len(l) != 1 and l != "":
            # on ajoute cette element dans la liste vide crée précedemment
            liste_words.append(l)
    for li in liste_words:
        if li == "":
            print("espace")

    try:  # on essaie de faire ce qui est indenté...
        # ... on tente de faire un tablo avec les tokens de la liste
        words = np.asarray(liste_words)
        for w in words:  # pour les tokens dans le tableau crée précedemment
            liste_vecteur = []  # une liste vide

            for w2 in words:  # pour chaque mot dans le tableau

                V = CountVectorizer(ngram_range=(2, 3), analyzer='char')
                X = V.fit_transform([w, w2]).toarray()
                distance_tab1 = sklearn.metrics.pairwise.cosine_distances(X)
                # on cacule la distance cosinus entre entre eux et on ajoute ce résultat dans la liste vide liste_vecteur
                liste_vecteur.append(distance_tab1[0][1])

            # on met la liste_vecteur dans la liste matrice--> [[]]
            matrice.append(liste_vecteur)
        # les resultat de distance cosinus sont représenté sous forme de tableau
        matrice_def = -1*np.array(matrice)

        affprop = AffinityPropagation(
            affinity="precomputed", damping=0.6, random_state=None)

        affprop.fit(matrice_def)
        for cluster_id in np.unique(affprop.labels_):
            exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
            cluster = np.unique(
                words[np.nonzero(affprop.labels_ == cluster_id)])
            cluster_str = ", ".join(cluster)
            cluster_list = cluster_str.split(", ")

            Id = "ID "+str(i)
            for cle, dic in new_d.items():
                if cle == exemplar:
                    dic_output[Id] = {}
                    dic_output[Id]["Centroïde"] = exemplar
                    dic_output[Id]["Freq. centroide"] = dic
                    dic_output[Id]["Termes"] = cluster_list

            i = i+1
           # print(dic_output)

    except:
        #print("**********Non OK***********", path)

        liste_nom_fichier.append(path)

    #     continue

# enregistrer en json
nomjson_clust = "cluster_carraud.json"
# write_json(dic_output,nomjson_clust)


# representation graphique
# essai 1 avec l'aide du programme dispo sur scikit learn
# cluster_center_ind= affprop.cluster_centers_indices_
# len_clusters= len(cluster_center_ind)
# #print(len_clusters)
# lab= affprop.labels_

# plt.close("all")
# plt.figure(1)
# plt.clf()

# forme_tablo=(6,-1)
# X_forme= X.reshape(forme_tablo)
# colors= plt.cycler("colors",plt.cm.viridis(np.linspace(0,19)))

# for k,col in zip(range(len_clusters),colors):
#     if cluster_center_ind[k]< len(X):
#         class_members= lab==k
#         cluster_center=X[cluster_center_ind[k]]
#         plt.scatter( X[class_members, 0], X[class_members, 1], color=col["color"], marker=".")
#         plt.scatter(cluster_center[0], cluster_center[1], s=14, color=col["color"], marker="o")
#         for x in X[class_members]:
#             plt.plot( [cluster_center[0], x[0]], [cluster_center[1], x[1]], color=col["color"])
#    # else:
#     #    print(f"invalid cluster center indice{cluster_center_ind[k]}")

# plt.figure(figsize=(6,4))
# #plt.show()
# FAIL

# essai2 but: extraire la liste de termes de chaque cluster puis faire un len pour chaque mini liste de terme
def retirefreq_centro(dico_imbrique):
    liste_tout=[]
    for cle, valeur in dico_imbrique.items():
        for cle2, valeur2 in valeur.items():
            if type(valeur2) != int:
                liste_tout.append(valeur2)
    return liste_tout #liste des token centroide et des terme similaire avec le centroide

def liste_termesimili(liste_tout):
    liste_termeclust=[]
    for elem in liste_tout:
        if type(elem)==list:
            liste_termeclust.append(elem)
    return liste_termeclust#[[liste terme],[liste terme],[liste terme]]

def long_liste_terme(liste_termeclust):
    donnee_graphique=[]
    for little_liste in liste_termeclust:
        donnee_graphique.append(len(little_liste))
    return donnee_graphique#[int,int,int]

liste_centro_terme= retirefreq_centro(dic_output)
liste_des_termes=liste_termesimili(liste_centro_terme)
donnee_pr_graph= long_liste_terme(liste_des_termes)


# # representation graphique 2
x_val = np.arange(len(donnee_pr_graph))
plt.scatter(x_val, donnee_pr_graph, color="pink")
plt.ylabel("nbr de termes dans chaque cluster")
plt.title("taille des clusters")
#plt.savefig("graphique len cluster F")
#plt.show()
