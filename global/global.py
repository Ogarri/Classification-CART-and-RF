import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np

# Implementation de l'algorithme CART (simplifie)
class CART:
    def __init__(self, profondeur_max=None):
        self.profondeur_max = profondeur_max
        self.arbre = None

    def ajuster(self, X, y):
        donnees = np.column_stack((X, y))
        self.arbre = self._construire_arbre(donnees, profondeur=0)

    def predire(self, X):
        return [self._predire_ligne(self.arbre, ligne) for ligne in X]

    def _gini(self, groupes, classes):
        # Calcul de l'indice de Gini
        n_instances = float(sum([len(groupe) for groupe in groupes]))
        gini = 0.0
        for groupe in groupes:
            taille = float(len(groupe))
            if taille == 0:
                continue
            score = 0.0
            for classe_val in classes:
                p = [ligne[-1] for ligne in groupe].count(classe_val) / taille
                score += p * p
            gini += (1.0 - score) * (taille / n_instances)
        return gini

    def _test_division(self, index, valeur, dataset):
        # Division du dataset
        gauche, droite = list(), list()
        for ligne in dataset:
            if ligne[index] < valeur:
                gauche.append(ligne)
            else:
                droite.append(ligne)
        return gauche, droite

    def _obtenir_division(self, dataset):
        # Trouver la meilleure division
        valeurs_classes = list(set(ligne[-1] for ligne in dataset))
        b_index, b_valeur, b_score, b_groupes = 999, 999, 999, None
        for index in range(len(dataset[0]) - 1):
            for ligne in dataset:
                groupes = self._test_division(index, ligne[index], dataset)
                gini = self._gini(groupes, valeurs_classes)
                if gini < b_score:
                    b_index, b_valeur, b_score, b_groupes = index, ligne[index], gini, groupes
        return {"index": b_index, "valeur": b_valeur, "groupes": b_groupes}

    def _à_terminal(self, groupe):
        # Creer un noeud terminal
        resultats = [ligne[-1] for ligne in groupe]
        return max(set(resultats), key=resultats.count)

    def _diviser(self, noeud, profondeur):
        gauche, droite = noeud["groupes"]
        del(noeud["groupes"])
        if not gauche or not droite:
            noeud["gauche"] = noeud["droite"] = self._à_terminal(gauche + droite)
            return
        if self.profondeur_max and profondeur >= self.profondeur_max:
            noeud["gauche"], noeud["droite"] = self._à_terminal(gauche), self._à_terminal(droite)
            return
        noeud["gauche"] = self._obtenir_division(gauche)
        self._diviser(noeud["gauche"], profondeur + 1)
        noeud["droite"] = self._obtenir_division(droite)
        self._diviser(noeud["droite"], profondeur + 1)

    def _construire_arbre(self, train, profondeur):
        racine = self._obtenir_division(train)
        self._diviser(racine, profondeur)
        return racine

    def _predire_ligne(self, noeud, ligne):
        if ligne[noeud["index"]] < noeud["valeur"]:
            if isinstance(noeud["gauche"], dict):
                return self._predire_ligne(noeud["gauche"], ligne)
            else:
                return noeud["gauche"]
        else:
            if isinstance(noeud["droite"], dict):
                return self._predire_ligne(noeud["droite"], ligne)
            else:
                return noeud["droite"]

class RandomForest:
    def __init__(self, n_arbres=10, profondeur_max=None, taille_echantillon=None):
        self.n_arbres = n_arbres
        self.profondeur_max = profondeur_max
        self.taille_echantillon = taille_echantillon
        self.arbres = []

    def ajuster(self, X, y):
        self.arbres = []
        for _ in range(self.n_arbres):
            echantillon_X, echantillon_y = self._sous_echantillon(X, y)
            arbre = CART(profondeur_max=self.profondeur_max)
            arbre.ajuster(echantillon_X, echantillon_y)
            self.arbres.append(arbre)

    def predire(self, X):
        predictions = [arbre.predire(X) for arbre in self.arbres]
        predictions = np.array(predictions)
        return [self._vote_majoritaire(pred) for pred in predictions.T]

    def _sous_echantillon(self, X, y):
        n_echantillons = len(X)
        taille_echantillon = self.taille_echantillon or n_echantillons
        indices = np.random.choice(n_echantillons, taille_echantillon, replace=True)
        return X[indices], y[indices]

    def _vote_majoritaire(self, predictions):
        return max(set(predictions), key=list(predictions).count)

    def compter_branches(self):
        return [self._compter_branches_arbre(arbre.arbre) for arbre in self.arbres]

    def _compter_branches_arbre(self, noeud):
        if isinstance(noeud, dict):
            return 1 + self._compter_branches_arbre(noeud["gauche"]) + self._compter_branches_arbre(noeud["droite"])
        else:
            return 0

# Initialisation de l'interface
def lancer_interface():
    racine = tk.Tk()
    racine.title("Classification: CART et Random Forest")
    
    def charger_fichier():
        chemin_fichier = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if chemin_fichier:
            try:
                global donnees, colonnes, cible, entries
                donnees = pd.read_csv(chemin_fichier)
                colonnes = list(donnees.columns)
                if len(colonnes) < 2:
                    raise ValueError("Le fichier CSV doit contenir au moins deux colonnes.")
                cible = colonnes[-1]  # Dernière colonne comme cible par défaut
                messagebox.showinfo("Succes", "Fichier charge avec succes !")
                
                # Mise à jour des entrées pour la prédiction
                for widget in cadre_entree.winfo_children():
                    widget.destroy()
                entries = []
                for i, col in enumerate(colonnes[:-1]):
                    tk.Label(cadre_entree, text=f"{col}:").grid(row=i, column=0)
                    entry = tk.Entry(cadre_entree)
                    entry.grid(row=i, column=1)
                    entries.append(entry)
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible de charger le fichier : {str(e)}")

    def entraîner_modele(algorithme):
        if donnees is None:
            messagebox.showerror("Erreur", "Veuillez charger un fichier CSV d'abord.")
            return

        try:
            # Preparer les donnees
            X = donnees.drop(columns=[cible]).values
            y = donnees[cible].values
            
            if algorithme == "CART":
                global modele_cart
                modele_cart = CART(profondeur_max=5)
                modele_cart.ajuster(X, y)
                precision = calculer_precision(modele_cart, X, y)
                lbl_precision.config(text=f"Taux de precision CART: {precision:.2f}")
                messagebox.showinfo("Succes", f"Modele {algorithme} entraîne avec succes ! Taux de precision: {precision:.2f}")
            elif algorithme == "Random Forest":
                global modele_fa
                modele_fa = RandomForest(n_arbres=10, profondeur_max=5)
                modele_fa.ajuster(X, y)
                precision = calculer_precision(modele_fa, X, y)
                lbl_precision.config(text=f"Taux de precision Random Forest: {precision:.2f}")
                nombre_branches = modele_fa.compter_branches()
                lbl_branches.config(text=f"Nombre de branches par arbre: {nombre_branches}")
                messagebox.showinfo("Succes", f"Modele {algorithme} entraîne avec succes ! Taux de precision: {precision:.2f}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Probleme lors de l'entraînement : {str(e)}")

    def calculer_precision(modele, X, y):
        predictions = modele.predire(X)
        correct = sum(pred == reel for pred, reel in zip(predictions, y))
        return correct / len(y)

    def predire_classe():
        try:
            if modele_cart is None and modele_fa is None:
                messagebox.showerror("Erreur", "Veuillez entraîner un modele d'abord.")
                return
            
            donnees_entree = [entry.get() for entry in entries]
            if len(donnees_entree) != len(colonnes) - 1:
                raise ValueError("Le nombre de valeurs d'entrée ne correspond pas au nombre de colonnes.")
            donnees_entree = [donnees_entree]
            if modele_cart:
                prediction = modele_cart.predire(donnees_entree)[0]
            elif modele_fa:
                prediction = modele_fa.predire(donnees_entree)[0]
            
            resultat = prediction
            messagebox.showinfo("Resultat", f"Prediction : {resultat}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Probleme lors de la prediction : {str(e)}")

    # Widgets de l'interface
    btn_charger = tk.Button(racine, text="Charger un fichier CSV", command=charger_fichier)
    btn_charger.pack(pady=10)

    btn_cart = tk.Button(racine, text="Entraîner CART", command=lambda: entraîner_modele("CART"))
    btn_cart.pack(pady=5)

    btn_fa = tk.Button(racine, text="Entraîner Random Forest", command=lambda: entraîner_modele("Random Forest"))
    btn_fa.pack(pady=5)

    lbl_precision = tk.Label(racine, text="Taux de precision: N/A")
    lbl_precision.pack(pady=5)

    lbl_branches = tk.Label(racine, text="Nombre de branches par arbre: N/A")
    lbl_branches.pack(pady=5)

    cadre_entree = tk.LabelFrame(racine, text="Exemple de prediction")
    cadre_entree.pack(pady=10, padx=10)

    entries = []

    btn_predire = tk.Button(racine, text="Prédire", command=predire_classe)
    btn_predire.pack(pady=10)

    racine.mainloop()

if __name__ == "__main__":
    donnees = None
    colonnes = []
    cible = None
    modele_cart = None
    modele_fa = None
    lancer_interface()
