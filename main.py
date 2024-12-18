import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np

# Implémentation de l'algorithme CART (simplifié)
class CART:
    def __init__(self, profondeur_max=None):
        self.profondeur_max = profondeur_max
        self.arbre = None

    def fit(self, X, y):
        données = np.column_stack((X, y))
        self.arbre = self._construire_arbre(données, profondeur=0)

    def predict(self, X):
        return [self._prédire_ligne(self.arbre, ligne) for ligne in X]

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

    def _test_split(self, index, valeur, dataset):
        # Division du dataset
        gauche, droite = list(), list()
        for ligne in dataset:
            if ligne[index] < valeur:
                gauche.append(ligne)
            else:
                droite.append(ligne)
        return gauche, droite

    def _get_split(self, dataset):
        # Trouver la meilleure division
        valeurs_classes = list(set(ligne[-1] for ligne in dataset))
        b_index, b_valeur, b_score, b_groupes = 999, 999, 999, None
        for index in range(len(dataset[0]) - 1):
            for ligne in dataset:
                groupes = self._test_split(index, ligne[index], dataset)
                gini = self._gini(groupes, valeurs_classes)
                if gini < b_score:
                    b_index, b_valeur, b_score, b_groupes = index, ligne[index], gini, groupes
        return {"index": b_index, "valeur": b_valeur, "groupes": b_groupes}

    def _to_terminal(self, groupe):
        # Créer un noeud terminal
        résultats = [ligne[-1] for ligne in groupe]
        return max(set(résultats), key=résultats.count)

    def _split(self, noeud, profondeur):
        gauche, droite = noeud["groupes"]
        del(noeud["groupes"])
        if not gauche or not droite:
            noeud["gauche"] = noeud["droite"] = self._to_terminal(gauche + droite)
            return
        if self.profondeur_max and profondeur >= self.profondeur_max:
            noeud["gauche"], noeud["droite"] = self._to_terminal(gauche), self._to_terminal(droite)
            return
        noeud["gauche"] = self._get_split(gauche)
        self._split(noeud["gauche"], profondeur + 1)
        noeud["droite"] = self._get_split(droite)
        self._split(noeud["droite"], profondeur + 1)

    def _construire_arbre(self, train, profondeur):
        racine = self._get_split(train)
        self._split(racine, profondeur)
        return racine

    def _prédire_ligne(self, noeud, ligne):
        if ligne[noeud["index"]] < noeud["valeur"]:
            if isinstance(noeud["gauche"], dict):
                return self._prédire_ligne(noeud["gauche"], ligne)
            else:
                return noeud["gauche"]
        else:
            if isinstance(noeud["droite"], dict):
                return self._prédire_ligne(noeud["droite"], ligne)
            else:
                return noeud["droite"]

class ForêtAléatoire:
    def __init__(self, n_arbres=10, profondeur_max=None, taille_échantillon=None):
        self.n_arbres = n_arbres
        self.profondeur_max = profondeur_max
        self.taille_échantillon = taille_échantillon
        self.arbres = []

    def fit(self, X, y):
        self.arbres = []
        for _ in range(self.n_arbres):
            échantillon_X, échantillon_y = self._sous_échantillon(X, y)
            arbre = CART(profondeur_max=self.profondeur_max)
            arbre.fit(échantillon_X, échantillon_y)
            self.arbres.append(arbre)

    def predict(self, X):
        prédictions = [arbre.predict(X) for arbre in self.arbres]
        prédictions = np.array(prédictions)
        return [self._vote_majoritaire(préd) for préd in prédictions.T]

    def _sous_échantillon(self, X, y):
        n_échantillons = len(X)
        taille_échantillon = self.taille_échantillon or n_échantillons
        indices = np.random.choice(n_échantillons, taille_échantillon, replace=True)
        return X[indices], y[indices]

    def _vote_majoritaire(self, prédictions):
        return max(set(prédictions), key=list(prédictions).count)

# Initialisation de l'interface
def lancer_interface():
    racine = tk.Tk()
    racine.title("Classification: CART et Forêt Aléatoire")
    
    def charger_fichier():
        chemin_fichier = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if chemin_fichier:
            try:
                global données, modèle
                données = pd.read_csv(chemin_fichier)
                messagebox.showinfo("Succès", "Fichier chargé avec succès !")
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible de charger le fichier : {str(e)}")

    def entraîner_modèle(algorithme):
        if données is None:
            messagebox.showerror("Erreur", "Veuillez charger un fichier CSV d'abord.")
            return

        try:
            # Préparer les données
            X = données[["Revenu (€)", "Montant du Prêt (€)", "Durée de l'Emploi (années)"]].values
            X = np.column_stack((X, données["Historique de Crédit"].map({"Bon": 1, "Mauvais": 0}).values))
            y = données["Prêt Approuvé"].map({"Oui": 1, "Non": 0}).values
            
            if algorithme == "CART":
                global modèle_cart
                modèle_cart = CART(profondeur_max=5)
                modèle_cart.fit(X, y)
                précision = calculer_précision(modèle_cart, X, y)
                lbl_précision.config(text=f"Taux de précision CART: {précision:.2f}")
                messagebox.showinfo("Succès", f"Modèle {algorithme} entraîné avec succès ! Taux de précision: {précision:.2f}")
            elif algorithme == "Forêt Aléatoire":
                global modèle_fa
                modèle_fa = ForêtAléatoire(n_arbres=10, profondeur_max=5)
                modèle_fa.fit(X, y)
                précision = calculer_précision(modèle_fa, X, y)
                lbl_précision.config(text=f"Taux de précision Forêt Aléatoire: {précision:.2f}")
                messagebox.showinfo("Succès", f"Modèle {algorithme} entraîné avec succès ! Taux de précision: {précision:.2f}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Problème lors de l'entraînement : {str(e)}")

    def calculer_précision(modèle, X, y):
        prédictions = modèle.predict(X)
        correct = sum(préd == réel for préd, réel in zip(prédictions, y))
        return correct / len(y)

    def prédire_classe():
        try:
            if modèle_cart is None and modèle_fa is None:
                messagebox.showerror("Erreur", "Veuillez entraîner un modèle d'abord.")
                return
            
            revenu = float(entry_revenu.get())
            montant = float(entry_montant.get())
            durée = int(entry_durée.get())
            historique = 1 if var_historique.get() == "Bon" else 0

            données_entrée = [[revenu, montant, durée, historique]]
            if modèle_cart:
                prédiction = modèle_cart.predict(données_entrée)[0]
            elif modèle_fa:
                prédiction = modèle_fa.predict(données_entrée)[0]
            
            résultat = "Oui" if prédiction == 1 else "Non"
            messagebox.showinfo("Résultat", f"Prédiction : {résultat}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Problème lors de la prédiction : {str(e)}")

    # Widgets de l'interface
    btn_charger = tk.Button(racine, text="Charger un fichier CSV", command=charger_fichier)
    btn_charger.pack(pady=10)

    btn_cart = tk.Button(racine, text="Entraîner CART", command=lambda: entraîner_modèle("CART"))
    btn_cart.pack(pady=5)

    btn_fa = tk.Button(racine, text="Entraîner Forêt Aléatoire", command=lambda: entraîner_modèle("Forêt Aléatoire"))
    btn_fa.pack(pady=5)

    lbl_précision = tk.Label(racine, text="Taux de précision: N/A")
    lbl_précision.pack(pady=5)

    cadre_entrée = tk.LabelFrame(racine, text="Exemple de prédiction")
    cadre_entrée.pack(pady=10, padx=10)

    tk.Label(cadre_entrée, text="Revenu (€):").grid(row=0, column=0)
    entry_revenu = tk.Entry(cadre_entrée)
    entry_revenu.grid(row=0, column=1)

    tk.Label(cadre_entrée, text="Montant du Prêt (€):").grid(row=1, column=0)
    entry_montant = tk.Entry(cadre_entrée)
    entry_montant.grid(row=1, column=1)

    tk.Label(cadre_entrée, text="Durée de l'Emploi (années):").grid(row=2, column=0)
    entry_durée = tk.Entry(cadre_entrée)
    entry_durée.grid(row=2, column=1)

    tk.Label(cadre_entrée, text="Historique de Crédit:").grid(row=3, column=0)
    var_historique = tk.StringVar(value="Bon")
    tk.OptionMenu(cadre_entrée, var_historique, "Bon", "Mauvais").grid(row=3, column=1)

    btn_prédire = tk.Button(racine, text="Prédire", command=prédire_classe)
    btn_prédire.pack(pady=10)

    racine.mainloop()

if __name__ == "__main__":
    données = None
    modèle_cart = None
    modèle_fa = None
    lancer_interface()

# Exécutez le script main.py pour lancer l'interface graphique. Vous pouvez charger un fichier CSV, entraîner un modèle CART ou Forêt Aléatoire, et effectuer des prédictions sur de nouvelles données.
# Notez que l'implémentation de Forêt Aléatoire n'est pas encore complète.
