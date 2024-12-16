import tkinter as tk
from tkinter import ttk, filedialog
from randomForest import construire_foret_aleatoire, predire_foret, charger_donnees_depuis_csv
from CART import construire_arbre_cart, afficher_arbre, charger_donnees_depuis_csv, predire_arbre

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Prédiction de Prêt")
        self.geometry("600x400")

        self.create_widgets()

    def create_widgets(self):
        # Labels et entrées pour les caractéristiques
        tk.Label(self, text="Revenu (€)").grid(row=0, column=0, padx=10, pady=10)
        self.revenu_entry = tk.Entry(self)
        self.revenu_entry.grid(row=0, column=1, padx=10, pady=10)

        tk.Label(self, text="Montant du prêt (€)").grid(row=1, column=0, padx=10, pady=10)
        self.montant_pret_entry = tk.Entry(self)
        self.montant_pret_entry.grid(row=1, column=1, padx=10, pady=10)

        tk.Label(self, text="Durée de l'emploi (années)").grid(row=2, column=0, padx=10, pady=10)
        self.duree_emploi_entry = tk.Entry(self)
        self.duree_emploi_entry.grid(row=2, column=1, padx=10, pady=10)

        tk.Label(self, text="Historique de crédit").grid(row=3, column=0, padx=10, pady=10)
        self.historique_credit_combobox = ttk.Combobox(self, values=["Bon", "Mauvais"])
        self.historique_credit_combobox.grid(row=3, column=1, padx=10, pady=10)

        # Boutons pour prédire avec CART et Random Forest
        tk.Button(self, text="Prédire avec CART", command=self.predire_cart).grid(row=4, column=0, padx=10, pady=10)
        tk.Button(self, text="Prédire avec Random Forest", command=self.predire_random_forest).grid(row=4, column=1, padx=10, pady=10)

        # Bouton pour charger le fichier CSV
        tk.Button(self, text="Charger fichier CSV", command=self.charger_fichier).grid(row=4, column=2, padx=10, pady=10)

        # Zone de texte pour afficher les résultats
        self.result_text = tk.Text(self, height=10, width=50)
        self.result_text.grid(row=5, column=0, columnspan=3, padx=10, pady=10)

    def predire_cart(self):
        echantillon = self.get_echantillon()
        chemin_fichier_csv = '/c:/Users/bamor/Desktop/SAE Algo 3/(explosion du CPU).csv'
        donnees = charger_donnees_depuis_csv(chemin_fichier_csv)
        arbre_cart = construire_arbre_cart(donnees)
        resultat = predire_arbre(arbre_cart, echantillon)
        self.result_text.insert(tk.END, f"Prédiction avec CART : {resultat}\n")

    def predire_random_forest(self):
        echantillon = self.get_echantillon()
        chemin_fichier_csv = '/c:/Users/bamor/Desktop/SAE Algo 3/(explosion du CPU).csv'
        donnees = charger_donnees_depuis_csv(chemin_fichier_csv)
        foret_aleatoire = construire_foret_aleatoire(donnees, n_arbres=5, n_caracteristiques=2)
        resultat = predire_foret(foret_aleatoire, echantillon)
        self.result_text.insert(tk.END, f"Prédiction avec Random Forest : {resultat}\n")

    def get_echantillon(self):
        return {
            'revenu': int(self.revenu_entry.get()),
            'montant_pret': int(self.montant_pret_entry.get()),
            'duree_emploi': int(self.duree_emploi_entry.get()),
            'historique_credit': self.historique_credit_combobox.get()
        }

    def charger_fichier(self):
        chemin_fichier = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if chemin_fichier:
            donnees = charger_donnees_depuis_csv(chemin_fichier)
            arbre_cart = construire_arbre_cart(donnees)
            afficher_arbre(arbre_cart)
            echantillon = self.get_echantillon()
            resultat = predire_arbre(arbre_cart, echantillon)
            self.result_text.insert(tk.END, f"Prédiction pour l'exemple : {resultat}\n")