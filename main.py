from CART import charger_donnees_depuis_csv, construire_arbre_cart, afficher_arbre, predire_arbre
import tkinter as tk
from tkinter import filedialog

def charger_fichier():
    chemin_fichier = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if chemin_fichier:
        # Charger les données à partir du fichier CSV.
        donnees = charger_donnees_depuis_csv(chemin_fichier)
        
        # Construire l'arbre.
        arbre_cart = construire_arbre_cart(donnees)
        
        # Afficher l'arbre.
        afficher_arbre(arbre_cart)
        
        # Exemple de prédiction.
        echantillon = {'revenu': 2800, 'montant_pret': 15000, 'duree_emploi': 3, 'historique_credit': 'Bon'}
        resultat = predire_arbre(arbre_cart, echantillon)
        print(f"Prédiction pour l'exemple : {resultat}")

def main():
    root = tk.Tk()
    root.title("Chargement de fichier CSV")
    root.geometry("300x100")

    bouton_charger = tk.Button(root, text="Charger fichier CSV", command=charger_fichier)
    bouton_charger.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()
