# Projet de Classification : CART et RandomForest

Ce projet implémente deux algorithmes de classification : CART (Classification and Regression Trees) et RandomForest. Une interface graphique est également fournie pour charger des données, entraîner les modèles et effectuer des prédictions.

## Prérequis

- Python 3.x
- Bibliothèques : `tkinter`, `pandas`, `numpy`

## Installation

1. Clonez le dépôt ou téléchargez les fichiers.
2. Assurez-vous d'avoir les bibliothèques nécessaires installées :
    ```bash
    pip install pandas numpy
    ```

## Utilisation

1. Exécutez le script `pret_banque.py` :
    ```bash
    python pret_banque.py
    ```
2. Utilisez l'interface graphique pour :
    - Charger un fichier CSV contenant les données.
    - Entraîner un modèle CART ou RandomForest.
    - Afficher le taux de précision du modèle entraîné.
    - Effectuer des prédictions basées sur des entrées utilisateur.

## Structure des Données

Le fichier CSV doit contenir les colonnes suivantes :
- `Revenu (€)`
- `Montant du Prêt (€)`
- `Durée de l'Emploi (années)`
- `Historique de Crédit` (valeurs possibles : "Bon", "Mauvais")
- `Prêt Approuvé` (valeurs possibles : "Oui", "Non")

## Exemple de Données

```csv
Revenu (€),Montant du Prêt (€),Durée de l'Emploi (années),Historique de Crédit,Prêt Approuvé
50000,20000,10,Bon,Oui
30000,10000,5,Mauvais,Non
...
```

## Fonctionnalités

- **Charger un fichier CSV** : Permet de charger les données depuis un fichier CSV.
- **Entraîner CART** : Entraîne un modèle CART avec une profondeur maximale de 5.
- **Entraîner RandomForest** : Entraîne un modèle de RandomForest avec 10 arbres et une profondeur maximale de 5.
- **Prédire** : Effectue une prédiction basée sur les entrées utilisateur.

## Auteur

Ce projet a été réalisé par Bastien MORLION.
