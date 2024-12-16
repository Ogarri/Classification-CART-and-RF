import csv
import pandas as pd

class Noeud:
    def __init__(self, question=None, gauche=None, droite=None, resultat=None):
        self.question = question  # La question à poser (ex : "Revenu > 3000 ?")
        self.gauche = gauche      # Branche de gauche (si la réponse est "Non")
        self.droite = droite      # Branche de droite (si la réponse est "Oui")
        self.resultat = resultat  # Résultat final (ex : "Prêt accordé" ou "Prêt refusé")

# Fonction pour construire un arbre de décision
# Les données sont une liste de dictionnaires représentant chaque demande de prêt.
def construire_arbre_cart(donnees):
    # Si toutes les données ont le même résultat, on crée une feuille terminale.
    if all(d['decision'] == donnees[0]['decision'] for d in donnees):
        return Noeud(resultat=donnees[0]['decision'])

    # Sinon, on choisit la meilleure question à poser pour diviser les données.
    question, donnees_vraies, donnees_fausses = trouver_meilleure_separation(donnees)

    # Si on ne peut pas diviser davantage, on crée une feuille terminale avec le résultat majoritaire.
    if not donnees_vraies or not donnees_fausses:
        return Noeud(resultat=decision_majoritaire(donnees))

    # On construit récursivement l'arbre pour les branches "Oui" et "Non".
    branche_gauche = construire_arbre_cart(donnees_fausses)
    branche_droite = construire_arbre_cart(donnees_vraies)

    return Noeud(question=question, gauche=branche_gauche, droite=branche_droite)

# Fonction pour choisir la meilleure question à poser (on utilise une métrique comme le Gini pour choisir).
def trouver_meilleure_separation(donnees):
    meilleur_gini = float('inf')  # Plus le Gini est bas, meilleure est la séparation.
    meilleure_question = None
    meilleures_donnees_vraies = None
    meilleures_donnees_fausses = None

    # Tester toutes les questions possibles (toutes les colonnes et valeurs seuils).
    for colonne in ['revenu', 'montant_pret', 'duree_emploi', 'historique_credit']:
        valeurs = set(d[colonne] for d in donnees)  # Toutes les valeurs possibles pour cette colonne.
        for valeur in valeurs:
            question = (colonne, valeur)  # Par exemple : ("revenu", 3000)
            donnees_vraies, donnees_fausses = separer_donnees(donnees, question)

            # Calculer le score Gini pour cette séparation.
            gini = impurete_gini(donnees_vraies, donnees_fausses)

            if gini < meilleur_gini:
                meilleur_gini = gini
                meilleure_question = question
                meilleures_donnees_vraies = donnees_vraies
                meilleures_donnees_fausses = donnees_fausses

    return meilleure_question, meilleures_donnees_vraies, meilleures_donnees_fausses

# Fonction pour séparer les données en deux groupes selon la question.
def separer_donnees(donnees, question):
    colonne, valeur = question
    if colonne == 'historique_credit':
        donnees_vraies = [d for d in donnees if d[colonne] == valeur]
        donnees_fausses = [d for d in donnees if d[colonne] != valeur]
    else:
        donnees_vraies = [d for d in donnees if d[colonne] > valeur]
        donnees_fausses = [d for d in donnees if d[colonne] <= valeur]
    return donnees_vraies, donnees_fausses

# Fonction pour calculer l'impureté Gini (mesure de la "pureté" des groupes).
def impurete_gini(donnees_vraies, donnees_fausses):
    taille_totale = len(donnees_vraies) + len(donnees_fausses)
    if taille_totale == 0:
        return 0

    # Calculer la proportion de chaque groupe.
    proportion_vraie = len(donnees_vraies) / taille_totale
    proportion_fausse = len(donnees_fausses) / taille_totale

    # Calculer le Gini pour chaque groupe.
    def gini_groupe(donnees):
        if len(donnees) == 0:
            return 0
        proportions = [sum(1 for d in donnees if d['decision'] == decision) / len(donnees) for decision in ['accorde', 'refuse']]
        return 1 - sum(p ** 2 for p in proportions)

    return proportion_vraie * gini_groupe(donnees_vraies) + proportion_fausse * gini_groupe(donnees_fausses)

# Fonction pour trouver le résultat majoritaire (en cas de feuille).
def decision_majoritaire(donnees):
    decisions = [d['decision'] for d in donnees]
    return max(set(decisions), key=decisions.count)

# Fonction pour afficher l'arbre de manière lisible.
def afficher_arbre(noeud, profondeur=0):
    if noeud.resultat is not None:
        print("  " * profondeur + f"Résultat : {noeud.resultat}")
    else:
        colonne, valeur = noeud.question
        print("  " * profondeur + f"{colonne} > {valeur} ?")
        afficher_arbre(noeud.droite, profondeur + 1)
        print("  " * profondeur + "Sinon :")
        afficher_arbre(noeud.gauche, profondeur + 1)

def charger_donnees_depuis_csv(chemin_fichier):
    donnees = pd.read_csv(chemin_fichier)
    donnees['revenu'] = donnees['revenu'].astype(int)
    donnees['montant_pret'] = donnees['montant_pret'].astype(int)
    donnees['duree_emploi'] = donnees['duree_emploi'].astype(int)
    return donnees.to_dict(orient='records')

# Prédire avec un seul arbre.
def predire_arbre(noeud, echantillon):
    if noeud.resultat is not None:
        return noeud.resultat
    colonne, valeur = noeud.question
    if colonne == 'historique_credit':
        if echantillon[colonne] == valeur:
            return predire_arbre(noeud.droite, echantillon)
        else:
            return predire_arbre(noeud.gauche, echantillon)
    else:
        if echantillon[colonne] > valeur:
            return predire_arbre(noeud.droite, echantillon)
        else:
            return predire_arbre(noeud.gauche, echantillon)

# Exemple de chemin de fichier CSV.
chemin_fichier_csv = '/c:/Users/bamor/Desktop/SAE Algo 3/data.csv'

# Charger les données à partir du fichier CSV.
donnees = charger_donnees_depuis_csv(chemin_fichier_csv)

# Construire l'arbre.
arbre_cart = construire_arbre_cart(donnees)

# Afficher l'arbre.
afficher_arbre(arbre_cart)
