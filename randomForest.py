import random
import pandas as pd

class Noeud:
    def __init__(self, question=None, gauche=None, droite=None, resultat=None):
        self.question = question  # La question à poser (ex : "Revenu > 3000 ?")
        self.gauche = gauche      # Branche de gauche (si la réponse est "Non")
        self.droite = droite      # Branche de droite (si la réponse est "Oui")
        self.resultat = resultat  # Résultat final (ex : "Prêt accordé" ou "Prêt refusé")

# Fonction pour construire un arbre de décision
# Les données sont une liste de dictionnaires représentant chaque demande de prêt.
def construire_arbre(donnees, caracteristiques):
    # Si toutes les données ont le même résultat, on crée une feuille terminale.
    if all(d['decision'] == donnees[0]['decision'] for d in donnees):
        return Noeud(resultat=donnees[0]['decision'])

    # Sinon, on choisit la meilleure question à poser pour diviser les données.
    question, donnees_vraies, donnees_fausses = trouver_meilleure_separation(donnees, caracteristiques)

    # Si on ne peut pas diviser davantage, on crée une feuille terminale avec le résultat majoritaire.
    if not donnees_vraies or not donnees_fausses:
        return Noeud(resultat=decision_majoritaire(donnees))

    # On construit récursivement l'arbre pour les branches "Oui" et "Non".
    branche_gauche = construire_arbre(donnees_fausses, caracteristiques)
    branche_droite = construire_arbre(donnees_vraies, caracteristiques)

    return Noeud(question=question, gauche=branche_gauche, droite=branche_droite)

# Fonction pour choisir la meilleure question à poser (on utilise une métrique comme le Gini pour choisir).
def trouver_meilleure_separation(donnees, caracteristiques):
    meilleur_gini = float('inf')  # Plus le Gini est bas, meilleure est la séparation.
    meilleure_question = None
    meilleures_donnees_vraies = None
    meilleures_donnees_fausses = None

    # Tester toutes les questions possibles (sur un sous-ensemble aléatoire des colonnes).
    for colonne in caracteristiques:
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

# Forêt aléatoire : Construire plusieurs arbres de décision avec des sous-ensembles aléatoires des données et des caractéristiques.
def construire_foret_aleatoire(donnees, n_arbres, n_caracteristiques):
    foret = []
    for _ in range(n_arbres):
        # Échantillonner les données avec remplacement (bootstrap)
        echantillon = [random.choice(donnees) for _ in range(len(donnees))]
        # Sélectionner un sous-ensemble aléatoire des caractéristiques
        caracteristiques = random.sample(['revenu', 'montant_pret', 'duree_emploi', 'historique_credit'], n_caracteristiques)
        # Construire un arbre avec cet échantillon et ces caractéristiques
        arbre = construire_arbre(echantillon, caracteristiques)
        foret.append(arbre)
    return foret

# Prédire avec une forêt aléatoire (majorité des votes des arbres).
def predire_foret(foret, echantillon):
    votes = []
    for arbre in foret:
        votes.append(predire_arbre(arbre, echantillon))
    return max(set(votes), key=votes.count)

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

# Fonction pour charger les données depuis un fichier CSV.
def charger_donnees_depuis_csv(chemin_fichier):
    donnees = pd.read_csv(chemin_fichier)
    donnees.columns = ['ID', 'revenu', 'montant_pret', 'duree_emploi', 'historique_credit', 'decision']
    donnees['decision'] = donnees['decision'].apply(lambda x: 'accorde' if x == 'Oui' else 'refuse')
    return donnees.to_dict(orient='records')

# Chemin du fichier CSV.
chemin_fichier_csv = '/c:/Users/bamor/Desktop/SAE Algo 3/(explosion du CPU).csv'

# Charger les données à partir du fichier CSV.
donnees = charger_donnees_depuis_csv(chemin_fichier_csv)

# Construire la forêt aléatoire.
foret_aleatoire = construire_foret_aleatoire(donnees, n_arbres=5, n_caracteristiques=2)

# Exemple de prédiction.
echantillon = {'revenu': 2800, 'montant_pret': 15000, 'duree_emploi': 3, 'historique_credit': 'Bon'}
resultat = predire_foret(foret_aleatoire, echantillon)
print(f"Prédiction pour l'exemple : {resultat}")
