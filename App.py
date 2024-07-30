import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression  # Example model
from sklearn.linear_model import LinearRegression # Import a regression model
import pickle
from sklearn.model_selection import train_test_split

data = pd.read_csv('Expresso_churn_dataset.csv')

# Charger le modèle
with open('modele_enregistre (1).sav', 'rb') as fichier:
    model = pickle.load(fichier)

# Titre de l'application
st.title("Application de Prédiction de Churn")

# Charger les données prétraitées
with open('donnees_pre traitees.pkl', 'rb') as fichier:
    data = pickle.load(fichier)

# Sélectionner les colonnes pour la saisie de l'utilisateur
colonnes_saisie = data.drop('CHURN', axis=1).columns

# Créer des champs de saisie pour chaque colonne
saisie_utilisateur = {}
for colonne in colonnes_saisie:
    saisie_utilisateur[colonne] = st.number_input(f"Entrez la valeur pour {colonne}:")

# Créer un bouton pour effectuer la prédiction
if st.button("Prédire"):
    # Convertir les entrées de l'utilisateur en DataFrame
    saisie_df = pd.DataFrame([saisie_utilisateur])

    # Effectuer la prédiction
    prediction = model.predict(saisie_df)

    # Afficher la prédiction
    st.write(f"Prédiction de Churn: {prediction[0]}")