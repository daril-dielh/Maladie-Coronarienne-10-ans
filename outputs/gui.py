import streamlit as st
import joblib as jlb
import numpy as np

# 1. Chargement du modele
from pathlib import Path
BASE_DIR = Path(__file__).parent
model = jlb.load(BASE_DIR / "best_model.pkl")

# 2. Configuration de l'interface utilisateur
st.set_page_config(page_title="Diagnostic Medical AI", page_icon=":heart:", layout="centered")
st.title("Assistant de Diagnostic de Maladie Coronarienne")
st.write("Ajustez les mesures ci-dessous pour obtenir une prediction du modele")

# 3. Creation des curseurs (Sliders) pour les entrees utilisateur
# Note : les noms correspondent a l'ordre exact des 11 variables du modele
st.sidebar.header("Parametres Cardiaques du Patient")

def user_input_features():
    age          = st.sidebar.slider("Age du Patient (ans)",               32,  90,  50)
    male         = st.sidebar.selectbox("Sexe", options=[0, 1], format_func=lambda x: "Femme" if x == 0 else "Homme")
    sysBP        = st.sidebar.slider("Pression Systolique (mmHg)",         80, 295, 130)
    diaBP        = st.sidebar.slider("Pression Diastolique (mmHg)",        40, 150,  85)
    prevalentHyp = st.sidebar.selectbox("Hypertension preexistante", options=[0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
    diabetes     = st.sidebar.selectbox("Diabete",                   options=[0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")

    # Calcul automatique des features engineered (coherent avec le notebook)
    pulse_pressure       = sysBP - diaBP
    age_sysBP            = age * sysBP
    Tension_Hypertension = 1 if sysBP >= 140 else 0
    Tension_Normale      = 1 if (sysBP < 120 and diaBP < 80) else 0
    risk_score           = (
        (1 if age > 55 else 0) * 2 +
        (1 if sysBP >= 140 else 0) * 2 +
        int(diabetes) +
        int(prevalentHyp)
    )

    # Ordre exact des colonnes du modele :
    # age_sysBP, risk_score, age, sysBP, pulse_pressure,
    # prevalentHyp, diaBP, Tension_Hypertension, Tension_Normale, diabetes, male
    data = np.array([[
        age_sysBP, risk_score, age, sysBP, pulse_pressure,
        prevalentHyp, diaBP, Tension_Hypertension, Tension_Normale,
        diabetes, male
    ]])

    return data

input_df = user_input_features()

# 4. Bouton de prediction
if st.button("Lancer le Diagnostic"):
    prediction   = model.predict(input_df)
    probability  = model.predict_proba(input_df)  # probabilite de la prediction

    st.subheader("Resultat du modele")

    if prediction[0] == 1:
        st.write("Le patient est diagnostique comme etant **a risque de maladie coronarienne dans 10 ans**.")
    else:
        st.write("Le patient est diagnostique comme **non a risque de maladie coronarienne dans 10 ans**.")

    # Affichage de la confiance du modele
    st.write(f"Confiance de la prediction: **{np.max(probability) * 100:.2f}%**")

# 5. Section d'information
st.markdown("---")
st.info("Note : Ce modele est un outil d'aide a la decision base sur la Regression Logistique et ne remplace pas l'avis d'un professionnel de sante.")
