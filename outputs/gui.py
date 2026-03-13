import streamlit as st
import joblib as jlb
import numpy as np
from pathlib import Path

# 1. Chargement du modele
BASE_DIR = Path(__file__).parent
model = jlb.load(BASE_DIR / "best_model.pkl")

# 2. Configuration de la page
st.set_page_config(
    page_title="Diagnostic CHD - IA Medicale",
    page_icon=":heart:",
    layout="centered"
)

# 3. En-tete
st.title("Assistant de Diagnostic de Maladie Coronarienne")
st.write("Ajustez les mesures dans le panneau lateral pour obtenir une prediction du risque a 10 ans.")

# Image descriptive (Wikimedia Commons - domaine public)
# Image locale (a placer dans outputs/coronary_disease.png)
img_path = BASE_DIR / "coronary.jpg"
if img_path.exists():
    st.image(
        str(img_path),
        caption="Maladie coronarienne : retrecissement des arteres coronaires par des plaques d'atherome",
        use_container_width=False
    )
else:
    st.warning("Image non trouvee. Placez 'coronary_disease.png' dans le dossier outputs/images/")

st.markdown("---")

# 4. Panneau lateral - Saisie des parametres
st.sidebar.header("Parametres du Patient")
st.sidebar.markdown("Renseignez les valeurs cliniques du patient :")

def user_input_features():
    age          = st.sidebar.slider("Age du patient (ans)",          32,  90,  50)
    male         = st.sidebar.selectbox("Sexe", options=[0, 1],
                       format_func=lambda x: "Femme" if x == 0 else "Homme")
    sysBP        = st.sidebar.slider("Pression systolique (mmHg)",    80, 295, 130)
    diaBP        = st.sidebar.slider("Pression diastolique (mmHg)",   40, 150,  85)
    prevalentHyp = st.sidebar.selectbox("Hypertension preexistante", options=[0, 1],
                       format_func=lambda x: "Non" if x == 0 else "Oui")
    diabetes     = st.sidebar.selectbox("Diabete", options=[0, 1],
                       format_func=lambda x: "Non" if x == 0 else "Oui")

    # Features engineered (coherent avec le notebook)
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

    # Ordre exact des colonnes du modele
    data = np.array([[
        age_sysBP, risk_score, age, sysBP, pulse_pressure,
        prevalentHyp, diaBP, Tension_Hypertension, Tension_Normale,
        diabetes, male
    ]])
    return data, age, sysBP, diaBP, risk_score

input_df, age, sysBP, diaBP, risk_score = user_input_features()

# 5. Recapitulatif des parametres
with st.expander("Recapitulatif des parametres saisis", expanded=False):
    col1, col2, col3 = st.columns(3)
    col1.metric("Age", f"{age} ans")
    col2.metric("Pression systolique", f"{sysBP} mmHg")
    col3.metric("Pression diastolique", f"{diaBP} mmHg")
    col1.metric("Score de risque calcule", risk_score)
    col2.metric("Pression pulsee", f"{sysBP - diaBP} mmHg")

st.markdown("---")

# 6. Bouton de prediction
if st.button("Lancer le Diagnostic", use_container_width=True):

    prediction  = model.predict(input_df)
    probability = model.predict_proba(input_df)

    proba_risque  = probability[0][1] * 100
    proba_no_risk = probability[0][0] * 100

    st.subheader("Resultat du diagnostic")

    if prediction[0] == 1:
        st.error(
            "**Risque eleve detecte**\n\n"
            "Le patient presente un **risque de maladie coronarienne dans les 10 prochaines annees**. "
            "Une consultation cardiologique est fortement recommandee."
        )
        st.info(
            f"**Probabilite de risque coronarien : {proba_risque:.1f}%**\n\n"
            f"Le modele estime a **{proba_risque:.1f}%** la probabilite que ce patient "
            f"developpe une maladie coronarienne dans les 10 ans."
        )
    else:
        st.success(
            "**Risque faible**\n\n"
            "Le patient ne presente **pas de risque significatif** de maladie coronarienne "
            "dans les 10 prochaines annees selon les parametres fournis."
        )
        st.info(
            f"**Probabilite d'absence de risque : {proba_no_risk:.1f}%**\n\n"
            f"Le modele estime a **{proba_no_risk:.1f}%** la probabilite que ce patient "
            f"ne developpe pas de maladie coronarienne dans les 10 ans."
        )

    # Barre de progression visuelle
    st.markdown("**Niveau de risque estime :**")
    st.progress(int(proba_risque))

    # Interpretation du score de risque clinique
    st.markdown("**Interpretation du score de risque clinique :**")
    if risk_score <= 1:
        st.info("Score de risque faible (0-1)")
    elif risk_score <= 3:
        st.warning("Score de risque modere (2-3)")
    else:
        st.error("Score de risque eleve (4 et plus)")

# 7. Note legale
st.markdown("---")
st.info(
    "**Note importante** : Cet outil est base sur un modele de Regression Logistique "
    "entraine sur l'etude Framingham Heart Study. Il constitue une **aide a la decision** "
    "et ne remplace en aucun cas l'avis d'un professionnel de sante."
)
