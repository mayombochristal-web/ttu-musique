import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import io
from audiorecorder import audiorecorder # Installation: pip install streamlit-audiorecorder

# Configuration
st.set_page_config(page_title="TTU-MC3 Mini DAW", layout="wide")

# --- INTRODUCTION ---
st.title("🎙️ TTU-MC³ Voice Processor & Mini DAW")
with st.expander("📖 Qu'est-ce que la TTU-MC³ ? (Explications)", expanded=True):
    st.write("""
    Cette application traite le son selon la **Théorie Triadique Unifiée (TTU-MC³)**. 
    Contrairement aux outils classiques, nous ne filtrons pas le son : nous simulons un système physique 
    qui cherche à se stabiliser vers un **attracteur de clarté**.
    
    * **Mémoire ($\Phi_M$)** : Gère l'inertie et la texture (le "corps" du son).
    * **Cohérence ($\Phi_C$)** : Gère la synchronisation des phases (la "clarté").
    * **Dissipation ($\Phi_D$)** : Gère l'élimination du désordre (le "nettoyage").
    """)

# --- INTERFACE DE CAPTURE ---
st.header("1. Source Audio")
col_input1, col_input2 = st.columns(2)

audio_data = None
sr = 22050

with col_input1:
    st.subheader("Enregistrement Direct")
    st.write("Microphone (Ordinateur ou Smartphone)")
    # Composant pour enregistrer directement
    audio_recorded = audiorecorder("Cliquer pour Enregistrer", "Cliquer pour Arrêter")
    
    if len(audio_recorded) > 0:
        audio_data = audio_recorded.export_to_me(format="wav").read()
        st.success("Enregistrement micro capturé !")

with col_input2:
    st.subheader("Fichier Local")
    uploaded_file = st.file_uploader("Importer un fichier (WAV, MP3)", type=['wav', 'mp3'])
    if uploaded_file:
        audio_data = uploaded_file.read()

# --- MOTEUR DE TRAITEMENT ---
def apply_ttu_processing(y, sr, alpha, beta, gamma, lmbda, eta, mu):
    phi_m, phi_c, phi_d = 0.0, 0.0, 0.1
    dt = 1/sr
    y_out = np.zeros_like(y)
    
    for i in range(len(y)):
        v_t = y[i]
        # Équations du flot triadique (système différentiel)
        dpm = -alpha * phi_m + beta * phi_d
        dpc = gamma * v_t - lmbda * phi_c * phi_d
        dpd = eta * phi_c**2 - mu * phi_d
        
        phi_m += dpm * dt
        phi_c += dpc * dt
        phi_d += dpd * dt
        
        # Le signal est extrait de la composante Cohérence stabilisée
        y_out[i] = phi_c * (1.0 - 0.05 * phi_d)
        
    return np.nan_to_num(y_out)

# --- COMMANDES DE TESTS ---
if audio_data:
    st.header("2. Paramétrage & Mixage")
    
    with st.sidebar:
        st.header("Commandes de Test TTU")
        st.info("""
        🧪 **Recettes de test :**
        - **Nettoyer** : ↑ Mu , ↓ Beta
        - **Percer** : ↑ Gamma , Ajuster Lambda
        - **Épaissir** : ↓ Alpha (Inertie)
        - **Expérimenter** : ↑↑ Eta (Saturation)
        """)
        
        alpha = st.slider("Alpha (Mémoire)", 0.01, 5.0, 1.0)
        beta = st.slider("Beta (Réinjection)", 0.0, 1.0, 0.1)
        gamma = st.slider("Gamma (Cohérence)", 0.1, 10.0, 1.5)
        lmbda = st.slider("Lambda (Couplage)", 0.1, 5.0, 1.0)
        eta = st.slider("Eta (Non-linéarité)", 0.1, 10.0, 1.0)
        mu = st.slider("Mu (Dissipation)", 0.1, 10.0, 1.0)

    # Chargement du signal
    y, sr_loaded = librosa.load(io.BytesIO(audio_data), sr=sr)
    
    if st.button("🚀 Appliquer le traitement TTU-MC³"):
        with st.spinner("Calcul de l'attracteur sonore..."):
            y_processed = apply_ttu_processing(y, sr_loaded, alpha, beta, gamma, lmbda, eta, mu)
            
            # Normalisation pour éviter les saturations numériques
            max_val = np.max(np.abs(y_processed))
            if max_val > 0:
                y_processed = y_processed / max_val
            
            # Affichage résultat
            st.subheader("Résultat Final")
            out_buffer = io.BytesIO()
            sf.write(out_buffer, y_processed, sr_loaded, format='WAV')
            st.audio(out_buffer, format='audio/wav')
            
            # Comparaison visuelle
            st.line_chart({"Original": y[:1000], "TTU-MC3": y_processed[:1000]})

else:
    st.info("En attente d'une source audio (micro ou fichier)...")
