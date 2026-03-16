import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import io
from audiorecorder import audiorecorder

# Configuration de la page
st.set_page_config(page_title="TTU-MC3 : Souveraineté Vocale", layout="wide")

# --- MOTEUR DE TRAITEMENT TTU-MC3 ---
def process_ttu_vocal(y, sr, a, b, g, l, e, m):
    if len(y) == 0: return y
    dt = 1.0 / sr
    y_out = np.zeros_like(y)
    phi_m, phi_c, phi_d = 0.0, 0.0, 0.1
    
    for i in range(len(y)):
        v_t = y[i]
        # Équations du flot triadique
        dpm = -a * phi_m + b * phi_d
        dpc = g * v_t - l * phi_c * phi_d
        dpd = e * (phi_c**2) - m * phi_d
        
        phi_m += dpm * dt
        phi_c += dpc * dt
        phi_d += dpd * dt
        
        # Sortie stabilisée avec compression naturelle (tanh)
        y_out[i] = np.tanh(phi_c * (1.0 - 0.05 * phi_d))
        
    return np.nan_to_num(y_out)

# --- INTERFACE ---
st.title("🎙️ TTU-MC³ : Souveraineté Vocale")

with st.sidebar:
    st.header("🎛️ Paramètres TTU")
    st.info("🧪 **Recettes :**\n\n- **Nettoyer** : ↑ Mu , ↓ Beta\n- **Percer** : ↑ Gamma , Ajuster Lambda\n- **Épaissir** : ↓ Alpha\n- **Expérimenter** : ↑↑ Eta")
    
    alpha = st.slider("Alpha (Inertie)", 0.1, 5.0, 1.0)
    beta = st.slider("Beta (Réinjection)", 0.0, 1.0, 0.1)
    gamma = st.slider("Gamma (Gain)", 0.1, 10.0, 1.5)
    lmbda = st.slider("Lambda (Couplage)", 0.1, 5.0, 1.0)
    eta = st.slider("Eta (Non-linéarité)", 0.1, 10.0, 1.0)
    mu = st.slider("Mu (Dissipation)", 0.1, 10.0, 1.0)

st.header("1. Capture de la source")
col1, col2 = st.columns(2)

audio_data = None

with col1:
    st.write("🎤 **Microphone**")
    # Utilisation du recorder
    try:
        audio_recorded = audiorecorder("Démarrer Micro", "Arrêter & Valider")
        if len(audio_recorded) > 0:
            audio_data = audio_recorded.export_to_me(format="wav").read()
    except Exception as e:
        st.error("Erreur d'accès au micro ou FFmpeg manquant.")

with col2:
    st.write("📂 **Importer un Master**")
    uploaded = st.file_uploader("Fichier WAV/MP3", type=['wav', 'mp3'])
    if uploaded:
        audio_data = uploaded.read()

if audio_data:
    st.header("2. Traitement & Stabilisation")
    if st.button("🚀 LANCER LE MASTERING TTU-MC³"):
        with st.spinner("Stabilisation de la trajectoire vocale..."):
            # Charger l'audio
            y, sr_in = librosa.load(io.BytesIO(audio_data), sr=22050)
            
            # Traiter
            y_final = process_ttu_vocal(y, sr_in, alpha, beta, gamma, lmbda, eta, mu)
            
            # Normaliser
            y_final = y_final / (np.max(np.abs(y_final)) + 1e-9)
            
            # Résultat
            out_io = io.BytesIO()
            sf.write(out_io, y_final, sr_in, format='WAV')
            st.audio(out_io, format='audio/wav')
            st.success("Master terminé.")
