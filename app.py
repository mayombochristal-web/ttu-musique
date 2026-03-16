import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import io
import os
from audiorecorder import audiorecorder

# --- CONFIGURATION & STYLE ---
st.set_page_config(page_title="TTU-MC3 : Souveraineté Vocale", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- INTRODUCTION ---
st.title("🎙️ TTU-MC³ : Souveraineté Vocale")
st.subheader("Transformez votre voix brute en un Master professionnel.")

with st.expander("📖 Comprendre la stabilisation TTU-MC³", expanded=False):
    st.write("""
    Le moteur utilise le **flot triadique** défini dans vos travaux :
    - **Mémoire ($\Phi_M$)** : Stabilise l'enveloppe spectrale.
    - **Cohérence ($\Phi_C$)** : Filtre les bruits pour ne garder que le signal utile (la voix).
    - **Dissipation ($\Phi_D$)** : Élimine l'entropie (bruit de fond) par convergence vers l'attracteur.
    """)

# --- BARRE LATÉRALE : COMMANDES DE TEST ---
with st.sidebar:
    st.header("🎛️ Paramètres TTU")
    st.info("🧪 **Recettes :**\n\n- **Nettoyer** : ↑ Mu , ↓ Beta\n- **Percer** : ↑ Gamma , Ajuster Lambda\n- **Épaissir** : ↓ Alpha\n- **Expérimenter** : ↑↑ Eta")
    
    alpha = st.slider("Alpha (Inertie/Mémoire)", 0.1, 5.0, 1.0)
    beta = st.slider("Beta (Réinjection)", 0.0, 1.0, 0.1)
    gamma = st.slider("Gamma (Gain Cohérence)", 0.1, 10.0, 1.5)
    lmbda = st.slider("Lambda (Couplage)", 0.1, 5.0, 1.0)
    eta = st.slider("Eta (Non-linéarité)", 0.1, 10.0, 1.0)
    mu = st.slider("Mu (Atténuation Dissipative)", 0.1, 10.0, 1.0)

# --- MOTEUR DE CALCUL (FLOT TRIADIQUE) ---
def process_ttu_vocal(y, sr, a, b, g, l, e, m):
    if len(y) == 0: return y
    
    dt = 1.0 / sr
    y_out = np.zeros_like(y)
    # États initiaux
    phi_m, phi_c, phi_d = 0.0, 0.0, 0.1
    
    # Simulation du système dynamique
    for i in range(len(y)):
        v_t = y[i]
        
        # Équations différentielles de la TTU-MC3
        dpm = -a * phi_m + b * phi_d
        dpc = g * v_t - l * phi_c * phi_d
        dpd = e * phi_c**2 - m * phi_d
        
        phi_m += dpm * dt
        phi_c += dpc * dt
        phi_d += dpd * dt
        
        # Extraction du signal stabilisé (Soft Clipping intégré)
        y_out[i] = np.tanh(phi_c * (1.0 - 0.05 * phi_d))
        
    return np.nan_to_num(y_out)

# --- SECTION CAPTURE ---
st.header("1. Capture de la source")
col1, col2 = st.columns(2)

audio_bytes = None

with col1:
    st.write("🎤 **Enregistrement Direct (Micro/Tel)**")
    # Utilisation du recorder
    audio_recorded = audiorecorder("Démarrer Micro", "Arrêter & Valider")
    if len(audio_recorded) > 0:
        audio_bytes = audio_recorded.export_to_me(format="wav").read()
        st.success("Signal capturé via le micro.")

with col2:
    st.write("📂 **Importer un fichier (TTU Musique)**")
    uploaded = st.file_uploader("Fichier WAV ou MP3", type=['wav', 'mp3'])
    if uploaded:
        audio_bytes = uploaded.read()

# --- SECTION TRAITEMENT ---
if audio_bytes:
    st.header("2. Traitement & Master")
    
    # Lecture audio originale pour prévisualisation
    st.audio(audio_bytes, format='audio/wav')
    
    if st.button("🚀 LANCER LA STABILISATION TTU-MC³"):
        with st.spinner("Calcul de la trajectoire vers l'attracteur sonore..."):
            try:
                # Chargement avec librosa
                y, sr_in = librosa.load(io.BytesIO(audio_bytes), sr=22050)
                
                # Application du moteur
                y_final = process_ttu_vocal(y, sr_in, alpha, beta, gamma, lmbda, eta, mu)
                
                # Normalisation Finale
                y_final = y_final / (np.max(np.abs(y_final)) + 1e-9)
                
                # Sortie
                st.subheader("✅ Résultat : Master Stabilisé")
                out_io = io.BytesIO()
                sf.write(out_io, y_final, sr_in, format='WAV')
                st.audio(out_io, format='audio/wav')
                
                # Visualisation
                st.line_chart({"Original": y[1000:2000], "Master TTU": y_final[1000:2000]})
                
            except Exception as ex:
                st.error(f"Erreur de traitement : {ex}")
                st.info("Note : Si vous êtes sur Cloud, vérifiez la présence de FFmpeg dans packages.txt")
else:
    st.warning("En attente d'un signal audio...")

st.markdown("---")
st.caption("TTU-MC³ Engine | USTM Franceville | Développé pour GEB")
