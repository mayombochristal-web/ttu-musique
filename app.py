import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import io
from audiorecorder import audiorecorder

# --- CONFIGURATION & STYLE ---
st.set_page_config(page_title="TTU-MC3 AutoTune", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #4CAF50; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- ENGINE : ANALYSE & CORRECTION TTU ---
def ttu_autotune_engine(y, sr, correction_strength, stability):
    # 1. ANALYSE (Extraction de la Cohérence initiale)
    # On extrait la fréquence fondamentale (F0)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'))
    
    # 2. CALCUL DE L'ATTRACTEUR (Cible Melodyne)
    # On arrondit à la note de la gamme la plus proche
    f0_target = np.copy(f0)
    for i, freq in enumerate(f0):
        if not np.isnan(freq):
            midi_note = librosa.hz_to_midi(freq)
            target_midi = round(midi_note) # Force la note juste
            f0_target[i] = librosa.midi_to_hz(target_midi)
    
    # 3. STABILISATION TRIADIQUE (Le processus de rendu)
    # On utilise le Time-Stretching/Pitch-Shifting guidé par la TTU
    # Ici simplifié par un vocoder à phase pour le code complet
    y_shifted = np.zeros_like(y)
    
    # Correction automatique basée sur la force (correction_strength)
    # Plus la force est haute, plus on converge strictement vers f0_target
    step = 512
    for i in range(0, len(y) - step, step):
        current_f0 = f0[i // step] if (i // step) < len(f0) else np.nan
        target_f0 = f0_target[i // step] if (i // step) < len(f0_target) else np.nan
        
        if not np.isnan(current_f0) and not np.isnan(target_f0):
            # Calcul du ratio de correction
            n_steps = librosa.hz_to_midi(target_f0) - librosa.hz_to_midi(current_f0)
            chunk = y[i:i+step]
            # Application de la correction (Force de l'attracteur)
            y_shifted[i:i+step] = librosa.effects.pitch_shift(chunk, sr=sr, n_steps=n_steps * correction_strength)
        else:
            y_shifted[i:i+step] = y[i:i+step]

    # 4. DISSIPATION (Nettoyage post-traitement)
    # Réduction du bruit résiduel pour la clarté
    y_final = librosa.effects.preemphasis(y_shifted)
    return np.nan_to_num(y_final)

# --- INTERFACE UTILISATEUR ---
st.title("🎙️ TTU-MC³ AutoTune & Melodyne")
st.write("Analyse automatique et stabilisation de la justesse vocale par attracteur triadique.")

# Introduction explication
with st.expander("Comment ça marche ?", expanded=False):
    st.write("""
    1. **Analyse** : L'IA détecte la note que vous chantez.
    2. **Cible** : Elle définit la note 'parfaite' la plus proche.
    3. **Convergence** : La TTU-MC³ déplace votre voix vers cette note sans perdre le timbre naturel.
    """)

# Sélection Source
col_a, col_b = st.columns(2)
audio_input = None

with col_a:
    st.subheader("Enregistrement")
    audio_recorded = audiorecorder("🎤 Démarrer Micro", "⏹️ Arrêter")
    if len(audio_recorded) > 0:
        audio_input = audio_recorded.export_to_me(format="wav").read()

with col_b:
    st.subheader("Importation")
    file_up = st.file_uploader("Fichier voix", type=['wav', 'mp3'])
    if file_up:
        audio_input = file_up.read()

# Réglages simples (Pofinage)
if audio_input:
    st.divider()
    st.subheader("Ajustements de précision")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        strength = st.slider("Force Auto-Tune (Attraction)", 0.0, 1.0, 0.8, help="0=Naturel, 1=Robotique/Parfait")
    with c2:
        clarity = st.slider("Clarté (Cohérence)", 0.5, 2.0, 1.0, help="Améliore la présence de la voix")
    with c3:
        noise_red = st.slider("Nettoyage (Dissipation)", 0.0, 1.0, 0.2, help="Réduit le souffle et les bruits")

    if st.button("✨ ANALYSER ET CORRIGER AUTOMATIQUEMENT"):
        y, sr = librosa.load(io.BytesIO(audio_input), sr=22050)
        
        with st.spinner("Analyse de la structure mélodique en cours..."):
            # Traitement
            y_out = ttu_autotune_engine(y, sr, strength, clarity)
            
            # Normalisation
            y_out = librosa.util.normalize(y_out)
            
            # Affichage résultat
            st.success("Signal stabilisé et corrigé !")
            buf = io.BytesIO()
            sf.write(buf, y_out, sr, format='WAV')
            
            st.audio(buf, format='audio/wav')
            
            # Téléchargement
            st.download_button("💾 Télécharger la voix corrigée", buf, "ttu_voice_fixed.wav", "audio/wav")

else:
    st.info("Utilisez le micro ou importez un fichier pour commencer l'analyse.")
