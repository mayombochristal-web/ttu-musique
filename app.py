import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import io
try:
    import matplotlib.pyplot as plt
except ImportError:
    st.error("Matplotlib est manquant. Ajoutez 'matplotlib' à votre requirements.txt")

from audiorecorder import audiorecorder

# --- CONFIGURATION ---
st.set_page_config(page_title="TTU-MC3 Artist Identity", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #e0e0e0; }
    .stButton>button { background-color: #2e7d32; color: white; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- MOTEUR TTU-MC3 ADVANCED ---
def process_voice_identity(y, sr, strength, warmth):
    # 1. ANALYSE (Extraction de l'identité fréquentielle)
    # On utilise pyin pour une détection de pitch haute précision
    f0, voiced_flag, voiced_probs = librosa.pyin(y, 
                                                 fmin=librosa.note_to_hz('C2'), 
                                                 fmax=librosa.note_to_hz('C6'))
    
    # 2. ALIGNEMENT (Attracteur vers la note juste)
    y_corrected = np.zeros_like(y)
    hop_length = 512
    
    for i in range(0, len(y) - hop_length, hop_length):
        idx = i // hop_length
        f0_current = f0[idx] if idx < len(f0) else np.nan
        
        if not np.isnan(f0_current) and voiced_flag[idx]:
            # Calcul de la note cible (Melodyne style)
            midi_note = librosa.hz_to_midi(f0_current)
            target_midi = round(midi_note)
            n_steps = (target_midi - midi_note) * strength
            
            # Correction par décalage de phase (préserve le timbre)
            chunk = y[i:i+hop_length]
            y_corrected[i:i+hop_length] = librosa.effects.pitch_shift(chunk, sr=sr, n_steps=n_steps)
        else:
            y_corrected[i:i+hop_length] = y[i:i+hop_length]

    # 3. CHALEUR HARMONIQUE (MC3 : Cohérence & Dissipation)
    # Ajout d'une saturation douce pour simuler un préampli à lampe
    y_final = y_corrected + (warmth * np.tanh(y_corrected * 2))
    
    return librosa.util.normalize(y_final)

# --- INTERFACE ---
st.title("🎙️ TTU-MC³ : Souveraineté Vocale")
st.write("### Transformez votre voix brute en un Master professionnel tout en restant VOUS-MÊME.")

# Zone de capture
col1, col2 = st.columns(2)
audio_source = None

with col1:
    st.info("🎤 Enregistrement Direct")
    audio_rec = audiorecorder("Démarrer l'enregistrement", "Arrêter")
    if len(audio_rec) > 0:
        audio_source = audio_rec.export_to_me(format="wav").read()

with col2:
    st.info("📂 Importation Studio")
    file_up = st.file_uploader("Fichier vocal (.wav, .mp3)", type=['wav', 'mp3'])
    if file_up:
        audio_source = file_up.read()

if audio_source:
    st.divider()
    y, sr = librosa.load(io.BytesIO(audio_source), sr=22050)
    
    # Réglages
    with st.sidebar:
        st.header("🎚️ Paramètres d'Identité")
        strength = st.slider("Précision (Attraction)", 0.0, 1.0, 0.8)
        warmth = st.slider("Chaleur (Harmoniques)", 0.0, 0.5, 0.1)
        st.caption("La 'Précision' ramène votre voix vers la note parfaite sans robotiser le grain.")

    if st.button("✨ GÉNÉRER LE MASTER TTU"):
        with st.spinner("Stabilisation de la signature vocale..."):
            y_output = process_voice_identity(y, sr, strength, warmth)
            
            # Export
            out_buf = io.BytesIO()
            sf.write(out_buf, y_output, sr, format='WAV')
            
            st.success("Master terminé !")
            st.audio(out_buf, format='audio/wav')
            
            # Visualisation
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.plot(y_output[:2000], color='#2e7d32')
            ax.set_axis_off()
            st.pyplot(fig)
            
            st.download_button("💾 Télécharger le Master", out_buf, "ttu_artist_master.wav")
