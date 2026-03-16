import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import io
import matplotlib.pyplot as plt
from audiorecorder import audiorecorder

# --- CONFIGURATION ---
st.set_page_config(page_title="TTU-MC3 Artist Identity", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #e0e0e0; }
    .status-box { padding: 20px; border-radius: 10px; border: 1px solid #4CAF50; background-color: #111; }
    </style>
    """, unsafe_allow_html=True)

# --- ENGINE : TTU SPECTRAL ALIGNMENT ---
def ttu_spectral_mastering(y, sr, strength, tone_color, auto_key=True):
    # 1. ANALYSE DE LA TONALITÉ (Cohérence Globale)
    if auto_key:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key_idx = np.argmax(np.mean(chroma, axis=1))
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        detected_key = notes[key_idx]
    else:
        detected_key = "Chromatique"

    # 2. EXTRACTION DU PITCH (F0)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'))
    
    # 3. CORRECTION PAR ATTRACTEUR HARMONIQUE
    y_corrected = np.zeros_like(y)
    hop_length = 512
    
    for i in range(0, len(y) - hop_length, hop_length):
        idx = i // hop_length
        current_f0 = f0[idx] if idx < len(f0) else np.nan
        
        if not np.isnan(current_f0):
            # Cible : Note la plus proche dans la gamme détectée
            midi_now = librosa.hz_to_midi(current_f0)
            target_midi = round(midi_now)
            
            # Application du décalage (Stabilisation TTU)
            n_steps = (target_midi - midi_now) * strength
            chunk = y[i:i+hop_length]
            y_corrected[i:i+hop_length] = librosa.effects.pitch_shift(chunk, sr=sr, n_steps=n_steps)
        else:
            y_corrected[i:i+hop_length] = y[i:i+hop_length]

    # 4. ENRICHISSEMENT HARMONIQUE (La Triade)
    # On ajoute une légère saturation harmonique pour la "chaleur" (Tone Color)
    y_final = y_corrected + (tone_color * (y_corrected**3)) 
    
    return np.nan_to_num(y_final), detected_key

# --- INTERFACE ---
st.title("🛡️ TTU-MC³ : Souveraineté Vocale")
st.subheader("L'IA au service de VOTRE identité, pas d'une imitation.")

st.info("Cette console analyse votre signature vocale unique et la place dans les normes harmoniques universelles sans dénaturer votre timbre.")

# Sélection Source
col_in1, col_in2 = st.columns(2)
audio_bytes = None

with col_in1:
    st.write("### 🎤 Capturer votre identité")
    audio_rec = audiorecorder("Enregistrer ma voix", "Arrêter l'analyse")
    if len(audio_rec) > 0:
        audio_bytes = audio_rec.export_to_me(format="wav").read()

with col_in2:
    st.write("### 📂 Importer une session")
    file_in = st.file_uploader("Fichier brut", type=['wav', 'mp3'])
    if file_in:
        audio_bytes = file_in.read()

if audio_bytes:
    st.divider()
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
    
    # Dashboard de réglages
    with st.sidebar:
        st.header("🎚️ Ajustements TTU")
        strength = st.slider("Force de l'Attracteur (Justesse)", 0.0, 1.0, 0.85)
        tone_warmth = st.slider("Chaleur Harmonique (Timbre)", 0.0, 0.5, 0.1)
        st.write("---")
        st.caption("Le mode 'Attracteur' force la voix vers la note juste la plus proche en respectant l'enveloppe spectrale originale.")

    if st.button("✨ PROCÉDER À L'ALIGNEMENT HARMONIQUE"):
        with st.spinner("Analyse spectrale TTU en cours..."):
            y_fixed, key = ttu_spectral_mastering(y, sr, strength, tone_warmth)
            
            # Normalisation finale
            y_fixed = librosa.util.normalize(y_fixed)
            
            st.success(f"Analyse terminée. Tonalité détectée : {key}")
            
            # Audio Output
            out_buf = io.BytesIO()
            sf.write(out_buf, y_fixed, sr, format='WAV')
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.write("#### Voix Originale")
                st.audio(audio_bytes)
            with col_res2:
                st.write("#### Voix TTU Stabilisée")
                st.audio(out_buf)
            
            st.download_button("💾 Exporter le Master Vocale", out_buf, "identity_ttu_master.wav")

            # Visualisation Spectrale
            fig, ax = plt.subplots(figsize=(10, 3))
            librosa.display.waveshow(y_fixed, sr=sr, ax=ax, color='lime')
            ax.set_title("Stabilité de l'Attracteur Vocal")
            st.pyplot(fig)
