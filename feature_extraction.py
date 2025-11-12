"""Feature extraction module for audio emotion recognition."""

import os
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import logging
from config import (
    RAVDESS_EMOTION_LABELS, AUDIODATA_EMOTION_LABELS, TESS_EMOTION_LABELS,
    EMOTION_SUBSET_MAPPING, N_MFCC, NOISE_DECIMAL,
    DATASET_RAVDESS, DATASET_AUDIODATA, DATASET_TESS, OUTPUT_AUDIO_DIR
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def inject_noise(audio, decimal=NOISE_DECIMAL):
    
    x = np.random.uniform(0, 1)
    max_value = np.max(np.abs(audio))
    noise_amplitude = decimal * x * max_value
    noise = noise_amplitude * np.random.normal(size=len(audio))
    noisy_audio = audio + noise
    return np.clip(noisy_audio, -1.0, 1.0)


def extract_features(file_path):
    
    audio, sr = librosa.load(file_path, sr=None)
    
    # MFCCs, Delta, and Delta-Delta
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfccs_mean = np.mean(mfccs, axis=1)  # 40 columns
    delta_mfccs = librosa.feature.delta(mfccs)
    delta_mfccs_mean = np.mean(delta_mfccs, axis=1)  # 40 columns
    delta_delta_mfccs = librosa.feature.delta(mfccs, order=2)
    delta_delta_mfccs_mean = np.mean(delta_delta_mfccs, axis=1)  # 40 columns
    
    # Chroma Features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)  # 12 columns
    
    # Spectral Features
    spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    spec_contrast_mean = np.mean(spec_contrast, axis=1)  # 7 columns
    spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spec_centroid_mean = np.mean(spec_centroid)  # 1 column
    spec_centroid_std = np.std(spec_centroid)  # 1 column
    spec_centroid_min = np.min(spec_centroid)  # 1 column
    spec_centroid_max = np.max(spec_centroid)  # 1 column
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    spec_bandwidth_mean = np.mean(spec_bandwidth)  # 1 column
    spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    spec_rolloff_mean = np.mean(spec_rolloff)  # 1 column
    spec_flatness = librosa.feature.spectral_flatness(y=audio)
    spec_flatness_mean = np.mean(spec_flatness)  # 1 column
    
    # Spectral Flux (approximation)
    stft = np.abs(librosa.stft(audio))
    spec_flux = np.mean(np.diff(stft, axis=1) ** 2, axis=0)
    spec_flux_mean = np.mean(spec_flux)  # 1 column
    
    # Spectral Entropy (approximation)
    stft_norm = stft / (np.sum(stft, axis=0, keepdims=True) + 1e-10)
    spec_entropy = -np.sum(stft_norm * np.log2(stft_norm + 1e-10), axis=0)
    spec_entropy_mean = np.mean(spec_entropy)  # 1 column
    
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    zcr_mean = np.mean(zcr)  # 1 column
    zcr_std = np.std(zcr)  # 1 column
    zcr_min = np.min(zcr)  # 1 column
    zcr_max = np.max(zcr)  # 1 column
    
    # RMS (Energy)
    rms = librosa.feature.rms(y=audio)
    rms_mean = np.mean(rms)  # 1 column
    rms_std = np.std(rms)  # 1 column
    rms_min = np.min(rms)  # 1 column
    rms_max = np.max(rms)  # 1 column
    
    # Log-Energy
    log_energy = np.log(np.sum(audio**2) + 1e-10)  # 1 column
    
    # Pitch (Fundamental Frequency)
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    pitch = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch_value = pitches[index, t]
        if pitch_value > 0:
            pitch.append(pitch_value)
    pitch = np.array(pitch)
    pitch_mean = np.mean(pitch) if len(pitch) > 0 else 0  # 1 column
    pitch_std = np.std(pitch) if len(pitch) > 0 else 0  # 1 column
    pitch_min = np.min(pitch) if len(pitch) > 0 else 0  # 1 column
    pitch_max = np.max(pitch) if len(pitch) > 0 else 0  # 1 column
    
    # Formants (approximation using LPC)
    try:
        lpc_coeffs = librosa.lpc(audio, order=12)
        roots = np.roots(lpc_coeffs)
        roots = roots[np.imag(roots) >= 0]
        angles = np.angle(roots)
        freqs = angles * (sr / (2 * np.pi))
        freqs = freqs[freqs > 0]
        freqs = np.sort(freqs)[:3]
        formant1 = freqs[0] if len(freqs) > 0 else 0  # 1 column
        formant2 = freqs[1] if len(freqs) > 1 else 0  # 1 column
        formant3 = freqs[2] if len(freqs) > 2 else 0  # 1 column
    except:
        formant1, formant2, formant3 = 0, 0, 0
    
    # Combine all features (165 columns)
    return np.hstack([
        mfccs_mean,  # 40
        delta_mfccs_mean,  # 40
        delta_delta_mfccs_mean,  # 40
        chroma_mean,  # 12
        spec_contrast_mean,  # 7
        [spec_centroid_mean, spec_centroid_std, spec_centroid_min, spec_centroid_max],  # 4
        [spec_bandwidth_mean],  # 1
        [spec_rolloff_mean],  # 1
        [spec_flatness_mean],  # 1
        [spec_flux_mean],  # 1
        [spec_entropy_mean],  # 1
        [zcr_mean, zcr_std, zcr_min, zcr_max],  # 4
        [rms_mean, rms_std, rms_min, rms_max],  # 4
        [log_energy],  # 1
        [pitch_mean, pitch_std, pitch_min, pitch_max],  # 4
        [formant1, formant2, formant3]  # 3
    ])


def process_file(args):
    
    idx, file, emotion, output_audio_dir = args
    try:
        # Map primary emotion to subset
        emotion_subset = EMOTION_SUBSET_MAPPING.get(emotion, "Unknown")
        if emotion_subset == "Unknown":
            logging.warning(f"Skipping file {file} due to unknown emotion: {emotion}")
            return []
        
        result = []
        # Original audio features
        logging.info(f"Processing original audio: {file}")
        original_features = extract_features(file)
        result.append(np.append(original_features, emotion_subset))
        
        # Noise-injected audio features
        logging.info(f"Processing noisy audio: {file}")
        audio, sr = librosa.load(file, sr=None)
        noisy_audio = inject_noise(audio, decimal=NOISE_DECIMAL)
        noisy_file_path = os.path.join(output_audio_dir, f"noisy_{idx}_{os.path.basename(file)}")
        sf.write(noisy_file_path, noisy_audio, sr)
        noisy_features = extract_features(noisy_file_path)
        result.append(np.append(noisy_features, emotion_subset))
        
        # Clean up
        os.remove(noisy_file_path)
        return result
    except Exception as e:
        logging.error(f"Error processing file {file}: {str(e)}")
        return []


def collect_audio_files():
    
    audio_files = []
    emotions = []
    
    # Process RAVDESS (Audio_Speech_Actors_01-24)
    for root, _, files in os.walk(DATASET_RAVDESS):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".wav"):
                emotion_code = file.split("-")[2]
                emotion = RAVDESS_EMOTION_LABELS.get(emotion_code, "Unknown")
                audio_files.append(file_path)
                emotions.append(emotion)
    
    # Process AudioData
    for file in os.listdir(DATASET_AUDIODATA):
        if file.endswith(".wav"):
            file_path = os.path.join(DATASET_AUDIODATA, file)
            prefix = file.split(".")[0]
            for key in AUDIODATA_EMOTION_LABELS.keys():
                if prefix.startswith(key):
                    emotion = AUDIODATA_EMOTION_LABELS[key]
                    audio_files.append(file_path)
                    emotions.append(emotion)
                    break
    
    # Process TESS
    for root, _, files in os.walk(DATASET_TESS):
        folder_name = os.path.basename(root)
        emotion = TESS_EMOTION_LABELS.get(folder_name, None)
        if emotion:
            for file in files:
                if file.endswith(".wav"):
                    file_path = os.path.join(root, file)
                    audio_files.append(file_path)
                    emotions.append(emotion)
    
    return audio_files, emotions


def extract_and_save_features(output_path):
    
    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)
    
    # Collect audio files
    audio_files, emotions = collect_audio_files()
    
    # Sequential processing
    feature_list = []
    args = [(idx, file, emotion, OUTPUT_AUDIO_DIR) 
            for idx, (file, emotion) in enumerate(zip(audio_files, emotions))]
    
    for arg in args:
        result = process_file(arg)
        feature_list.extend(result)
    
    # Define column names (165 features + 1 emotion = 166 columns)
    audio_features = (
        [f"MFCC_{i+1}" for i in range(40)] +
        [f"Delta_MFCC_{i+1}" for i in range(40)] +
        [f"Delta_Delta_MFCC_{i+1}" for i in range(40)] +
        [f"Chroma_{i+1}" for i in range(12)] +
        [f"Spectral_Contrast_{i+1}" for i in range(7)] +
        ["Spectral_Centroid_Mean", "Spectral_Centroid_Std", "Spectral_Centroid_Min", "Spectral_Centroid_Max"] +
        ["Spectral_Bandwidth_Mean"] +
        ["Spectral_Rolloff_Mean"] +
        ["Spectral_Flatness_Mean"] +
        ["Spectral_Flux_Mean"] +
        ["Spectral_Entropy_Mean"] +
        ["Zero_Crossing_Rate_Mean", "Zero_Crossing_Rate_Std", "Zero_Crossing_Rate_Min", "Zero_Crossing_Rate_Max"] +
        ["RMS_Mean", "RMS_Std", "RMS_Min", "RMS_Max"] +
        ["Log_Energy"] +
        ["Pitch_Mean", "Pitch_Std", "Pitch_Min", "Pitch_Max"] +
        ["Formant_1", "Formant_2", "Formant_3"] +
        ["Emotion"]
    )
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(feature_list, columns=audio_features)
    df.to_csv(output_path, index=False)
    
    logging.info(f"Feature extraction complete! Saved as {output_path}")
    return df


if __name__ == "__main__":
    from config import OUTPUT_CSV_ORIGINAL
    extract_and_save_features(OUTPUT_CSV_ORIGINAL)
