"""
Audio Feature Analyzer
=======================
Extracts music features from audio files using librosa.

Extracts:
  - duration_min: Song duration in minutes
  - tempo_bpm: Estimated beats per minute (with half-tempo correction)
  - energy: RMS energy normalized to 0-1
  - danceability: Estimated from beat strength & tempo regularity (0-1)
  - loudness_db: Average loudness in dB (approx -30 to 0)

Known fix: librosa's beat_track often detects half-tempo for bass-heavy and
EDM tracks. We use multi-method tempo estimation + prior weighting to correct this.
"""

import os
import numpy as np
import librosa


def _estimate_tempo_robust(y, sr):
    """
    Robust tempo estimation that corrects librosa's half-tempo problem.
    
    Strategy:
     1. Use librosa.beat.beat_track for initial tempo
     2. Use librosa.beat.tempo with a prior (centered around typical music range)
     3. Use onset-based autocorrelation as a third estimate
     4. Compare candidates and prefer the one closest to 80-160 BPM range
        (since most popular music falls here)
     5. If the detected tempo is suspiciously low (< 80 BPM), check if doubling
        puts it in a more common range
    """
    
    # Method 1: Standard beat_track
    tempo1, _ = librosa.beat.beat_track(y=y, sr=sr)
    if hasattr(tempo1, '__len__'):
        tempo1 = float(tempo1[0]) if len(tempo1) > 0 else 120.0
    else:
        tempo1 = float(tempo1)
    
    # Method 2: tempo() with start_bpm prior
    # This biases toward the 120 BPM range (more common for pop/EDM)
    try:
        tempo2 = librosa.feature.rhythm.tempo(y=y, sr=sr, start_bpm=120)
        if hasattr(tempo2, '__len__'):
            tempo2 = float(tempo2[0]) if len(tempo2) > 0 else tempo1
        else:
            tempo2 = float(tempo2)
    except Exception:
        tempo2 = tempo1
    
    # Method 3: Onset-based autocorrelation
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        # Use tempogram for multi-resolution tempo analysis
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        # Get the dominant tempo from the tempogram's mean
        ac_global = np.mean(tempogram, axis=1)
        # Convert lag to BPM: BPM = 60 * sr / (hop_length * lag)
        hop_length = 512  # librosa default
        freqs = librosa.tempo_frequencies(len(ac_global), sr=sr, hop_length=hop_length)
        # Find the peak in a reasonable BPM range
        valid = (freqs >= 60) & (freqs <= 220)
        if np.any(valid):
            valid_idx = np.where(valid)[0]
            peak_idx = valid_idx[np.argmax(ac_global[valid_idx])]
            tempo3 = freqs[peak_idx]
        else:
            tempo3 = tempo1
    except Exception:
        tempo3 = tempo1
    
    # Collect candidates (original and doubled versions)
    candidates = []
    for t in [tempo1, tempo2, tempo3]:
        candidates.append(t)
        candidates.append(t * 2)  # doubled
        if t > 80:
            candidates.append(t / 2)  # halved
    
    # Score each candidate: prefer tempos in the 80-170 BPM range
    # This is where the majority of popular music sits
    def score(bpm):
        if bpm < 40 or bpm > 250:
            return -100
        # Sweet spot: 80-170 BPM
        if 80 <= bpm <= 170:
            return 10
        elif 70 <= bpm <= 180:
            return 5
        else:
            return 0
    
    # Also add agreement bonus: if multiple methods agree, that's better
    scored = []
    for c in candidates:
        s = score(c)
        # Count how many raw estimates are within 5% of this candidate (or its double/half)
        for raw in [tempo1, tempo2, tempo3]:
            if abs(c - raw) / max(c, 1) < 0.05:
                s += 3  # exact match bonus
            elif abs(c - raw * 2) / max(c, 1) < 0.05:
                s += 2  # double match
            elif abs(c - raw / 2) / max(c, 1) < 0.05:
                s += 2  # half match
        scored.append((s, c))
    
    # Pick the best scoring tempo
    scored.sort(key=lambda x: (-x[0], abs(x[1] - 120)))  # highest score, closest to 120
    best_tempo = scored[0][1]
    
    return round(best_tempo, 1)


def analyze_audio(file_path):
    """
    Analyze an audio file and extract 5 features for popularity prediction.
    
    Args:
        file_path: Path to audio file (mp3, wav, ogg, flac, etc.)
    
    Returns:
        dict with keys: duration_min, tempo_bpm, energy, danceability, loudness_db
    
    Raises:
        ValueError: If the file cannot be loaded or analyzed
    """
    try:
        # 1. Get full duration first (fast, doesn't decode whole file)
        duration_sec = librosa.get_duration(path=file_path)
        
        # 2. Load only first 60 seconds for analysis to save RAM (Render Free Tier limit)
        #    Most features (tempo, energy, etc.) are consistent throughout the track.
        y, sr = librosa.load(file_path, sr=22050, mono=True, duration=60)
    except Exception as e:
        raise ValueError(f"ไม่สามารถโหลดไฟล์เสียงได้: {str(e)}")

    if len(y) == 0:
        raise ValueError("ไฟล์เสียงว่างเปล่า")

    # 1. Duration (minutes) - Use FULL duration
    duration_min = round(duration_sec / 60.0, 2)
    duration_min = max(1.5, min(8.0, duration_min))  # Clip to model range

    # 2. Tempo (BPM) - Using robust multi-method estimation
    tempo_bpm = _estimate_tempo_robust(y, sr)
    tempo_bpm = round(max(60, min(200, tempo_bpm)), 1)

    # 3. Energy (RMS-based, normalized 0-1)
    #    Use a wider normalization reference. Typical mastered tracks have
    #    RMS around 0.1-0.3; very loud tracks might hit 0.4-0.5.
    #    Using 0.4 as reference instead of 0.25 prevents saturation at 1.0.
    rms = librosa.feature.rms(y=y)[0]
    mean_rms = float(np.mean(rms))
    
    # Normalization: map through a curve that doesn't saturate easily
    # Using sqrt to compress high values and spread low values
    energy = np.sqrt(min(1.0, mean_rms / 0.35))
    energy = round(max(0.0, min(1.0, energy)), 3)

    # 4. Danceability (improved estimation)
    #    Components:
    #    - Beat regularity (autocorrelation of onset strength)
    #    - Onset rate (more onsets = more danceable)
    #    - Spectral contrast (rhythmic variation)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    
    # Beat regularity via autocorrelation
    ac = librosa.autocorrelate(onset_env, max_size=sr // 512)
    if len(ac) > 1:
        ac_norm = ac / (ac[0] + 1e-10)
        # Look for strong peaks in the beat-lag range
        beat_range = ac_norm[1:min(len(ac_norm), 150)]
        regularity = float(np.max(beat_range))  # Peak regularity, not mean
        regularity = min(1.0, regularity)
    else:
        regularity = 0.5
    
    # Onset rate (normalized per second)
    onsets = librosa.onset.onset_detect(y=y, sr=sr)
    onset_rate = len(onsets) / max(duration_sec, 1.0)
    # Normalize: typical 2-5 onsets/sec for danceable tracks
    onset_factor = min(1.0, onset_rate / 5.0)
    
    # Tempo factor: preferred dance range 95-135 BPM
    tempo_factor = 1.0
    if tempo_bpm < 85 or tempo_bpm > 150:
        tempo_factor = 0.7
    elif 95 <= tempo_bpm <= 135:
        tempo_factor = 1.0
    else:
        tempo_factor = 0.85
    
    danceability = (0.35 * regularity + 
                    0.25 * onset_factor + 
                    0.20 * energy + 
                    0.20 * tempo_factor)
    danceability = round(max(0.0, min(1.0, danceability)), 3)

    # 5. Loudness (dB)
    # Use a percentile-based approach for more stable loudness
    if mean_rms > 0:
        # Use the 90th percentile of RMS for a more representative loudness
        rms_90 = float(np.percentile(rms, 90))
        loudness_db = float(20 * np.log10(rms_90 + 1e-10))
        loudness_db = round(max(-30, min(0, loudness_db)), 1)
    else:
        loudness_db = -30.0

    features = {
        'duration_min': duration_min,
        'tempo_bpm': tempo_bpm,
        'energy': energy,
        'danceability': danceability,
        'loudness_db': loudness_db,
    }

    return features


def get_audio_info(file_path):
    """
    Get basic audio file information.
    
    Returns:
        dict with file_name, file_size_mb, duration_sec, sample_rate
    """
    try:
        # Just load a tiny bit to get sample rate, use get_duration for length
        # This avoids decoding the whole file for basic info
        duration_sec = librosa.get_duration(path=file_path)
        y, sr = librosa.load(file_path, sr=22050, mono=True, duration=5.0) 
        
        file_size = os.path.getsize(file_path)

        return {
            'file_name': os.path.basename(file_path),
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'duration_sec': round(duration_sec, 1),
            'sample_rate': sr,
        }
    except Exception:
        # Fallback if optimized loading fails
        return {
            'file_name': os.path.basename(file_path),
            'file_size_mb': 0,
            'duration_sec': 0,
            'sample_rate': 22050
        }



if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"🎵 Analyzing: {path}")
        features = analyze_audio(path)
        print("\n📊 Extracted Features:")
        for k, v in features.items():
            print(f"   {k}: {v}")
    else:
        print("Usage: python audio_analyzer.py <audio_file>")
