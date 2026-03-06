
import os
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
from pathlib import Path

def generate_tone(frequency, duration, sr=8000, noise_level=0.01):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Generate a tone with some harmonics to make it "voice-like"
    sig = 0.6 * np.sin(2 * np.pi * frequency * t)
    sig += 0.3 * np.sin(2 * np.pi * 2 * frequency * t)
    sig += 0.1 * np.sin(2 * np.pi * 3 * frequency * t)
    
    # Add noise (jitter/shimmer simulation essentially)
    noise = np.random.normal(0, noise_level, sig.shape)
    sig = sig + noise
    
    # Normalize
    sig = sig / np.max(np.abs(sig))
    return sig

def main():
    base_dir = Path("synthetic_voice_data")
    if base_dir.exists():
        import shutil
        shutil.rmtree(base_dir)
    base_dir.mkdir()

    # Create subdirectories structure like VOC-ALS
    folders = ["phonationA", "phonationE", "phonationI", "phonationO", "phonationU", 
               "rhythmKA", "rhythmPA", "rhythmTA"]
    for f in folders:
        (base_dir / f).mkdir()

    # Generate synthetic subjects
    # CT = Control, PZ = Patient
    n_controls = 10
    n_patients = 10
    
    print(f"Generating synthetic audio data in {base_dir}...")

    # Controls (Healthy) - Cleaner sound, stable pitch
    for i in range(1, n_controls + 1):
        subject_id = f"CT{i:03d}"
        
        # Vowels: A, E, I, O, U with varying frequencies
        vowels = {
            "phonationA": np.random.uniform(100, 250),
            "phonationE": np.random.uniform(120, 270),
            "phonationI": np.random.uniform(140, 300),
            "phonationO": np.random.uniform(90, 200),
            "phonationU": np.random.uniform(80, 180),
        }
        
        for vowel, freq in vowels.items():
            sig = generate_tone(freq, duration=3.0, noise_level=0.005)
            wav.write(base_dir / vowel / f"{subject_id}_{vowel}.wav", 8000, sig.astype(np.float32))
        
        # Rhythms: KA, PA, TA (repetitive syllables, simulated as shorter bursts)
        rhythms = {
            "rhythmKA": np.random.uniform(150, 250),
            "rhythmPA": np.random.uniform(140, 230),
            "rhythmTA": np.random.uniform(160, 260),
        }
        
        for rhythm, freq in rhythms.items():
            # Create 5 repetitions of syllables (bursts)
            sig = np.array([])
            for _ in range(5):
                burst = generate_tone(freq, duration=0.3, noise_level=0.005)
                sig = np.concatenate([sig, burst])
                # Add silence between bursts
                sig = np.concatenate([sig, np.zeros(int(8000 * 0.2))])
            wav.write(base_dir / rhythm / f"{subject_id}_{rhythm}.wav", 8000, sig.astype(np.float32))

    # Patients (ALS) - More noise, jitter (simulated by varying freq/amp lightly or adding more noise)
    for i in range(1, n_patients + 1):
        subject_id = f"PZ{i:03d}"
        
        # Vowels: A, E, I, O, U with higher noise and tremolo
        vowels = {
            "phonationA": np.random.uniform(100, 250),
            "phonationE": np.random.uniform(120, 270),
            "phonationI": np.random.uniform(140, 300),
            "phonationO": np.random.uniform(90, 200),
            "phonationU": np.random.uniform(80, 180),
        }
        
        for vowel, freq in vowels.items():
            sig = generate_tone(freq, duration=3.0, noise_level=0.05)
            # Add tremolo (amplitude modulation) - 5Hz tremor
            t = np.linspace(0, 3.0, len(sig))
            mod = 1.0 + 0.2 * np.sin(2 * np.pi * 5 * t)
            sig = sig * mod
            wav.write(base_dir / vowel / f"{subject_id}_{vowel}.wav", 8000, sig.astype(np.float32))
        
        # Rhythms: KA, PA, TA (repetitive syllables with dysarthria effects)
        rhythms = {
            "rhythmKA": np.random.uniform(150, 250),
            "rhythmPA": np.random.uniform(140, 230),
            "rhythmTA": np.random.uniform(160, 260),
        }
        
        for rhythm, freq in rhythms.items():
            # Create 5 repetitions with irregular timing (dysarthria simulation)
            sig = np.array([])
            for j in range(5):
                # Vary burst duration slightly for irregularity
                burst_dur = 0.3 + np.random.uniform(-0.05, 0.05)
                burst = generate_tone(freq, duration=burst_dur, noise_level=0.05)
                # Add tremolo to each burst
                t = np.linspace(0, burst_dur, len(burst))
                mod = 1.0 + 0.2 * np.sin(2 * np.pi * 5 * t)
                burst = burst * mod
                sig = np.concatenate([sig, burst])
                # Add irregular silence between bursts
                silence_dur = 0.2 + np.random.uniform(-0.05, 0.05)
                sig = np.concatenate([sig, np.zeros(int(8000 * silence_dur))])
            wav.write(base_dir / rhythm / f"{subject_id}_{rhythm}.wav", 8000, sig.astype(np.float32))

    print("Done. Created synthetic wav files.")

if __name__ == "__main__":
    main()
