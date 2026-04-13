# NOIZEUS Dataset Organization

This folder contains the NOIZEUS speech enhancement dataset, properly organized for use in your audio processing program.

## Structure

- **clean/**: 30 clean speech files (sp01.wav to sp30.wav)
- **{noise_type}_{snr}dB/**: Noisy speech files with specific Signal-to-Noise Ratios
  - Noise types: airport, babble, car, exhibition, restaurant, station, street, train
  - SNR levels: 0dB, 5dB, 10dB, 15dB
  - Each folder contains 30 files: sp01_{noise}_sn{snr}.wav to sp30_{noise}_sn{snr}.wav

## Usage in Python

```python
import os
import librosa

# Path to dataset
dataset_path = "assets/NOIZEUS"

# Load clean speech
clean_audio, sr = librosa.load(os.path.join(dataset_path, "clean", "sp01.wav"), sr=None)

# Load corresponding noisy speech (e.g., airport noise at 0dB SNR)
noisy_audio, _ = librosa.load(os.path.join(dataset_path, "airport_0dB", "sp01_airport_sn0.wav"), sr=sr)

# Your processing algorithm here...
```

## File Naming Convention

- Clean files: `sp{XX}.wav` (XX = 01-30)
- Noisy files: `sp{XX}_{noise_type}_sn{SNR}.wav`

## Total Dataset Size

- 31 folders × 30 files = 930 audio files
- All files are 16-bit WAV format at various sample rates