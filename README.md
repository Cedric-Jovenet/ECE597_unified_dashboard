---
title: ECE597 Unified Dashboard
emoji: 🎵
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
---

# ECE597 Unified Dashboard

A unified dashboard for audio research algorithms, featuring LMS, Spectral, and Subspace methods for noise reduction.

## Features

- Interactive audio processing with multiple algorithms
- Real-time evaluation metrics (PESQ, STOI)
- Sample audio datasets from NOIZEUS
- Streamlit-based web interface

## Usage

Upload your audio files or use the provided samples to test different noise reduction algorithms.

## Requirements

- Python 3.8+
- Streamlit
- NumPy, SciPy, Librosa, Matplotlib, Soundfile, PESQ, PySTOI

Install dependencies with:
```
pip install -r requirements.txt
```

Run locally with:
```
streamlit run app.py
```

## Datasets

This project uses the NOIZEUS dataset for evaluation, containing various noise types at different SNR levels.

## License

[Add license information here]