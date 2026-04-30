---
title: Unified Speech Enhancement Dashboard
emoji: 🎵
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
---

# Unified Speech Enhancement Dashboard

A unified dashboard for classical speech enhancement research, centered on interpretable algorithms, the NOIZEUS dataset, and reproducible benchmarking.

## Features

- Interactive single-sample listening and visualization workspace
- Classical implemented families: spectral subtraction, Wiener filtering, statistical model-based enhancement, and subspace enhancement
- Reusable noise-estimation backends: stationary, minimum statistics, MCRA, and IMCRA
- Reproducible benchmark workflow for NOIZEUS
- Literature-oriented dataset, metric, and protocol notes in the same interface

## Usage

Use the NOIZEUS samples shipped with the project to inspect algorithm behavior on individual utterances, then move to the benchmark page to run comparable evaluations.

## Requirements

- Python 3.10+
- Streamlit
- NumPy, SciPy, Librosa, Matplotlib, Soundfile, PESQ, PySTOI

Install dependencies with:
```
pip install -r requirements.txt
```

Run locally with:
```
streamlit run app.py

(Start the app from the parent folder of app.py (\research_audio_project))
```

Run the offline benchmark helper with:
```bash
python -m src.evaluation.noizeus_benchmark --noise babble --output-dir artifacts
```

This command writes raw and summary CSV files that can be reused in the written report.
Use `--snr 5dB` only when you want to drill down to a single noise-condition slice.

## Datasets

This project uses the NOIZEUS dataset for the runnable benchmark and tracks VoiceBank+DEMAND as a reference protocol in the literature section.

## License

[Add license information here]