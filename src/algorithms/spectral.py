import numpy as np
import librosa

def spectral_subtraction(y, sr, n_fft=2048, hop_length=512, noise_frames_start=0, noise_frames_end=30):
    D_noisy = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude_noisy = np.abs(D_noisy)
    phase_noisy = np.exp(1j * np.angle(D_noisy))
    noise_magnitude_spectrum = np.mean(magnitude_noisy[:, noise_frames_start:noise_frames_end], axis=1, keepdims=True)
    magnitude_denoised = np.maximum(magnitude_noisy - noise_magnitude_spectrum, 0)
    D_denoised = magnitude_denoised * phase_noisy
    return librosa.istft(D_denoised, hop_length=hop_length, length=len(y))

def spectral_subtraction_oversubtraction(y, sr, n_fft=2048, hop_length=512, noise_frames_start=0, noise_frames_end=30, snr_db=None):
    D_noisy = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude_noisy = np.abs(D_noisy)
    phase_noisy = np.exp(1j * np.angle(D_noisy))
    noise_magnitude_spectrum = np.mean(magnitude_noisy[:, noise_frames_start:noise_frames_end], axis=1, keepdims=True)

    if snr_db is not None:
        alpha = np.maximum(1.0, 4 - snr_db / 5)
        beta = np.maximum(0.001, 0.005 - snr_db / 100)
    else:
        alpha = 1.2
        beta = 0.002

    magnitude_denoised = np.maximum(magnitude_noisy - alpha * noise_magnitude_spectrum, beta * magnitude_noisy)
    D_denoised = magnitude_denoised * phase_noisy
    return librosa.istft(D_denoised, hop_length=hop_length, length=len(y))

def multiband_spectral_subtraction(y, sr, n_fft=2048, hop_length=512, noise_frames_start=0, noise_frames_end=30, alpha=1.2, beta=0.002, n_bands=5):
    D_noisy = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude_noisy = np.abs(D_noisy)
    phase_noisy = np.exp(1j * np.angle(D_noisy))
    noise_magnitude_spectrum_full = np.mean(magnitude_noisy[:, noise_frames_start:noise_frames_end], axis=1, keepdims=True)
    magnitude_denoised = np.zeros_like(magnitude_noisy)

    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band_edges = np.linspace(0, len(freq_bins) - 1, n_bands + 1).astype(int)

    for i in range(n_bands):
        start_bin = band_edges[i]
        end_bin = band_edges[i+1]
        band_magnitude_noisy = magnitude_noisy[start_bin:end_bin, :]
        band_noise_magnitude = noise_magnitude_spectrum_full[start_bin:end_bin, :]
        band_magnitude_denoised = np.maximum(band_magnitude_noisy - alpha * band_noise_magnitude, beta * band_magnitude_noisy)
        magnitude_denoised[start_bin:end_bin, :] = band_magnitude_denoised

    D_denoised = magnitude_denoised * phase_noisy
    return librosa.istft(D_denoised, hop_length=hop_length, length=len(y))
