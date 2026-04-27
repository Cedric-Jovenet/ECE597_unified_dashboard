import numpy as np
import librosa

from src.algorithms.noise_estimation import minimum_statistics_noise_estimate


def wiener_filter_enhance(y, sr, n_fft=2048, hop_length=512, noise_frames_start=0, noise_frames_end=30, floor=0.01):
    D_noisy = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude_noisy = np.abs(D_noisy)
    phase_noisy = np.exp(1j * np.angle(D_noisy))

    noise_power = np.mean(magnitude_noisy[:, noise_frames_start:noise_frames_end] ** 2, axis=1, keepdims=True)
    noisy_power = magnitude_noisy ** 2
    speech_power = np.maximum(noisy_power - noise_power, floor * noisy_power)
    gain = speech_power / (speech_power + noise_power + 1e-12)

    D_denoised = gain * magnitude_noisy * phase_noisy
    return librosa.istft(D_denoised, hop_length=hop_length, length=len(y))


def adaptive_spectral_subtraction(y, sr, n_fft=2048, hop_length=512, beta=0.002, noise_method="minimum_statistics"):
    D_noisy = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude_noisy = np.abs(D_noisy)
    power_noisy = np.maximum(magnitude_noisy ** 2, 1e-10)
    phase_noisy = np.exp(1j * np.angle(D_noisy))
    noise_power = minimum_statistics_noise_estimate(power_noisy) if noise_method == "minimum_statistics" else power_noisy
    noise_magnitude = np.sqrt(np.maximum(noise_power, 1e-10))

    frame_snr_db = 10.0 * np.log10(
        np.mean(power_noisy, axis=0) / np.maximum(np.mean(noise_power, axis=0), 1e-10)
    )
    alpha = np.where(
        frame_snr_db < -5.0,
        5.0,
        np.where(frame_snr_db <= 20.0, 4.0 - 3.0 * frame_snr_db / 20.0, 1.0),
    )
    magnitude_denoised = np.maximum(magnitude_noisy - alpha[np.newaxis, :] * noise_magnitude, beta * magnitude_noisy)
    D_denoised = magnitude_denoised * phase_noisy
    return librosa.istft(D_denoised, hop_length=hop_length, length=len(y))


def power_spectral_subtraction(y, sr, n_fft=2048, hop_length=512, alpha=1.0, beta=0.01):
    D_noisy = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude_noisy = np.abs(D_noisy)
    power_noisy = np.maximum(magnitude_noisy ** 2, 1e-10)
    phase_noisy = np.exp(1j * np.angle(D_noisy))
    noise_power = minimum_statistics_noise_estimate(power_noisy)
    clean_power = np.maximum(power_noisy - alpha * noise_power, beta * power_noisy)
    magnitude_denoised = np.sqrt(clean_power)
    D_denoised = magnitude_denoised * phase_noisy
    return librosa.istft(D_denoised, hop_length=hop_length, length=len(y))


def generalized_spectral_subtraction(y, sr, n_fft=2048, hop_length=512, alpha=1.0, beta=0.01, domain_exponent=1.5):
    D_noisy = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude_noisy = np.abs(D_noisy)
    phase_noisy = np.exp(1j * np.angle(D_noisy))
    power_noisy = np.maximum(magnitude_noisy ** 2, 1e-10)
    noise_power = minimum_statistics_noise_estimate(power_noisy)
    noise_magnitude = np.sqrt(np.maximum(noise_power, 1e-10))

    transformed_noisy = magnitude_noisy ** domain_exponent
    transformed_noise = noise_magnitude ** domain_exponent
    transformed_clean = np.maximum(transformed_noisy - alpha * transformed_noise, beta * transformed_noisy)
    magnitude_denoised = transformed_clean ** (1.0 / domain_exponent)
    D_denoised = magnitude_denoised * phase_noisy
    return librosa.istft(D_denoised, hop_length=hop_length, length=len(y))

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
    band_alpha_scales = np.linspace(0.9, 1.3, n_bands)
    band_beta_scales = np.linspace(1.0, 1.5, n_bands)

    for i in range(n_bands):
        start_bin = band_edges[i]
        end_bin = band_edges[i+1]
        band_magnitude_noisy = magnitude_noisy[start_bin:end_bin, :]
        band_noise_magnitude = noise_magnitude_spectrum_full[start_bin:end_bin, :]
        band_alpha = alpha * band_alpha_scales[i]
        band_beta = beta * band_beta_scales[i]
        band_magnitude_denoised = np.maximum(band_magnitude_noisy - band_alpha * band_noise_magnitude, band_beta * band_magnitude_noisy)
        magnitude_denoised[start_bin:end_bin, :] = band_magnitude_denoised

    D_denoised = magnitude_denoised * phase_noisy
    return librosa.istft(D_denoised, hop_length=hop_length, length=len(y))
