import librosa
import numpy as np
from scipy.special import exp1, i0, i1

from src.algorithms.noise_estimation import (
    imcra_noise_estimate,
    mcra_noise_estimate,
    minimum_statistics_noise_estimate,
    stationary_noise_power_estimate,
)


EPSILON = 1e-10


def _spectral_components(y, n_fft, hop_length):
    stft_matrix = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft_matrix)
    power = np.maximum(magnitude ** 2, EPSILON)
    return stft_matrix, magnitude, power


def _estimate_noise_power(power_spectrogram, method, noise_frames=30, return_spp=False):
    if method == "stationary":
        noise_power = stationary_noise_power_estimate(power_spectrogram, noise_frames=noise_frames)
        if return_spp:
            ratio = power_spectrogram / np.maximum(noise_power, EPSILON)
            speech_presence = np.clip((ratio - 1.0) / 2.0, 0.0, 1.0)
            return noise_power, speech_presence
        return noise_power
    if method == "minimum_statistics":
        noise_power = minimum_statistics_noise_estimate(power_spectrogram)
        if return_spp:
            ratio = power_spectrogram / np.maximum(noise_power, EPSILON)
            speech_presence = np.clip((ratio - 1.0) / 2.0, 0.0, 1.0)
            return noise_power, speech_presence
        return noise_power
    if method == "mcra":
        return mcra_noise_estimate(power_spectrogram, return_spp=return_spp)
    if method == "imcra":
        return imcra_noise_estimate(power_spectrogram, return_spp=return_spp)
    raise ValueError(f"Unsupported noise estimation method: {method}")


def _decision_directed_prior_snr(gamma_frame, previous_gain, previous_gamma, alpha, min_prior_snr):
    if previous_gain is None or previous_gamma is None:
        return np.maximum(gamma_frame - 1.0, min_prior_snr)

    prior_snr = (
        alpha * (previous_gain ** 2) * previous_gamma
        + (1.0 - alpha) * np.maximum(gamma_frame - 1.0, 0.0)
    )
    return np.maximum(prior_snr, min_prior_snr)


def _gain_loop(power_spectrogram, noise_power, gain_function, alpha=0.98, min_prior_snr=1e-3):
    posteriori_snr = power_spectrogram / np.maximum(noise_power, EPSILON)
    gain = np.zeros_like(power_spectrogram, dtype=np.float64)
    previous_gain = None
    previous_gamma = None

    for frame_index in range(power_spectrogram.shape[1]):
        gamma_frame = posteriori_snr[:, frame_index]
        priori_snr = _decision_directed_prior_snr(
            gamma_frame,
            previous_gain,
            previous_gamma,
            alpha,
            min_prior_snr,
        )
        gain[:, frame_index] = gain_function(priori_snr, gamma_frame)
        gain[:, frame_index] = np.clip(gain[:, frame_index], 0.0, 1.0)
        previous_gain = gain[:, frame_index]
        previous_gamma = gamma_frame

    return gain


def decision_directed_wiener_filter(
    y,
    sr,
    n_fft=2048,
    hop_length=512,
    noise_method="minimum_statistics",
    alpha=0.98,
    min_prior_snr=1e-3,
):
    stft_matrix, _, power = _spectral_components(y, n_fft, hop_length)
    noise_power = _estimate_noise_power(power, noise_method)

    def wiener_gain(priori_snr, gamma_frame):
        return priori_snr / (1.0 + priori_snr)

    gain = _gain_loop(power, noise_power, wiener_gain, alpha=alpha, min_prior_snr=min_prior_snr)
    enhanced = gain * stft_matrix
    return librosa.istft(enhanced, hop_length=hop_length, length=len(y))


def mmse_stsa_enhance(
    y,
    sr,
    n_fft=2048,
    hop_length=512,
    noise_method="minimum_statistics",
    alpha=0.98,
    min_prior_snr=1e-3,
):
    stft_matrix, _, power = _spectral_components(y, n_fft, hop_length)
    noise_power = _estimate_noise_power(power, noise_method)

    def mmse_gain(priori_snr, gamma_frame):
        nu = (priori_snr * gamma_frame) / np.maximum(1.0 + priori_snr, EPSILON)
        nu_half = np.minimum(0.5 * nu, 20.0)
        gain = (
            (np.sqrt(np.pi) / 2.0)
            * np.sqrt(np.maximum(nu, EPSILON))
            / np.maximum(gamma_frame, EPSILON)
            * np.exp(-nu_half)
            * ((1.0 + nu) * i0(nu_half) + nu * i1(nu_half))
        )
        high_snr = nu > 5.0
        gain[high_snr] = np.sqrt(priori_snr[high_snr] / np.maximum(1.0 + priori_snr[high_snr], EPSILON))
        return gain

    gain = _gain_loop(power, noise_power, mmse_gain, alpha=alpha, min_prior_snr=min_prior_snr)
    enhanced = gain * stft_matrix
    return librosa.istft(enhanced, hop_length=hop_length, length=len(y))


def log_mmse_enhance(
    y,
    sr,
    n_fft=2048,
    hop_length=512,
    noise_method="minimum_statistics",
    alpha=0.98,
    min_prior_snr=1e-3,
):
    stft_matrix, _, power = _spectral_components(y, n_fft, hop_length)
    noise_power = _estimate_noise_power(power, noise_method)

    def log_mmse_gain(priori_snr, gamma_frame):
        v = np.clip((priori_snr * gamma_frame) / np.maximum(1.0 + priori_snr, EPSILON), EPSILON, 100.0)
        gain = (priori_snr / np.maximum(1.0 + priori_snr, EPSILON)) * np.exp(0.5 * exp1(v))
        return np.minimum(gain, 1.0)

    gain = _gain_loop(power, noise_power, log_mmse_gain, alpha=alpha, min_prior_snr=min_prior_snr)
    enhanced = gain * stft_matrix
    return librosa.istft(enhanced, hop_length=hop_length, length=len(y))


def om_lsa_enhance(
    y,
    sr,
    n_fft=2048,
    hop_length=512,
    noise_method="imcra",
    alpha=0.98,
    min_prior_snr=1e-3,
    gain_floor=0.05,
):
    stft_matrix, _, power = _spectral_components(y, n_fft, hop_length)
    noise_power, speech_presence = _estimate_noise_power(power, noise_method, return_spp=True)
    posteriori_snr = power / np.maximum(noise_power, EPSILON)
    gain = np.zeros_like(power, dtype=np.float64)
    previous_gain = None
    previous_gamma = None

    for frame_index in range(power.shape[1]):
        gamma_frame = posteriori_snr[:, frame_index]
        priori_snr = _decision_directed_prior_snr(
            gamma_frame,
            previous_gain,
            previous_gamma,
            alpha,
            min_prior_snr,
        )
        v = np.clip((priori_snr * gamma_frame) / np.maximum(1.0 + priori_snr, EPSILON), EPSILON, 100.0)
        log_mmse_gain = (priori_snr / np.maximum(1.0 + priori_snr, EPSILON)) * np.exp(0.5 * exp1(v))
        speech_presence_frame = np.clip(speech_presence[:, frame_index], 0.0, 1.0)
        gain[:, frame_index] = np.power(np.maximum(log_mmse_gain, gain_floor), speech_presence_frame) * np.power(
            gain_floor,
            1.0 - speech_presence_frame,
        )
        gain[:, frame_index] = np.clip(gain[:, frame_index], gain_floor, 1.0)
        previous_gain = gain[:, frame_index]
        previous_gamma = gamma_frame

    enhanced = gain * stft_matrix
    return librosa.istft(enhanced, hop_length=hop_length, length=len(y))