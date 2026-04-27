import numpy as np
import librosa
from scipy.linalg import toeplitz, svd

def subspace_enhance(noisy_signal, p=10, k=5):
    # Assurer que le signal est assez long pour la matrice de Toeplitz
    if len(noisy_signal) < p:
        return noisy_signal.copy()

    first_col = noisy_signal[:p]
    first_row = noisy_signal[p-1:]
    H = toeplitz(first_col, first_row)

    U, S, Vh = svd(H, full_matrices=False)

    S_clean = np.zeros_like(S)
    S_clean[:k] = S[:k]

    H_clean = U @ np.diag(S_clean) @ Vh
    enhanced_signal = np.concatenate([H_clean[:, 0], H_clean[-1, 1:]])

    # Ajuster la longueur pour correspondre au signal d'entrée
    if len(enhanced_signal) > len(noisy_signal):
        enhanced_signal = enhanced_signal[:len(noisy_signal)]
    elif len(enhanced_signal) < len(noisy_signal):
        # Padding avec les dernières valeurs
        padding = np.full(len(noisy_signal) - len(enhanced_signal), enhanced_signal[-1])
        enhanced_signal = np.concatenate([enhanced_signal, padding])

    return enhanced_signal


def _diagonal_average(matrix):
    output_length = matrix.shape[0] + matrix.shape[1] - 1
    signal = np.zeros(output_length, dtype=np.float64)
    counts = np.zeros(output_length, dtype=np.float64)

    for row_index in range(matrix.shape[0]):
        signal[row_index:row_index + matrix.shape[1]] += matrix[row_index, :]
        counts[row_index:row_index + matrix.shape[1]] += 1.0

    return signal / np.maximum(counts, 1.0)


def signal_subspace_enhance_evt(noisy_signal, frame_length=256, hop_length=128, subspace_order=32, noise_frames=6, gain_floor=0.05):
    if len(noisy_signal) < frame_length:
        return noisy_signal.copy()

    framed = librosa.util.frame(noisy_signal, frame_length=frame_length, hop_length=hop_length).astype(np.float64)
    window = np.hanning(frame_length)
    noise_variance = np.var((framed[:, :min(noise_frames, framed.shape[1])] * window[:, np.newaxis]).ravel())
    embedding_order = min(subspace_order, frame_length // 2)

    enhanced_signal = np.zeros(len(noisy_signal), dtype=np.float64)
    normalization = np.zeros(len(noisy_signal), dtype=np.float64)

    for frame_index in range(framed.shape[1]):
        frame = framed[:, frame_index] * window
        trajectory = librosa.util.frame(frame, frame_length=embedding_order, hop_length=1).astype(np.float64)
        covariance = (trajectory @ trajectory.T) / max(trajectory.shape[1], 1)
        left_vectors, singular_values, _ = svd(covariance, full_matrices=False)
        gains = np.clip(1.0 - noise_variance / np.maximum(singular_values, 1e-10), gain_floor, 1.0)
        projection = left_vectors @ np.diag(gains) @ left_vectors.T
        enhanced_trajectory = projection @ trajectory
        enhanced_frame = _diagonal_average(enhanced_trajectory)
        input_rms = np.sqrt(np.mean(frame ** 2))
        enhanced_rms = np.sqrt(np.mean(enhanced_frame ** 2))
        if enhanced_rms > 1e-10:
            enhanced_frame *= min(1.0, input_rms / enhanced_rms)

        start = frame_index * hop_length
        end = start + frame_length
        enhanced_signal[start:end] += enhanced_frame
        normalization[start:end] += 1.0

    valid = normalization > 1e-10
    enhanced_signal[valid] /= normalization[valid]
    enhanced_signal[~valid] = noisy_signal[~valid]
    return enhanced_signal[:len(noisy_signal)]
