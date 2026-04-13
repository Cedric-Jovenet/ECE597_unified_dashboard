import numpy as np
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
