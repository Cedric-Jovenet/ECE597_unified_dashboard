from collections import deque

import numpy as np


def stationary_noise_power_estimate(power_spectrogram, noise_frames=30, floor=1e-10):
    frame_count = min(noise_frames, power_spectrogram.shape[1])
    if frame_count <= 0:
        return np.full_like(power_spectrogram, floor)

    noise_profile = np.mean(power_spectrogram[:, :frame_count], axis=1, keepdims=True)
    noise_profile = np.maximum(noise_profile, floor)
    return np.repeat(noise_profile, power_spectrogram.shape[1], axis=1)


def _smooth_power(power_spectrogram, smoothing):
    smoothed = np.zeros_like(power_spectrogram, dtype=np.float64)
    smoothed[:, 0] = power_spectrogram[:, 0]
    for frame_index in range(1, power_spectrogram.shape[1]):
        smoothed[:, frame_index] = (
            smoothing * smoothed[:, frame_index - 1]
            + (1.0 - smoothing) * power_spectrogram[:, frame_index]
        )
    return smoothed


def minimum_statistics_noise_estimate(
    power_spectrogram,
    smoothing=0.98,
    min_window=20,
    bias_factor=1.5,
    floor=1e-10,
):
    smoothed = _smooth_power(power_spectrogram, smoothing)
    noise_power = np.zeros_like(smoothed, dtype=np.float64)
    history = deque(maxlen=max(2, min_window))
    base_floor = max(floor, floor * float(np.max(power_spectrogram)))

    for frame_index in range(smoothed.shape[1]):
        current_frame = smoothed[:, frame_index]
        history.append(current_frame.copy())
        running_minimum = np.min(np.stack(history, axis=1), axis=1)
        target = np.minimum(current_frame, bias_factor * running_minimum)

        if frame_index == 0:
            noise_power[:, frame_index] = np.maximum(target, base_floor)
            continue

        noise_power[:, frame_index] = (
            smoothing * noise_power[:, frame_index - 1]
            + (1.0 - smoothing) * target
        )
        noise_power[:, frame_index] = np.maximum(noise_power[:, frame_index], base_floor)

    return noise_power


def mcra_noise_estimate(
    power_spectrogram,
    smoothing=0.9,
    adaptation_rate=0.1,
    threshold=1.8,
    floor=1e-10,
    return_spp=False,
):
    smoothed = _smooth_power(power_spectrogram, smoothing)
    noise_power = np.zeros_like(smoothed, dtype=np.float64)
    speech_presence = np.zeros_like(smoothed, dtype=np.float64)

    initial_noise = stationary_noise_power_estimate(power_spectrogram, noise_frames=min(30, power_spectrogram.shape[1]))
    noise_power[:, 0] = initial_noise[:, 0]
    base_floor = max(floor, floor * float(np.max(power_spectrogram)))

    for frame_index in range(1, smoothed.shape[1]):
        previous_noise = np.maximum(noise_power[:, frame_index - 1], base_floor)
        ratio = smoothed[:, frame_index] / previous_noise
        speech_presence[:, frame_index] = np.clip((ratio - 1.0) / max(threshold - 1.0, 1e-10), 0.0, 1.0)

        update_weight = adaptation_rate * (1.0 - speech_presence[:, frame_index])
        target = np.minimum(smoothed[:, frame_index], threshold * previous_noise)
        noise_power[:, frame_index] = (
            (1.0 - update_weight) * previous_noise
            + update_weight * target
        )
        noise_power[:, frame_index] = np.maximum(noise_power[:, frame_index], base_floor)

    if return_spp:
        return noise_power, speech_presence
    return noise_power


def imcra_noise_estimate(
    power_spectrogram,
    smoothing=0.96,
    min_window=15,
    threshold=2.0,
    floor_factor=0.02,
    floor=1e-10,
    return_spp=False,
):
    smoothed = _smooth_power(power_spectrogram, smoothing)
    noise_power = np.zeros_like(smoothed, dtype=np.float64)
    speech_presence = np.zeros_like(smoothed, dtype=np.float64)
    history = deque(maxlen=max(2, min_window))

    initial_noise = stationary_noise_power_estimate(power_spectrogram, noise_frames=min(30, power_spectrogram.shape[1]))
    noise_power[:, 0] = initial_noise[:, 0]
    global_floor = max(floor, floor_factor * float(np.median(power_spectrogram)))

    for frame_index in range(smoothed.shape[1]):
        current_frame = smoothed[:, frame_index]
        history.append(current_frame.copy())
        local_minimum = np.min(np.stack(history, axis=1), axis=1)

        if frame_index == 0:
            noise_power[:, frame_index] = np.maximum(np.minimum(current_frame, threshold * local_minimum), global_floor)
            continue

        previous_noise = np.maximum(noise_power[:, frame_index - 1], global_floor)
        ratio = current_frame / np.maximum(local_minimum, global_floor)
        speech_presence[:, frame_index] = np.clip((ratio - 1.0) / max(threshold - 1.0, 1e-10), 0.0, 1.0)
        adaptive_smoothing = smoothing + (1.0 - smoothing) * speech_presence[:, frame_index]
        target = np.minimum(current_frame, threshold * local_minimum)
        noise_power[:, frame_index] = (
            adaptive_smoothing * previous_noise
            + (1.0 - adaptive_smoothing) * target
        )
        noise_power[:, frame_index] = np.maximum(noise_power[:, frame_index], global_floor)

    if return_spp:
        return noise_power, speech_presence
    return noise_power