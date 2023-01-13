import numpy as np
from skimage.morphology import opening


def spectral_subtraction(
    signal_spectrum: np.ndarray,
    reference_spectrum: np.ndarray,
    n_split: int = 1000,
    alpha0: float = 4,
    s: float = 20 / 3,
    beta: float = 2e-3,
) -> np.ndarray:
    signal_bins, reference_bins = np.array_split(
        signal_spectrum, n_split
    ), np.array_split(reference_spectrum, n_split)
    SNRs = [calculate_SNR(p, n) for p, n in zip(signal_bins, reference_bins)]
    alphas = _oversubtraction_factor(SNRs, alpha0, s)
    alpha_array = np.repeat(alphas, [len(p) for p in reference_bins])
    subtracted = signal_spectrum - alpha_array * reference_spectrum
    clipped = np.clip(subtracted, beta * signal_spectrum, np.inf)
    return clipped


def calculate_SNR(
    signal_power_spectrum: np.ndarray, noise_power_spectrum: np.ndarray
) -> np.ndarray:
    ratio = np.sum(signal_power_spectrum, axis=-1) / np.sum(
        noise_power_spectrum, axis=-1
    )
    return 10 * np.log10(ratio)


def _oversubtraction_factor(SNR: np.ndarray | float, alpha0=4, s=20 / 3) -> np.ndarray:
    SNR = np.array(SNR)
    alpha = alpha0 - np.clip(SNR, -5, 20) / s
    return alpha


def baseline_morphological(
    power_spectrum: np.ndarray, tol: float = 1e-9, max_iter: int = 200
) -> np.ndarray:
    baseline = np.array(power_spectrum)
    for i in range(max_iter):
        opened = opening(baseline, footprint=np.ones(i + 2))
        error = np.sum(baseline) - np.sum(opened)
        if error <= tol:
            break
        baseline = opened
    else:
        print(
            "Max iteration reached, may need to try again with a larger max_iter value or larger tol value"
        )
    return baseline


def baseline_rayleigh(power_spectrum, tol=0.999995, max_iter=10000) -> np.ndarray:

    power = np.array(power_spectrum)
    b = np.sum(power) / (2 * len(power))
    for _ in range(max_iter):
        threshold = 2 * b * (-np.log10(tol))
        power[power > threshold] = 0
        b_old, b = b, np.sum(power) / (2 * len(power))
        if b - b_old <= 1e-2:
            break
    else:
        print(
            "Max iteration reached, may need to try again with a larger max_iter value"
        )
    return threshold
