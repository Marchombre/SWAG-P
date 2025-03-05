import numpy as np
from scipy.special import erf
import numpy.fft as fft




### Fonctions pour le calcul de la permittivité par les modèles de dispersion

## faddeeva implémente une approximation numérique en utilisant une méthode 
# spectrale basée sur une transformée de Fourier rapide (FFT).

def faddeeva(z, N):
    """Calcul approximatif de la fonction de Faddeeva."""
    w = np.zeros(z.size, dtype=complex)
    idx = np.real(z) == 0
    w[idx] = np.exp(-np.abs(z[idx]**2)) * (1 - erf(np.imag(z[idx])))
    idx = np.invert(idx)
    idx1 = idx + (np.imag(z) < 0)
    z[idx1] = np.conj(z[idx1])
    M = 2 * N
    M2 = 2 * M
    k = np.arange(-M + 1, M)
    L = np.sqrt(N / np.sqrt(2))
    theta = k * np.pi / M
    t = L * np.tan(theta / 2)
    f = np.exp(-t**2) * (L**2 + t**2)
    f = np.append(0, f)
    a = np.real(np.fft.fft(np.fft.fftshift(f))) / M2
    a = np.flipud(a[1:N+1])
    Z = (L + 1.0j * z[idx]) / (L - 1.0j * z[idx])
    p = np.polyval(a, Z)
    w[idx] = 2 * p / (L - 1.0j * z[idx])**2 + (1 / np.sqrt(np.pi)) / (L - 1.0j * z[idx])
    w[idx1] = np.conj(2 * np.exp(-z[idx1]**2) - w[idx1])
    return w