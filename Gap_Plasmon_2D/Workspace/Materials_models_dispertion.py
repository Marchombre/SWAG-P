### Fonctions pour le calcul de la permittivité par les modèles de dispersion
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

def epsAubb(lam):
    """Permet de calculer la permittivité de l'or (Au) via le modèle Brendel-Bormann."""
    w = 6.62606957e-25 * 299792458 / 1.602176565e-19 / lam
    f0 = 0.770; Gamma0 = 0.050; omega_p = 9.03
    f = np.array([0.054, 0.050, 0.312, 0.719, 1.648])
    Gamma = np.array([0.074, 0.035, 0.083, 0.125, 0.179])
    omega = np.array([0.218, 2.885, 4.069, 6.137, 27.97])
    sigma = np.array([0.742, 0.349, 0.830, 1.246, 1.795])
    a = np.sqrt(w * (w + 1.0j * Gamma))
    a = a * np.sign(np.real(a))
    x = (a - omega) / (np.sqrt(2) * sigma)
    y = (a + omega) / (np.sqrt(2) * sigma)
    epsilon = 1 - omega_p**2 * f0 / (w * (w + 1.0j * Gamma0)) \
              + np.sum(1.0j * np.sqrt(np.pi) * f * omega_p**2 / (2 * np.sqrt(2) * a * sigma) * (faddeeva(x, 64) + faddeeva(y, 64)))
    return epsilon

def epsAgbb(lam):
    """Permet de calculer la permittivité de l'argent (Ag) via le modèle Brendel-Bormann."""
    w = 6.62606957e-25 * 299792458 / 1.602176565e-19 / lam
    f0 = 0.821; Gamma0 = 0.049; omega_p = 9.01
    f = np.array([0.050, 0.133, 0.051, 0.467, 4.000])
    Gamma = np.array([0.189, 0.067, 0.019, 0.117, 0.052])
    omega = np.array([2.025, 5.185, 4.343, 9.809, 18.56])
    sigma = np.array([1.894, 0.665, 0.189, 1.170, 0.516])
    a = np.sqrt(w * (w + 1.0j * Gamma))
    a = a * np.sign(np.real(a))
    x = (a - omega) / (np.sqrt(2) * sigma)
    y = (a + omega) / (np.sqrt(2) * sigma)
    aha = 1.0j * np.sqrt(np.pi) * f * omega_p**2 / (2 * np.sqrt(2) * a * sigma) * (faddeeva(x, 64) + faddeeva(y, 64))
    epsilon = 1 - omega_p**2 * f0 / (w * (w + 1.0j * Gamma0)) + np.sum(aha)
    return epsilon

def epsCrbb(lam):
    """Permet de calculer la permittivité du chrome (Cr) via le modèle Brendel-Bormann."""
    w = 6.62606957e-25 * 299792458 / 1.602176565e-19 / lam
    f0 = 0.154; Gamma0 = 0.048; omega_p = 10.75
    f = np.array([0.338, 0.261, 0.817, 0.105])
    Gamma = np.array([4.256, 3.957, 2.218, 6.983])
    omega = np.array([0.281, 0.584, 1.919, 6.997])
    sigma = np.array([0.115, 0.252, 0.225, 4.903])
    a = np.sqrt(w * (w + 1.0j * Gamma))
    a = a * np.sign(np.real(a))
    x = (a - omega) / (np.sqrt(2) * sigma)
    y = (a + omega) / (np.sqrt(2) * sigma)
    aha = 1.0j * np.sqrt(np.pi) * f * omega_p**2 / (2 * np.sqrt(2) * a * sigma) * (faddeeva(x, 64) + faddeeva(y, 64))
    epsilon = 1 - omega_p**2 * f0 / (w * (w + 1.0j * Gamma0)) + np.sum(aha)
    return epsilon

def epsAlbb(lam):
    """Permet de calculer la permittivité de l'aluminium (Al) via le modèle Brendel-Bormann."""
    w = 6.62606957e-25 * 299792458 / 1.602176565e-19 / lam
    f0 = 0.526; Gamma0 = 0.047; omega_p = 14.98
    f = np.array([0.213, 0.060, 0.182, 0.014])
    Gamma = np.array([0.312, 0.315, 1.587, 2.145])
    omega = np.array([0.163, 1.561, 1.827, 4.495])
    sigma = np.array([0.013, 0.042, 0.256, 1.735])
    a = np.sqrt(w * (w + 1.0j * Gamma))
    a = a * np.sign(np.real(a))
    x = (a - omega) / (np.sqrt(2) * sigma)
    y = (a + omega) / (np.sqrt(2) * sigma)
    aha = 1.0j * np.sqrt(np.pi) * f * omega_p**2 / (2 * np.sqrt(2) * a * sigma) * (faddeeva(x, 64) + faddeeva(y, 64))
    epsilon = 1 - omega_p**2 * f0 / (w * (w + 1.0j * Gamma0)) + np.sum(aha)
    return epsilon

def epsNibb(lam):
    """Permet de calculer la permittivité du nickel (Ni) via le modèle Brendel-Bormann."""
    w = 6.62606957e-25 * 299792458 / 1.602176565e-19 / lam
    f0 = 0.083; Gamma0 = 0.022; omega_p = 15.92
    f = np.array([0.357, 0.039, 0.127, 0.654])
    Gamma = np.array([2.820, 0.120, 1.822, 6.637])
    omega = np.array([0.317, 1.059, 4.583, 8.825])
    sigma = np.array([0.606, 1.454, 0.379, 0.510])
    a = np.sqrt(w * (w + 1.0j * Gamma))
    a = a * np.sign(np.real(a))
    x = (a - omega) / (np.sqrt(2) * sigma)
    y = (a + omega) / (np.sqrt(2) * sigma)
    aha = 1.0j * np.sqrt(np.pi) * f * omega_p**2 / (2 * np.sqrt(2) * a * sigma) * (faddeeva(x, 64) + faddeeva(y, 64))
    epsilon = 1 - omega_p**2 * f0 / (w * (w + 1.0j * Gamma0)) + np.sum(aha)
    return epsilon