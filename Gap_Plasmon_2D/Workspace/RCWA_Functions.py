### Fonctions RCWA

def cascade(T, U):
    '''Cascade deux matrices de diffusion T et U (dimensions 2n x 2n).'''
    n = int(T.shape[1] / 2)
    J = np.linalg.inv(np.eye(n) - np.matmul(U[0:n, 0:n], T[n:2*n, n:2*n]))
    K = np.linalg.inv(np.eye(n) - np.matmul(T[n:2*n, n:2*n], U[0:n, 0:n]))
    S = np.block([
        [T[0:n, 0:n] + np.matmul(np.matmul(np.matmul(T[0:n, n:2*n], J), U[0:n, 0:n]), T[n:2*n, 0:n]),
         np.matmul(np.matmul(T[0:n, n:2*n], J), U[0:n, n:2*n])],
        [np.matmul(np.matmul(U[n:2*n, 0:n], K), T[n:2*n, 0:n]),
         U[n:2*n, n:2*n] + np.matmul(np.matmul(np.matmul(U[n:2*n, 0:n], K), T[n:2*n, n:2*n]), U[0:n, n:2*n])]
    ])
    return S

def c_bas(A, V, h):
    '''Cascade la matrice de diffusion A avec la propagation d'une couche de hauteur h.'''
    n = int(A.shape[1] / 2)
    D = np.diag(np.exp(1j * V * h))
    S = np.block([
        [A[0:n, 0:n], np.matmul(A[0:n, n:2*n], D)],
        [np.matmul(D, A[n:2*n, 0:n]), np.matmul(np.matmul(D, A[n:2*n, n:2*n]), D)]
    ])
    return S

def c_haut(A, valp, h):
    n = int(A[0].size / 2)
    D = np.diag(np.exp(1j * valp * h))
    S11 = np.dot(D, np.dot(A[0:n, 0:n], D))
    S12 = np.dot(D, A[0:n, n:2*n])
    S21 = np.dot(A[n:2*n, 0:n], D)
    S22 = A[n:2*n, n:2*n]
    S1 = np.append(S11, S12, axis=1)
    S2 = np.append(S21, S22, axis=1)
    S = np.append(S1, S2, axis=0)
    return S    

def intermediaire(T, U):
    n = int(T.shape[0] / 2)
    H = np.linalg.inv(np.eye(n) - np.matmul(U[0:n, 0:n], T[n:2*n, n:2*n]))
    K = np.linalg.inv(np.eye(n) - np.matmul(T[n:2*n, n:2*n], U[0:n, 0:n]))
    a = np.matmul(K, T[n:2*n, 0:n])
    b = np.matmul(K, np.matmul(T[n:2*n, n:2*n], U[0:n, n:2*n]))
    c = np.matmul(H, np.matmul(U[0:n, 0:n], T[n:2*n, 0:n]))
    d = np.matmul(H, U[0:n, n:2*n])
    S = np.block([[a, b], [c, d]])
    return S

def couche(valp, h):
    n = len(valp)
    AA = np.diag(np.exp(1j * valp * h))
    C = np.block([[np.zeros((n, n)), AA], [AA, np.zeros((n, n))]])
    return C

def step(a, b, w, x0, n):
    '''Calcule la matrice de Toeplitz générée par la série de Fourier d'une fonction en escalier.'''
    tmp = np.exp(-2 * 1j * np.pi * (x0 + w / 2) * np.arange(0, n)) * np.sinc(w * np.arange(0, n)) * w
    l = np.conj(tmp) * (b - a)
    m = tmp * (b - a)
    l[0] = l[0] + a
    m[0] = l[0]
    T = toeplitz(l, m)
    return T

def grating(k0, a0, pol, e1, e2, n, blocs):
    '''Génère la matrice de Fourier pour un réseau constitué de blocs de matériau e2 dans un milieu e1.'''
    n_blocs = blocs.shape[0]
    nmod = int(n / 2)
    M1 = e1 * np.eye(n, dtype=complex)
    M2 = 1 / e1 * np.eye(n, dtype=complex)
    for k in range(n_blocs):
        M1 = M1 + step(0, e2 - e1, blocs[k, 0], blocs[k, 1], n)
        M2 = M2 + step(0, 1 / e2 - 1 / e1, blocs[k, 0], blocs[k, 1], n)
    alpha = np.diag(a0 + 2 * np.pi * np.arange(-nmod, nmod + 1)) + 0j
    if pol == 0:
        M = alpha * alpha - k0**2 * M1
        L, E = np.linalg.eig(M)
        L = np.sqrt(-L + 0j)
        L = (1 - 2 * (np.imag(L) < -1e-15)) * L
        P = np.block([[E], [np.matmul(E, np.diag(L))]])
    else:
        T_inv = np.linalg.inv(M2)
        M = np.matmul(np.matmul(np.matmul(T_inv, alpha), np.linalg.inv(M1)), alpha) - k0**2 * T_inv
        L, E = np.linalg.eig(M)
        L = np.sqrt(-L + 0j)
        L = (1 - 2 * (np.imag(L) < -1e-15)) * L
        P = np.block([[E], [np.matmul(np.matmul(M2, E), np.diag(L))]])
    return P, L

def homogene(k0, a0, pol, epsilon, n):
    nmod = int(n / 2)
    valp = np.sqrt(epsilon * k0**2 - (a0 + 2 * np.pi * np.arange(-nmod, nmod + 1))**2 + 0j)
    valp = valp * (1 - 2 * (valp < 0))
    P = np.block([[np.eye(n, dtype=complex)], [np.diag(valp * (pol / epsilon + (1 - pol)))]])
    return P, valp

def interface(P, Q):
    '''Calcule la matrice de diffusion d'une interface à partir de P et Q.'''
    n = int(P.shape[1])
    A = np.block([[P[0:n, 0:n], -Q[0:n, 0:n]],
                  [P[n:2*n, 0:n],  Q[n:2*n, 0:n]]])
    B = np.block([[-P[0:n, 0:n], Q[0:n, 0:n]],
                  [ P[n:2*n, 0:n], Q[n:2*n, 0:n]]])
    S = np.matmul(np.linalg.inv(A), B)
    return S

def HErmes(T, U, V, P, Amp, ny, h, a0):
    n = int(np.shape(T)[0] / 2)
    nmod = int((n - 1) / 2)
    nx = n
    X = np.matmul(intermediaire(T, cascade(couche(V, h), U)), Amp.reshape(Amp.size, 1))
    D = X[0:n]
    X = np.matmul(intermediaire(cascade(T, couche(V, h)), U), Amp.reshape(Amp.size, 1))
    E = X[n:2*n]
    M = np.zeros((ny, nx - 1), dtype=complex)
    for k in range(ny):
        y = h / ny * (k + 1)
        Fourier = np.matmul(P, np.matmul(np.diag(np.exp(1j * V * y)), D) + np.matmul(np.diag(np.exp(1j * V * (h - y))), E))
        MM = np.fft.ifftshift(Fourier[0:len(Fourier) - 1])
        M[k, :] = MM.reshape(len(MM))
    M = np.conj(np.fft.ifft(np.conj(M).T, axis=0)).T * n
    x, y = np.meshgrid(np.linspace(0, 1, nx - 1), np.linspace(0, 1, ny))
    M = M * np.exp(1j * a0 * x)
    return M