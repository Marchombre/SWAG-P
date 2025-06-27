

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from joblib import Parallel, delayed


def cascade(T,U):
    '''Cascading of two scattering matrices T and U.
    Since T and U are scattering matrices, it is expected that they are square
    and have the same dimensions which are necessarily EVEN.
    '''
    n=int(T.shape[1] / 2)
    J=np.linalg.inv( np.eye(n) - np.matmul(U[0:n,0:n],T[n:2*n,n:2*n] ) )
    K=np.linalg.inv( np.eye(n) - np.matmul(T[n:2*n,n:2*n],U[0:n,0:n] ) )
    S=np.block([[T[0:n,0:n] + np.matmul(np.matmul(np.matmul(T[0:n,n:2*n],J),
    U[0:n,0:n]),T[n:2*n,0:n]),np.matmul(np.matmul(T[0:n,n:2*n],J),U[0:n,n:2*n])
    ],[np.matmul(np.matmul(U[n:2*n,0:n],K),T[n:2*n,0:n]),U[n:2*n,n:2*n]
    + np.matmul(np.matmul(np.matmul(U[n:2*n,0:n],K),T[n:2*n,n:2*n]),U[0:n,n:2*n])
    ]])
    return S

def c_bas(A,V,h):
    ''' Directly cascading any scattering matrix A (square and with even
    dimensions) with the scattering matrix of a layer of thickness h in which
    the wavevectors are given by V. Since the layer matrix is
    essentially empty, the cascading is much quicker if this is taken
    into account.
    '''
    n=int(A.shape[1]/2)
    D=np.diag(np.exp(1j*V*h))
    S=np.block([[A[0:n,0:n],np.matmul(A[0:n,n:2*n],D)],[np.matmul(D,A[n:2*n,0:n]),np.matmul(np.matmul(D,A[n:2*n,n:2*n]),D)]])
    return S

def step(a,b,w,x0,n):
    '''Computes the Fourier series for a piecewise function having the value
    b over a portion w of the period, starting at position x0
    and the value a otherwise. The period is supposed to be equal to 1.
    Then returns the toeplitz matrix generated using the Fourier series.
    '''
    from scipy.linalg import toeplitz
    from numpy import sinc
    l=np.zeros(n,dtype=np.complex128)
    m=np.zeros(n,dtype=np.complex128)
    tmp=np.exp(-2*1j*np.pi*(x0+w/2)*np.arange(0,n))*sinc(w*np.arange(0,n))*w
    l=np.conj(tmp)*(b-a)
    m=tmp*(b-a)
    l[0]=l[0]+a
    m[0]=l[0]
    T=toeplitz(l,m)
    return T

def fpml(q,g,n):
    from scipy.linalg import toeplitz
    from numpy import sinc,flipud
    x=np.arange(-n,n+1)
    v=-q/2*((1+g/4)*sinc(q*x)+(sinc(q*x-1)+sinc(q*x+1))*0.5-g*0.125*(sinc(q*x-2)+sinc(q*x+2)))
    v[n]=v[n]+1
    T=toeplitz(flipud(v[1:n+1]),v[n:2*n])
    return T

def aper(k0,a0,pol,e1,e2,n,blocs):
    '''Warning: blocs is a vector with N lines and 2 columns. Each
    line refers to a block of material e2 inside a matrix of material e1,
    giving its size relatively to the period (first column) and its starting
    position.
    Warning : There is nothing checking that the blocks don't overlapp.
    '''
    n_blocs=blocs.shape[0];
    nmod=int(n/2)
    M1=e1*np.eye(n,n)
    M2=1/e1*np.eye(n,n)
    for k in range(0,n_blocs):
        M1=M1+step(0,e2-e1,blocs[k,0],blocs[k,1],n)
        M2=M2+step(0,1/e2-1/e1,blocs[k,0],blocs[k,1],n)
    alpha=np.diag(a0+2*np.pi*np.arange(-nmod,nmod+1))+0j
    g=1/(1-1j);
    fprime=fpml(0.2001,g,n) #fixe la proporsion de periode qui constitue la PML (à gauche et à droite)
    if (pol==0):
        tmp=np.linalg.inv(fprime)
        M=np.matmul(tmp, np.matmul(alpha, np.matmul(tmp, alpha)))\
        -k0*k0*M1
        L,E=np.linalg.eig(M)
        L=np.sqrt(-L+0j)
        L=(1-2*(np.imag(L)<-1e-15))*L
        P=np.block([[E],[np.matmul(E,np.diag(L))]])
    else:
        M=np.matmul(np.linalg.inv(np.matmul(fprime, M2)),\
        -k0*k0*fprime+np.matmul(alpha, np.matmul(np.linalg.inv(np.matmul(M1, fprime)), alpha)))
        L,E=np.linalg.eig(M)
        L=np.sqrt(-L+0j)
        L=(1-2*(np.imag(L)<-1e-15))*L
        P=np.block([[E],[np.matmul(np.matmul(M2,E),np.diag(L))]])
    return P,L

def interface(P,Q):
    '''Computation of the scattering matrix of an interface, P and Q being the
    matrices given for each layer by homogene, reseau or creneau.
    '''
    n=int(P.shape[1])
    S=np.matmul(np.linalg.inv(np.block([[P[0:n,0:n],-Q[0:n,0:n]],[P[n:2*n,0:n],Q[n:2*n,0:n]]])),np.block([[-P[0:n,0:n],Q[0:n,0:n]],[P[n:2*n,0:n],Q[n:2*n,0:n]]]))
    return S

def taper(X): #verif ok
# X is a (n_layers x 2) matrix. The first column are the thicknesses,
# the second column are the widths of the waveguides. The positions of the waveguides are imposed.
# Number of modes retained (odd number); 31 is suffisent
    n=201
    d=3000
    n_layers=int(X.shape[0])
    n_cubes = 1
    lam=1550
    pol=0
    a0=0
    e1=2.1316
    e2=12.11
    # Definition des variables
    # Adimensionalization
    x=X/d
    t = X[:,0]
    t = t/d
    w = X[:,1]
    w = w/d;
    p = 0.5 - w/2
    w_in=120/d
    w_out=450/d
    l=lam/d
    k0=2*np.pi/l
    bloc_in=np.array([[w_in,0.5-w_in/2]])
    bloc_out=np.array([[w_out,0.5-w_out/2]])
    # Starting with neutral S matrix
    S=np.block([[np.zeros([n,n]),np.eye(n,dtype=np.complex128)],[np.eye(n),np.zeros([n,n])]])
    P,V=aper(k0,a0,pol,e1,e2,n,bloc_in)
    a=np.argmin(abs(V-2.35*k0)) #guide entrée, 2.35 = indice effectif
    for k in range(0,n_layers):
        cubes = np.array([[w[k],p[k]]])
        P_new,V_new=aper(k0,a0,pol,e1,e2,n,cubes)
        S=cascade(S,interface(P,P_new))
        S=c_bas(S,V_new,t[k])
        P,V=P_new,V_new
    P_out,V_out=aper(k0,a0,pol,e1,e2,n,bloc_out)
    S=cascade(S,interface(P,P_out))
    b=np.argmin(abs(V_out-3.24033*k0)) # guide sortie, 3.24 = indice effectif
    cost=1-abs(S[b+n,a])**2
    return cost

def DE_general(budget, Npop, Nlayers, Ncubes, bornes, periode, Nopt):
    """
    Algorithme Differential Evolution “variant DE/rand-to-best/1/bin ”.
    Arguments :
      - budget : nombre total d’évaluations de la fonction de coût
      - Npop   : taille de la population (nombre d’individus)
      - Nlayers: nombre de couches à optimiser
      - Ncubes : nombre de « cubes » (paramètres horizontaux) par couche
      - bornes : tuple/list [t_min, t_max, w_min, w_max] 
                 bornes pour l’épaisseur (dim 0) et la largeur (dim 1)
      - periode: période (non utilisée dans cette version)
      - Nopt   : identifiant de l’optimisation, pour nommer les fichiers de sortie
    Retour :
      - conv   : vecteur de convergence (meilleur coût par génération)
      - best   : individu optimal trouvé (paramètres)
    """

    # 1) Nombre de générations = budget ÷ taille population
    Ngen = int(budget / Npop)

    # 2) Facteurs de mutation pour DE (poids des vecteurs)
    F1 = 0.9
    F2 = 0.8

    # 3) Buffers pour stocker :
    #    cf[g]    = coût (fonction taper) de chaque individu
    #    conv[g]  = meilleur coût observé à chaque génération g
    cf   = np.zeros(Npop)
    conv = np.zeros(Ngen)

    # 4) Nombre de paramètres par couche : 1 épaisseur + Ncubes largeurs
    Nparam = Ncubes + 1

    # ──────────────────────────────────────────────────────────────── #
    # 5) INITIALISATION DE LA POPULATION
    # ──────────────────────────────────────────────────────────────── #
    #    arr shape = (Npop, Nlayers, Nparam), valeurs uniformes [0,1)
    arr = np.random.rand(Npop, Nlayers, Nparam)

    # 6) Application des contraintes continues :
    #    - pour la dimension 0 (épaisseurs) : map de [0,1) → [bornes[0], bornes[1]]
    arr[:, :, 0] = bornes[0] + (bornes[1] - bornes[0]) * arr[:, :, 0]

    #    - pour la dimension 1 (largeurs) : map de [0,1) → [bornes[2], bornes[3]]
    arr[:, :, 1] = bornes[2] + (bornes[3] - bornes[2]) * arr[:, :, 1]


    # 7) ÉVALUATION INITIALE DE CHAQUE INDIVIDU
    #    taper(individu) calcule le coût pour la matrice de taille (Nlayers, Nparam)
    for l in range(Npop):
        cf[l] = taper(arr[l])


    # ──────────────────────────────────────────────────────────────── #
    # 8) BOUCLE PRINCIPALE DE DE SUR Ngen GÉNÉRATIONS
    # ──────────────────────────────────────────────────────────────── #
    for g in range(Ngen):
        for p in range(Npop):
            # 8a) Sélection aléatoire de 3 indices distincts dans [0..Npop-1]
            idxs = np.random.randint(Npop, size=(3,))
            a, b, c = arr[idxs[0]], arr[idxs[1]], arr[idxs[2]]

            # 8b) Récupère l’individu « best » au plus petit coût actuel
            best = arr[np.argmin(cf)]

            # 8c) MUTATION current-to-best/1
            #     y = c + F1*(a - b) + F2*(best - c)
            y = c + F1 * (a - b) + F2 * (best - c)

            # 8d) CROSSOVER binomial (probabilité cr pour chaque ligne)
            cr = 0.8
            ii = np.random.rand(Nlayers, 1)  # une probabilité par couche
            z  = np.zeros((Nlayers, Nparam))
            for nl in range(Nlayers):
                # si ii[nl] <= cr → on prend y[nl], sinon arr[p,nl]
                z[nl, :] = (ii[nl] <= cr) * y[nl, :] + \
                           (ii[nl] >  cr) * arr[p, nl, :]

                # 8e) REMISE AUX BORNES si hors [min,max]
                #     bornes[0..3] mappent sur idx=0..1 respectivamente
                for idx in range(2):  
                    cond_min = z[nl, idx] < bornes[2*idx]
                    cond_max = z[nl, idx] > bornes[2*idx+1]
                    # si hors bornes, on remet la valeur précédente arr[p,nl,idx]
                    z[nl, idx] = (cond_min == cond_max) * z[nl, idx] \
                                + cond_min * arr[p, nl, idx] \
                                + cond_max * arr[p, nl, idx]

            # 8f) Évaluation du vecteur candidat z
            cfz = taper(z)

            # 8g) SÉLECTION : si z est meilleur, on remplace p par z
            if cfz < cf[p]:
                arr[p] = z.copy()
                cf[p]  = cfz

        # 8h) Convergence : stocke le plus petit coût de cette génération
        conv[g] = cf.min()


    # ──────────────────────────────────────────────────────────────── #
    # 9) RÉÉVALUATION FINALE (pour avoir best_final fiable)
    # ──────────────────────────────────────────────────────────────── #
    cf_final = np.zeros(Npop)
    for l in range(Npop):
        cf_final[l] = taper(arr[l])

    best       = arr[np.argmin(cf)]        # meilleur selon évaluation initiale
    best_final = arr[np.argmin(cf_final)]  # meilleur selon ré-évaluation

    # 10) Sauvegarde des résultats
    savemat(f'taper_{Nopt}.mat', {
        'conv':      conv,
        'best':      best,
        'cf_final':  cf_final,
        'best_final':best_final
    })
    np.savez(f"taper_{Nopt}.npz", conv=conv, best=best)

    # 11) Retourne l’historique de convergence et le meilleur individu
    return conv, best



#Optimisation
budget = 100 #32000
Npop = 30
periode = 3000
thick_min = 1
thick_max = 30
width_min = 5
width_max = 70 #pas plus
NC = 1

bornes = np.array([thick_min,thick_max,width_min,width_max])
NL = 3

Parallel(n_jobs=10)(delayed(DE_general)(budget, Npop, NL, NC, bornes, periode,Nopt) for Nopt in range(50))
