

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

def DE_general(budget,Npop,Nlayers,Ncubes, bornes,periode,Nopt):
    Ngen=int(budget/Npop)
    F1=0.9
    F2=0.8
    cf=np.zeros(Npop)
    conv=np.zeros(Ngen)
    Nparam = Ncubes+1

    #initialisation pop
    arr = np.random.rand(Npop,Nlayers,Nparam)
    #contraintes sur les epaisseurs
    arr[:,:,0] = bornes[0] + (bornes[1]-bornes[0])*arr[:,:,0]
    #contraintes sur les largeurs
    arr[:,:,1] = bornes[2]+(bornes[3]-bornes[2])*arr[:,:,1]

    for l in range(Npop):
        cf[l]=taper(arr[l])

    for g in range(Ngen):
        for p in range(Npop):
            index=np.random.randint(Npop,size=(3))
            a=arr[index[0]]
            b=arr[index[1]]
            c=arr[index[2]]
            best=arr[np.argmin(cf)]
            y=c+F1*(a-b)+F2*(best-c)
            #print("y=", y)
            cr=0.8
            ii=np.random.rand(Nlayers,1)
            z = np.zeros((Nlayers, Nparam))
            for nl in range(Nlayers):
                z[nl,:] = (ii[nl]<=cr)*y[nl,:] + (ii[nl]>cr)*arr[p,nl,:]
                for idx in range(0,2):
                    cond_min = z[nl,idx] < bornes[2*idx]
                    cond_max = z[nl,idx] > bornes[2*idx+1]
                    z[nl,idx] = (cond_min==cond_max)*z[nl,idx]+cond_min*arr[p,nl,idx]+cond_max*arr[p,nl,idx]
            #print("z=", z)
            cfz=taper(z)
            if cfz<cf[p]:
                arr[p]=z
                cf[p]=cfz
        conv[g]=np.min(cf)
    cf_final = np.zeros(Npop)
    for l in range(Npop):
        cf_final[l]=taper(arr[l])
    best=arr[np.argmin(cf)]
    best_final=arr[np.argmin(cf_final)]
    savemat('taper_%d.mat' %Nopt,{'conv':conv,'best':best,'cf_final' : cf_final,'best_final': best_final})
    np.savez("taper_%d.npz" %Nopt,conv=conv,best=best)
    return(conv,best)
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
