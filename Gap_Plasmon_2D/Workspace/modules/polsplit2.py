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

def marche(a,b,p,n,x):
    '''Computes the Fourier series for a piecewise function having the value
    a over a portion p of the period, starting at position x
    and the value b otherwise. The period is supposed to be equal to 1.
    Division by zero or very small values being not welcome, think about
    not taking round values for the period or for p. Then takes the toeplitz
    matrix generated using the Fourier series.
    '''
    from scipy.linalg import toeplitz
    l=np.zeros(n,dtype=np.complex)
    m=np.zeros(n,dtype=np.complex)
    tmp=1/(2*np.pi*np.arange(1,n))*(np.exp(-2*1j*np.pi*p*np.arange(1,n))-1)*np.exp(-2*1j*np.pi*np.arange(1,n)*x)
    l[1:n]=1j*(a-b)*tmp 
    l[0]=p*a+(1-p)*b
    m[0]=l[0]
    m[1:n]=1j*(b-a)*np.conj(tmp)
    T=toeplitz(l,m)
    return T

def creneau(k0,a0,pol,e1,e2,a,n,x0):
    '''Attention : a refers to the proportion of e1 in the period, and x0
    to the starting position of this inclusion in a material of
    permittivity e2'''
    nmod=int(n/2)
    alpha=np.diag(a0+2*np.pi*np.arange(-nmod,nmod+1))
    if (pol==0):
        M=alpha*alpha-k0*k0*marche(e1,e2,a,n,x0)
        L,E=np.linalg.eig(M)
        L=np.sqrt(-L+0j)
        L=(1-2*(np.imag(L)<-1e-15))*L
        P=np.block([[E],[np.matmul(E,np.diag(L))]])
    else:
        U=marche(1/e1,1/e2,a,n,x0)
        T=np.linalg.inv(U)
        M=np.matmul(np.matmul(np.matmul(T,alpha),np.linalg.inv(marche(e1,e2,a,n,x0))),alpha)-k0*k0*T
        L,E=np.linalg.eig(M)
        L=np.sqrt(-L+0j)
        L=(1-2*(np.imag(L)<-1e-15))*L
        P=np.block([[E],[np.matmul(np.matmul(U,E),np.diag(L))]])
    return P,L

def reseau(k0,a0,pol,e1,e2,n,blocs):
    '''Warning: blocs is a vector with N lines and 2 columns. Each
    line refers to a block of material e2 inside a matrix of material e1,
    giving its size relatively to the period (first column) and its starting
    position. #not anymore
    Warning : There is nothing checking that the blocks don't overlapp.
    '''
    n_blocs=blocs.shape[0]; # Nombre de lignes
    nmod=int(n/2)
    M1=marche(e2,e1,blocs[0,0],n,blocs[0,1]) # Renvoi Toeplitz
    M2=marche(1/e2,1/e1,blocs[0,0],n,blocs[0,1])
    if n_blocs>1:
        for k in range(1,n_blocs):
            M1=M1+marche(e2-e1,0,blocs[k,0],n,blocs[k,1])
            M2=M2+marche(1/e2-1/e1,0,blocs[k,0],n,blocs[k,1])
    alpha=np.diag(a0+2*np.pi*np.arange(-nmod,nmod+1))+0j
    if (pol==0):
        M=alpha*alpha-k0*k0*M1
        L,E=np.linalg.eig(M)
        L=np.sqrt(-L+0j)
        L=(1-2*(np.imag(L)<-1e-15))*L
        P=np.block([[E],[np.matmul(E,np.diag(L))]])
    else:
        T=np.linalg.inv(M2)
        M=np.matmul(np.matmul(np.matmul(T,alpha),np.linalg.inv(M1)),alpha)-k0*k0*T
        L,E=np.linalg.eig(M)
        L=np.sqrt(-L+0j)
        L=(1-2*(np.imag(L)<-1e-15))*L
        P=np.block([[E],[np.matmul(np.matmul(M2,E),np.diag(L))]])
    return P,L

def homogene(k0,a0,pol,epsilon,n):
    '''Generates the P matrix and the wavevectors exactly as for a
    periodic layer, just for an homogeneous layer. The results are
    analytic in that case.
    '''
    nmod=int(n/2)
    valp=np.sqrt(epsilon*k0*k0-(a0+2*np.pi*np.arange(-nmod,nmod+1))**2+0j)
    valp=valp*(1-2*(valp<0))*(pol/epsilon+(1-pol))
    P=np.block([[np.eye(n)],[np.diag(valp)]])
    return P,valp

def interface(P,Q):
    '''Computation of the scattering matrix of an interface, P and Q being the
    matrices given for each layer by homogene, reseau or creneau.
    '''
    n=int(P.shape[1])
    S=np.matmul(np.linalg.inv(np.block([[P[0:n,0:n],-Q[0:n,0:n]],[P[n:2*n,0:n],Q[n:2*n,0:n]]])),np.block([[-P[0:n,0:n],Q[0:n,0:n]],[P[n:2*n,0:n],Q[n:2*n,0:n]]]))
    return S

def polsplitt2(X):
    '''Splitter en polarisation. Multiples éléments par couche'''
    lam=600
    d=600*1.4142135623730951
    e1=1
    e2=1.46**2

    x=X/d
    #print(x)
    l=lam/d
    k0=2*np.pi/l

    nmod=25
    n=2*nmod+1
    n_layers=x.shape[1]
    n_cubes = 2

    pol=0
    S=np.block([[np.zeros([n,n]),np.eye(n,dtype=np.complex)],[np.eye(n),np.zeros([n,n])]])
    P,V=homogene(k0,0,pol,e1,n)
    for k in range(0,n_layers):
#        Pc,Vc=creneau(k0,0,pol,e2,1,x[k,1],n,x[k,2])
        cubes_vecteur = x[1:5,k]
        cubes = np.reshape(cubes_vecteur,(n_cubes,2))
        Pc,Vc=reseau(k0,0,pol,e1,e2,n,cubes)
        S=cascade(S,interface(P,Pc))
        S=c_bas(S,Vc,x[0,k])
        P=Pc
        V=Vc
    Pc,Vc=homogene(k0,0,pol,e2,n)
    S=cascade(S,interface(P,Pc))
    P,V=homogene(k0,0,pol,e1,n)
    Te=np.zeros(3,dtype=np.float)
    for j in range(-1,2):
        Te[j+1]=abs(S[j+nmod,n+nmod])**2*np.real(V[j+nmod])/(1.46*k0)
    #print(Te)

    pol=1
    S=np.block([[np.zeros([n,n]),np.eye(n,dtype=np.complex)],[np.eye(n),np.zeros([n,n])]])
    P,V=homogene(k0,0,pol,e1,n)
    for k in range(0,n_layers):
#        Pc,Vc=creneau(k0,0,pol,e2,1,x[k,1],n,x[k,2])
        cubes_vecteur = x[1:5,k]
        cubes = np.reshape(cubes_vecteur,(n_cubes,2))
        Pc,Vc=reseau(k0,0,pol,e1,e2,n,cubes)
        S=cascade(S,interface(P,Pc))
        S=c_bas(S,Vc,x[0,k])
        P=Pc
        V=Vc
    Pc,Vc=homogene(k0,0,pol,e2,n)
    S=cascade(S,interface(P,Pc))
    P,V=homogene(k0,0,pol,e1,n)
    Tm=np.zeros(3,dtype=np.float)
    for j in range(-1,2):
        Tm[j+1]=abs(S[j+nmod,n+nmod])**2*np.real(V[j+nmod])*1.46/k0
    #print(Tm)

    cost=1-(Te[0]+Tm[2])/2

    return cost

def DE_general(budget,Npop,Nlayers,bornes,periode,Nopt):
    Ngen=int(budget/Npop)
    F1=0.9
    F2=0.8
    cf=np.zeros(Npop)
    conv=np.zeros(Ngen)
    Nparam=int(bornes.size/2)
    x=np.random.rand(Nparam,Npop,Nlayers)
    arr=np.zeros(((Npop,Nparam,Nlayers)))
    for k in range(Nparam):
        x[k]=bornes[2*k]+(bornes[2*k+1]-bornes[2*k])*x[k]
    for l in range(Npop):
        for idx in range(Nparam):
            arr[l,idx,:]=x[idx,l]
        cf[l]=polsplitt2(arr[l])
    for g in range(Ngen):
        #print(g)
        for p in range(Npop):
            index=np.random.randint(Npop,size=(3))
            a=arr[index[0]]
            b=arr[index[1]]
            c=arr[index[2]]
            best=arr[np.argmin(cf)]
            y=c+F1*(a-b)+F2*(best-c)
            cr=0.8
            ii=np.random.rand(1,Nlayers)
            z=(ii<=cr)*y+(ii>cr)*arr[p]
            for idx in range(Nparam): #modification de cette ligne : Nparam -1 est devenu Nparam -0 !
                cond_min=z[idx]<bornes[2*idx]
                cond_max=z[idx]>bornes[2*idx+1]
                z[idx]=(cond_min==cond_max)*z[idx]+cond_min*arr[p,idx]+cond_max*arr[p,idx]
            cfz=polsplitt2(z)
            if cfz<cf[p]:
                arr[p]=z #modification de cette ligne
                cf[p]=cfz
        conv[g]=np.min(cf)
    cf_final = np.zeros(Npop)
    for l in range(Npop):
        cf_final[l]=polsplitt2(arr[l])
    best=arr[np.argmin(cf)]
    best_final=arr[np.argmin(cf_final)]
    savemat('pola_%d.mat' %Nopt,{'conv':conv,'best':best,'cf_final' : cf_final,'best_final': best_final})
    np.savez("pola_%d.npz" %Nopt,conv=conv,best=best)
    return(conv,best)

budget=200000
Npop=30
periode = 600*1.4142135623730951
t_min = 0
t_max = 200
w1_min = 70
w1_max = 300
p1_min = 0
p1_max = 300
w2_min = 70
w2_max = 300
p2_min = 0
p2_max = 300

bornes=np.array([t_min,t_max,w1_min,w1_max,p1_min,p1_max,w2_min,w2_max,p2_min,p2_max])
NL=3
Parallel(n_jobs=10)(delayed(DE_general)(budget,Npop,NL,bornes,periode,Nopt) for Nopt in range(50))
#conv,best = DE_general(budget, Npop, NL, bornes, periode,1)

# best = np.array([t1,t2,t3],
                #[w11,w12,w13],
                #[p11,p12,p13],
                #[w21,w22,w23],
                #[p21,p22,p23])
