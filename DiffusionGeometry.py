import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh, svd
from opt_einsum import contract
from numpy.linalg import norm
import scipy as sp

def Del0(x, epsilon, n0, alpha = 1):
    d = squareform(pdist(x))
    # if epsilon is None:
    #     k = 1 + np.ceil(np.log(n))
    #     epsilon = np.mean(np.partition(d, k, axis=0)[:k, :])
    d = np.exp(-d**2 / (4*epsilon**2))
    if not alpha == 0:
        D = diags(1 / np.sum(d, axis=1) ** alpha)
        d = D @ d @ D
    D = np.sum(d, axis=1)
    l, u = eigsh(d, n0, diags(D), which = 'LM')
    u = u.T
    u *= np.sign(u.sum(axis=1)).reshape(-1,1)
    l = - np.log(l) / epsilon**2
    return u[::-1], l[::-1], D

def Diffusion_Operator(x, epsilon, alpha = 1):
    d = squareform(pdist(x))
    # if epsilon is None:
    #     k = 1 + np.ceil(np.log(n))
    #     epsilon = np.mean(np.partition(d, k, axis=0)[:k, :])
    d = np.exp(-d**2 / (4*epsilon**2))
    if not alpha == 0:
        D = diags(1 / np.sum(d, axis=1) ** alpha)
        d = D @ d @ D
    D = np.sum(d, axis=1)
    return d / D

def C3_110(u, D, parameters):
    n,n0,n1 = parameters['n'], parameters['n0'], parameters['n1']
    return (u[:n1].reshape(n1,1,1,n)
            *u[:n1].reshape(1,n1,1,n)
            *u.reshape(1,1,n0,n)*D).sum(axis=-1)

def C3_010(u, D, parameters):
    n,n0,n1 = parameters['n'], parameters['n0'], parameters['n1']
    return (u.reshape(n0,1,1,n)
            *u[:n1].reshape(1,n1,1,n)
            *u.reshape(1,1,n0,n)*D).sum(axis=-1)

def C3_000(u, D, parameters):
    n,n0 = parameters['n'], parameters['n0']
    return (u.reshape(n0,1,1,n)
            *u.reshape(1,n0,1,n)
            *u.reshape(1,1,n0,n)*D).sum(axis=-1)

def C4_1100(C3_110, C3_000):
    return contract('ijs,kls->ijkl', C3_110, C3_000)

def C4_0000(C3_000):
    return contract('ijs,kls->ijkl', C3_000, C3_000)

def CdC_220(lam, C3_110, parameters):
    n2 = parameters['n2']
    return (1/2)*(lam[:n2].reshape(-1,1,1)
                 +lam[:n2].reshape(1,-1,1)
                 -lam.reshape(1,1,-1))*C3_110[:n2,:n2]

def CdC_020(lam, C3_010, parameters):
    n2 = parameters['n2']
    return (1/2)*(lam.reshape(-1,1,1)
                 +lam[:n2].reshape(1,-1,1)
                 -lam.reshape(1,1,-1))*C3_010[:,:n2]

def G1(C3_110, CdC_220, parameters):
    n1,n2 = parameters['n1'], parameters['n2']
    tensorG1 = contract('iks,jls->ijkl', C3_110, CdC_220)
    G1 = tensorG1.reshape(n1*n2,n1*n2)
    return G1

def metric1(C3_000, CdC_220, parameters):
    n0,n1,n2 = parameters['n0'], parameters['n1'], parameters['n2']
    tensorg1 = contract('iku,jlv,uvs->ijkls', C3_000[:n1,:n1], CdC_220, C3_000)
    # tensorG1 = tensorG1.transpose([1,0,3,2])
    g1 = tensorg1.reshape(n1*n2,n1*n2,n0)
    # G1 = (G1+G1.T)/2
    return g1

def G1_VF(C3_010, CdC_020, parameters):
    n0,n1,n2 = parameters['n0'], parameters['n1'], parameters['n2']
    tensorG1_vf = contract('iks,jls->ijkl', C3_010, CdC_020)
    G1_vf = tensorG1_vf.reshape(n0**2,n1*n2)
    return G1_vf

def G2(C4_1100, CdC_220, parameters):
    n0,n1,n2 = parameters['n0'], parameters['n1'], parameters['n2']
    Gamma2 = CdC_220.reshape(n2,1,n2,1,n0,1)*CdC_220.reshape(1,n2,1,n2,1,n0) - CdC_220.reshape(1,n2,n2,1,n0,1)*CdC_220.reshape(n2,1,1,n2,1,n0)
    tensorG2 = contract('iIst, jkJKst -> ijkIJK', C4_1100, Gamma2)
    G2 = tensorG2.reshape(n1*n2*n2,n1*n2*n2)
    return G2

def metric2(C3_000, C4_0000, CdC_220, parameters):
    n0,n1,n2 = parameters['n0'], parameters['n1'], parameters['n2']
    Gamma2 = CdC_220.reshape(n2,1,n2,1,n0,1)*CdC_220.reshape(1,n2,1,n2,1,n0) - CdC_220.reshape(1,n2,n2,1,n0,1)*CdC_220.reshape(n2,1,1,n2,1,n0)
    tensorg2 = contract('iIu,ustl,jkJKst->ijkIJKl', C3_000[:n1,:n1], C4_0000, Gamma2)
    g2 = tensorg2.reshape(n1*n2*n2,n1*n2*n2,n0)
    return g2

def HodgeEnergy1(C3_110, lam, parameters):
    n1,n2 = parameters['n1'], parameters['n2']
    cikjl1 = contract('ijs,kls,s->ikjl', C3_110, C3_110[:n2,:n2], lam)
    ciljk1 = contract('ijs,kls,s->ilkj', C3_110[:,:n2], C3_110[:,:n2], lam)
    tensorD1 = (1/4)*(lam[:n1].reshape(-1,1,1,1) 
                + lam[:n2].reshape(1,-1,1,1) 
                + lam[:n1].reshape(1,1,-1,1) 
                + lam[:n2].reshape(1,1,1,-1))*(ciljk1-cikjl1)

    cijkl1 = contract('ijs,kls,s->ijkl', C3_110[:,:n2], C3_110[:,:n2], lam)
    tensorD1 += (1/4)*(-lam[:n1].reshape(-1,1,1,1) 
                + lam[:n2].reshape(1,-1,1,1) 
                - lam[:n1].reshape(1,1,-1,1) 
                + lam[:n2].reshape(1,1,1,-1))*cijkl1

    cikjl2 = contract('ijs,kls,s->ikjl', C3_110, C3_110[:n2,:n2], lam**2)
    ciljk2 = contract('ijs,kls,s->ilkj', C3_110[:,:n2], C3_110[:,:n2], lam**2)
    cijkl2 = contract('ijs,kls,s->ijkl', C3_110[:,:n2], C3_110[:,:n2], lam**2)
    tensorD1 += (1/4)*(cikjl2 + cijkl2 - ciljk2)

    D1 = tensorD1.reshape(n1*n2,n1*n2)
    return D1

def Up_Energy1(C3_110, lam, parameters):
    n1,n2 = parameters['n1'], parameters['n2']
    cijkl0 = contract('ijs,kls->ijkl', C3_110[:,:n2], C3_110[:,:n2])
    tensorD1 = (1/4)*((lam[:n1].reshape(-1,1,1,1)
                       -lam[:n2].reshape(1,-1,1,1))
                       *(-lam[:n1].reshape(1,1,-1,1)
                         +lam[:n2].reshape(1,1,1,-1)))*cijkl0
    cikjl1 = contract('ijs,kls,s->ikjl', C3_110, C3_110[:n2,:n2], lam)
    ciljk1 = contract('ijs,kls,s->ilkj', C3_110[:,:n2], C3_110[:,:n2], lam)
    tensorD1 += (1/4)*(lam[:n1].reshape(-1,1,1,1) 
                + lam[:n2].reshape(1,-1,1,1) 
                + lam[:n1].reshape(1,1,-1,1) 
                + lam[:n2].reshape(1,1,1,-1))*(ciljk1-cikjl1)

    cikjl2 = contract('ijs,kls,s->ikjl', C3_110, C3_110[:n2,:n2], lam**2)
    ciljk2 = contract('ijs,kls,s->ilkj', C3_110[:,:n2], C3_110[:,:n2], lam**2)
    tensorD1 += (1/4)*(cikjl2 - ciljk2)

    D1 = tensorD1.reshape(n1*n2,n1*n2)
    return D1

def Up_Energy1_new(CdC_110, parameters):
    n2 = parameters['n2']
    UD1 = contract('iIs,jJs -> ijIJ', CdC_110, CdC_110[:n2,:n2])
    UD1 -= contract('jIs,iJs -> ijIJ', CdC_110[:n2], CdC_110[:,:n2])
    return UD1

# def SVD_projection(G1, D1, tol = 1e-3):
#     U, S, _ = svd(G1)
#     min_index = np.where(S > S[0]*tol)[0][-1]
#     P1 = U[:,:min_index]
#     D1_proj = P1.T @ D1 @ P1
#     G1_proj = P1.T @ G1 @ P1
#     # D1_proj = (D1_proj+D1_proj.T)/2
#     # G1_proj = (G1_proj+G1_proj.T)/2
#     return G1_proj, D1_proj, P1

def eig_projection(G1, parameters):
    tol = parameters['projection_tol']
    S, U = eigh(G1)
    min_index = np.where(S > S[-1]*tol)[0][0]
    P1 = U[:,min_index:]
    # G1_proj = P1.T @ G1 @ P1
    G1_proj = diags(S[min_index:])
    G1_proj_inv = diags(1/S[min_index:])
    G1_inv = P1 @ G1_proj_inv @ P1.T
    return G1_proj, G1_proj_inv, G1_inv, P1

def ext_derivative0(u, parameters):
    n,n0,n1,n2 = parameters['n'], parameters['n0'], parameters['n1'], parameters['n2']
    tensord0 = (n/u[0,0])*np.concatenate((np.eye(n2,n0).reshape(1,n2,n0), np.zeros((n1-1,n2,n0))), axis = 0)
    return tensord0.reshape(n1*n2,n0)

def ext_derivative0_weak(C3_110, lam, parameters):
    n0,n1,n2 = parameters['n0'], parameters['n1'], parameters['n2']
    C3_120 = C3_110[:,:n2]
    CdC_201 = (1/2)*(-lam[:n1].reshape(-1,1,1)
                    +lam[:n2].reshape(1,-1,1)
                    +lam.reshape(1,1,-1))*C3_120
    return CdC_201.reshape(n1*n2,n0)

def ext_derivative1_weak(CdC_220, C3_010, d0_w, parameters):
    n0,n1,n2 = parameters['n0'], parameters['n1'], parameters['n2']
    CdC_210 = d0_w.reshape(n1,n2,n0).transpose((1,0,2))
    Gamma2_2212 = CdC_210.reshape(n2,1,n1,1,n0,1)*CdC_220.reshape(1,n2,1,n2,1,n0) - CdC_210.transpose((1,0,2)).reshape(1,n2,n1,1,n0,1)*CdC_220.reshape(n2,1,1,n2,1,n0)
    return contract('ist, jkIJst -> ijkIJ', C3_010.transpose((1,0,2)), Gamma2_2212).reshape(n1*n2*n2, n1*n2)

def wedge_product_matrix01(C3_110, parameters):
    # a_i, f_j -> (fa)_k
    n0,n1,n2 = parameters['n0'], parameters['n1'], parameters['n2']
    tensorW01 = C3_110.transpose(0,2,1).reshape(n1,1,n0,n1,1) * np.eye(n2).reshape(1,n2,1,1,n2)
    return tensorW01.reshape(n1*n2,n0,n1*n2)

def wedge_product_matrix02(C3_110, parameters):
    # a_i, f_j -> (fa)_k
    n0,n1,n2 = parameters['n0'], parameters['n1'], parameters['n2']
    tensorW02 = C3_110.transpose(0,2,1).reshape(n1,1,1,n0,n1,1,1) * np.eye(n2).reshape(1,n2,1,1,1,n2,1) * np.eye(n2).reshape(1,1,n2,1,1,1,n2)
    return tensorW02.reshape(n1*n2*n2,n0,n1*n2*n2)

def wedge_product_matrix11(C3_110, parameters):
    # a_i, b_j -> (ab)_k
    n1,n2 = parameters['n1'], parameters['n2']
    tensorW02 = C3_110[:,:,:n1].reshape(n1,1,n1,1,n1,1,1) * np.eye(n2).reshape(1,n2,1,1,1,n2,1) * np.eye(n2).reshape(1,1,1,n2,1,1,n2)
    return tensorW02.reshape(n1*n2,n1*n2,n1*n2*n2)

def lie_bracket_matrix(G1_vf, G1_inv, parameters):
    # a_i, b_j -> [a,b]_k
    n0,n1,n2 = parameters['n0'], parameters['n1'], parameters['n2']
    # Compute the frame vector fields in operator form.
    vfs = G1_vf.reshape(n0,n0,n1*n2)
    # Compute the Lie bracket vector fields.
    bracket = contract('ijs,jkt->ikst', vfs, vfs)
    bracket -= bracket.transpose((0,1,3,2))
    # Convert the vector field back to a 1-form with the pseudo-inverse of G1.
    bracket = contract('ij,jkl->ikl', G1_inv, bracket[:n1,:n2].reshape(n1*n2, n1*n2, n1*n2))
    return bracket.transpose((1,2,0))

def bilinear(a,b,M):
    return contract('i,ijk,j->k', a, M, b)

def Levi_Civita_matrix(G1, G1_VF, G1_inv, g1, LB, u, parameters):
    # a_i, b_j -> [a,b]_k
    n0,n1,n2 = parameters['n0'], parameters['n1'], parameters['n2']
    # \int X(g(Y,Z))
    LCC1 = contract('jk,lmj->klm', G1_VF.reshape(n0,n0,n1*n2)[0], g1) / u[0,0]
    # \int X(g(Y,Z)) + \int Y(g(Z,X)) - \int Z(g(X,Y))
    LCC1 = LCC1 + LCC1.transpose((1,2,0)) - LCC1.transpose((2,0,1))
    # <[X,Y],Z>
    LCC2 = contract('ijk,kl', LB, G1)
    # <[X,Y],Z> - <[Y,Z],X> + <[Z,X],Y>
    LCC2 = LCC2 - LCC2.transpose((2,0,1)) + LCC2.transpose((0,2,1))
    LCC = (LCC1 + LCC2)/2
    # Convert the variational form into the frame representation with the pseudo-inverse of G1.
    LCC = contract('ijk,kl', LCC, G1_inv)
    return LCC

# def delta_projection(G1, parameters):
#     n1,n2 = parameters['n1'], parameters['n2']
#     return np.concatenate((np.zeros((1, n1*n2)),
#                            np.linalg.solve(G1[1:n2,1:n2], G1[1:n2]), 
#                            np.zeros((n1*n2-n2, n1*n2))), axis = 0)

def gradient_decomp(d0_w, lam, thres = 1e-8):
    l = np.copy(lam)
    l[l<thres] = 1
    return d0_w.T / l.reshape(-1,1)

def Riemann_Curvature_tensor(LCC, LB):
    # a_i, b_j, c_k -> ( R(a,b)c )_k
    RCT = contract('isl,jks->ijkl', LCC, LCC)
    RCT -= RCT.transpose((1,0,2,3))
    RCT -= contract('ijs,skl->ijkl', LB, LCC)
    return RCT

def Hess(G1_vf, LCC, parameters):
    # f_i, X_j, Y_k -> H(f)(X,Y)_l
    n0,n1,n2 = parameters['n0'], parameters['n1'], parameters['n2']
    op_form = G1_vf.reshape(n0,n0,n1*n2)
    XY = contract('ijs,jkt->ikst', op_form, op_form) 
    LCC_op = contract('stj,kj', LCC, G1_vf).reshape(n0,n0,n1*n2,n1*n2)
    return (XY - LCC_op).transpose((1,2,3,0))

