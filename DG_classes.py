import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh, svd
from numpy.linalg import norm
import scipy
from DiffusionGeometry import *
from visualisation import *

class DG_objects:
    def __init__(self):

        self.u = None
        self.lam = None
        self.D = None

        self.C3_110 = None
        self.C3_010 = None
        self.C3_000 = None
        self.C4_1100 = None
        self.C4_0000 = None
        self.CdC_220 = None
        self.CdC_020 = None

        self.G1 = None
        self.G1_VF = None
        self.G1_proj = None
        self.G1_proj_inv = None
        self.G1_inv = None
        self.P1 = None

        self.G2 = None
        self.G2_proj = None
        self.G2_proj_inv = None
        self.G2_inv = None
        self.P2 = None

        self.D1 = None
        self.UD1 = None
        self.grad_decomp_matrix = None
        
        self.g1 = None
        self.g2 = None
        
        self.d0_w = None
        self.d0 = None
        self.d1_w = None
        self.d1 = None

        self.wedge_01 = None
        self.wedge_02 = None
        self.wedge_11 = None

        self.Lie = None
        self.Levi_Civita = None
        self.Riemann = None
        self.Hessian = None

class DG:

    ### Initialisation
    def __init__(self, data, parameters):
        self.data = data
        self.parameters = parameters
        self.parameters['n'] = data.shape[0]
        if not self.parameters['n0'] < self.parameters['n']:
            self.parameters['n0'] = self.parameters['n'] - 1
        self.objects = DG_objects()

    ### Visualisation
    
    def plot(self):
        dim = self.data.shape[1]
        if dim == 2:
            plot_2d(self.data)
        elif dim == 3:
            plot_3d(self.data)
        else:
            raise TypeError('Data is not 2 or 3 dimensional, so cannot plot.')

    def plot_functions(self, functions, rows, columns, cap = -1):
        dim = self.data.shape[1]
        if dim == 2:
            plot_functions_2d(functions, self.data, rows, columns, cap)
        else:
            raise TypeError('Data is not 2 dimensional, so cannot plot multiple functions.')

    def plot_eigenfunctions(self, rows, cols):
        self.plot_functions(self.u(), rows, cols)

    def vector_field_coords(self, v):
        return vector_field_coordinates(v, 
                                        self.u(),
                                        self.D(),
                                        self.G1_VF(),
                                        self.data,
                                        self.parameters)
    


    ### Computing DG objects

    def u(self):
        if self.objects.u is None:
            ep, n0, alpha = self.parameters['ep'], self.parameters['n0'], self.parameters['alpha']
            self.objects.u, self.objects.lam, self.objects.D = Del0(self.data, ep, n0, alpha)
        return self.objects.u
    
    def lam(self):
        if self.objects.lam is None:
            ep, n0, alpha = self.parameters['ep'], self.parameters['n0'], self.parameters['alpha']
            self.objects.u, self.objects.lam, self.objects.D = Del0(self.data, ep, n0, alpha)
        return self.objects.lam
    
    def D(self):
        if self.objects.D is None:
            ep, n0, alpha = self.parameters['ep'], self.parameters['n0'], self.parameters['alpha']
            self.objects.u, self.objects.lam, self.objects.D = Del0(self.data, ep, n0, alpha)
        return self.objects.D
    
    def C3_110(self):
        if self.objects.C3_110 is None:
            if self.objects.C3_010 is None:
                self.objects.C3_110 = C3_110(self.u(), self.D(), self.parameters)
            else:
                self.objects.C3_110 = self.objects.C3_010[:self.parameters['n1']]
        return self.objects.C3_110
    
    def C3_010(self):
        if self.objects.C3_010 is None:
            if self.objects.C3_000 is None:
                self.objects.C3_010 = C3_010(self.u(), self.D(), self.parameters)
            else:
                self.objects.C3_010 = self.objects.C3_000[:,:self.parameters['n1']]
        return self.objects.C3_010
    
    def C3_000(self):
        if self.objects.C3_000 is None:
            self.objects.C3_000 = C3_000(self.u(), self.D(), self.parameters)
        return self.objects.C3_000
    
    def C4_0000(self):
        if self.objects.C4_0000 is None:
            self.objects.C4_0000 = C4_0000(self.C3_000())
        return self.objects.C4_0000
    
    def C4_1100(self):
        if self.objects.C4_1100 is None:
            if self.objects.C4_0000 is None:
                self.objects.C4_1100 = C4_1100(self.C3_110(), self.C3_000())
            else:
                self.objects.C4_1100 = self.objects.C4_0000[:self.parameters['n1'],:self.parameters['n1']]
        return self.objects.C4_1100

    def CdC_220(self):
        if self.objects.CdC_220 is None:
            if self.objects.CdC_020 is None:
                self.objects.CdC_220 = CdC_220(self.lam(), self.C3_010(), self.parameters)
            else:
                self.objects.CdC_220 = self.objects.CdC_020[:self.parameters['n2']]
        return self.objects.CdC_220
    
    def CdC_020(self):
            if self.objects.CdC_020 is None:
                self.objects.CdC_020 = CdC_020(self.lam(), self.C3_010(), self.parameters)
            return self.objects.CdC_020

    ## 1-forms: Gram matrix etc.

    def G1(self):
        if self.objects.G1 is None:
            if self.objects.G1_VF is None:
                self.objects.G1 = G1(self.C3_110(), self.CdC_220(), self.parameters)
            else:
                G1_VF = self.objects.G1_VF
                n0,n1,n2 = self.parameters['n0'],self.parameters['n1'],self.parameters['n2']
                self.objects.G1 = G1_VF.reshape(n0,n0,n1,n2)[:n1,:n2].reshape(n1*n2,n1*n2)
        return self.objects.G1
    
    def G1_VF(self):
        if self.objects.G1_VF is None:
            self.objects.G1_VF = G1_VF(self.C3_010(), self.CdC_020(), self.parameters)
        return self.objects.G1_VF
    
    def G1_proj(self):
        if self.objects.G1_proj is None:
            self.objects.G1_proj, self.objects.G1_proj_inv, self.objects.G1_inv, self.objects.P1 = eig_projection(self.G1(), self.parameters)
        return self.objects.G1_proj
    
    def G1_proj_inv(self):
        if self.objects.G1_proj_inv is None:
            self.objects.G1_proj, self.objects.G1_proj_inv, self.objects.G1_inv, self.objects.P1 = eig_projection(self.G1(), self.parameters)
        return self.objects.G1_proj_inv
    
    def G1_inv(self):
        if self.objects.G1_inv is None:
            self.objects.G1_proj, self.objects.G1_proj_inv, self.objects.G1_inv, self.objects.P1 = eig_projection(self.G1(), self.parameters)
        return self.objects.G1_inv
    
    def P1(self):
        if self.objects.P1 is None:
            self.objects.G1_proj, self.objects.G1_proj_inv, self.objects.G1_inv, self.objects.P1 = eig_projection(self.G1(), self.parameters)
        return self.objects.P1
    
    def D1(self):
        if self.objects.D1 is None:
            self.objects.D1 = HodgeEnergy1(self.C3_110(), self.lam(), self.parameters)
        return self.objects.D1

    def UD1(self):
        if self.objects.UD1 is None:
            self.objects.UD1 = Up_Energy1(self.C3_110(), self.lam(), self.parameters)
        return self.objects.UD1
    
    ## 2-forms: Gram matrix etc.
    
    def G2(self):
        if self.objects.G2 is None:
            self.objects.G2 = G2(self.C4_1100(), self.CdC_220(), self.parameters)
        return self.objects.G2
    
    def G2_proj(self):
        if self.objects.G2_proj is None:
            self.objects.G2_proj, self.objects.G2_proj_inv, self.objects.G2_inv, self.objects.P1 = eig_projection(self.G2(), self.parameters)
        return self.objects.G2_proj
    
    def G2_proj_inv(self):
        if self.objects.G2_proj_inv is None:
            self.objects.G2_proj, self.objects.G2_proj_inv, self.objects.G2_inv, self.objects.P1 = eig_projection(self.G2(), self.parameters)
        return self.objects.G2_proj_inv
    
    def G2_inv(self):
        if self.objects.G2_inv is None:
            self.objects.G2_proj, self.objects.G2_proj_inv, self.objects.G2_inv, self.objects.P1 = eig_projection(self.G2(), self.parameters)
        return self.objects.G2_inv
    
    def P2(self):
        if self.objects.P1 is None:
            self.objects.G2_proj, self.objects.G2_proj_inv, self.objects.G2_inv, self.objects.P1 = eig_projection(self.G2(), self.parameters)
        return self.objects.P1
    
    def weak_eigenproblem_1(self, matrix):
        if self.objects.P1 is None:
            self.objects.G1_proj, self.objects.G1_proj_inv, self.objects.G1_inv, self.objects.P1 = eig_projection(self.G1(), self.parameters)
        P1 = self.objects.P1
        matrix_proj = P1.T @ matrix @ P1 
        L1, U1 = sp.linalg.eigh(matrix_proj, self.G1_proj().toarray())
        try:
            pos = np.where(L1<0)[0][-1] + 1
        except:
            pos = 0
        L1, U1 = L1[pos:], U1[:,pos:]
        U1 = P1 @ U1
        return L1, U1
    
    def g1(self):
        if self.objects.g1 is None:
            self.objects.g1 = metric1(self.C3_000(), self.CdC_220(), self.parameters)
        return self.objects.g1
    
    def g2(self):
        if self.objects.g2 is None:
            self.objects.g2 = metric2(self.C3_000(), self.C4_0000(), self.CdC_220(), self.parameters)
        return self.objects.g2

    def d0_w(self):
        if self.objects.d0_w is None:
            self.objects.d0_w = ext_derivative0_weak(self.C3_110(), self.lam(), self.parameters)
        return self.objects.d0_w
    
    def d0(self):
        if self.objects.d0 is None:
            self.objects.d0 = self.G1_inv() @ self.d0_w()
        return self.objects.d0
    
    def d1_w(self):
        if self.objects.d1_w is None:
            self.objects.d1_w = ext_derivative1_weak(self.CdC_220(), self.C3_010(), self.d0_w(), self.parameters)
        return self.objects.d1_w
    
    def d1(self):
        if self.objects.d1 is None:
            self.objects.d1 = self.G2_inv() @ self.d1_w()
        return self.objects.d1
    
    # def d1(self):
    #     if self.objects.d1 is None:
    #         self.objects.d1 = self.G2_inv() @ self.d1_w()
    #     return self.objects.d1
    
    def grad_decomp_matrix(self):
        if self.objects.grad_decomp_matrix is None:
            self.objects.grad_decomp_matrix = gradient_decomp(self.d0_w(), self.lam())
        return self.objects.grad_decomp_matrix
    
    def wedge_01(self):
        if self.objects.wedge_01 is None:
            self.objects.wedge_01 = wedge_product_matrix01(self.C3_110(), self.parameters)
        return self.objects.wedge_01
    
    def wedge_02(self):
        if self.objects.wedge_02 is None:
            self.objects.wedge_02 = wedge_product_matrix02(self.C3_110(), self.parameters)
        return self.objects.wedge_02
    
    def wedge_11(self):
        if self.objects.wedge_11 is None:
            self.objects.wedge_11 = wedge_product_matrix11(self.C3_110(), self.parameters)
        return self.objects.wedge_11
    
    def Lie(self):
        if self.objects.Lie is None:
            self.objects.Lie = lie_bracket_matrix(self.G1_VF(), 
                                                  self.G1_inv(),
                                                  self.parameters)
        return self.objects.Lie
    
    def Levi_Civita(self):
        if self.objects.Levi_Civita is None:
            self.objects.Levi_Civita = Levi_Civita_matrix(self.G1(), 
                                                          self.G1_VF(), 
                                                          self.G1_inv(), 
                                                          self.g1(),
                                                          self.Lie(), 
                                                          self.u(), 
                                                          self.parameters)
        return self.objects.Levi_Civita
    
    def Riemann(self):
        if self.objects.Riemann is None:
            self.objects.Riemann = Riemann_Curvature_tensor(self.Levi_Civita(), self.Lie())
        return self.objects.Riemann
    
    def Hessian(self):
        if self.objects.Hessian is None:
            self.objects.Hessian = Hess(self.G1_VF(), self.Levi_Civita(), self.parameters)
        return self.objects.Hessian