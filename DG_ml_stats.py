import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh, svd

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from opt_einsum import contract
from numpy.linalg import norm
import scipy
from DG_classes import *

def feature_iteration_full(DG_class, features, parameters, limits):
    n0 = parameters['n0']
    n1 = parameters['n1']
    n2 = parameters['n2']
    numbers, functions, forms1 = features
    numbers_new, functions_new, forms1_new = [],[],[]

    ### 0 operations
    # Function product - only upper triangle because it's symmetric
    function_products = contract('in,jm,ijs->nms', functions, functions, DG_class.C3_000())
    functions_new.append(function_products[np.triu_indices(functions.shape[1])].T)

    # D0
    d0 = DG_class.d0() @ functions
    forms1_new.append(d0)

    ### 01 operations
    # Wedge product
    W01 = contract('in,jm,ijs->nms', forms1, functions, DG_class.wedge_01())
    forms1_new.append(W01.reshape(-1,n1*n2).T)

    # Hessian - only upper triangle because it's symmetric
    Hess = contract('in,jm,kl,ijks->mlns', functions, forms1, forms1, DG_class.Hessian())
    functions_new.append(Hess[np.triu_indices(forms1.shape[1])].reshape(-1,n0).T)

    ### 1 operations
    # Metric - only upper triangle because it's symmetric
    g1 = contract('in,jm,ijs->nms', forms1, forms1, DG_class.g1())
    functions_new.append(g1[np.triu_indices(forms1.shape[1])].T)

    # Inner product - only upper triangle because it's symmetric
    G1 = contract('in,jm,ij->nm', forms1, forms1, DG_class.G1())
    numbers_new.append(G1[np.triu_indices(forms1.shape[1])].T)

    # Codifferential
    d0_star = DG_class.d0().T @ forms1
    functions_new.append(d0_star)

    # Lie bracket - only upper triangle because it's antisymmetric
    LB = contract('in,jm,ijs->nms', forms1, forms1, DG_class.Lie())
    forms1_new.append(LB[np.triu_indices(forms1.shape[1])].T)

    # Levi-Civita connection - only upper triangle because 
    # it's torsion-free and we already have the Lie bracket
    LCC = contract('in,jm,ijs->nms', forms1, forms1, DG_class.Levi_Civita())
    forms1_new.append(LCC[np.triu_indices(forms1.shape[1])].T)

    # Hodge decomposition
    # grad = DG_class.d0() @ DG_class.grad_decomp_matrix() @ forms1
    # forms1_new.append(grad)
    # div_only_proportions = contract('si,st,ti->i', grad, DG_class.G1(), grad)
    # div_only_indices = np.where(div_only_proportions > 0.5)
    # div_only_forms = grad[:, div_only_indices].reshape(parameters['n1']*parameters['n2'],-1)
    # forms1_new.append(div_only_forms)

    # curl_only = forms1 - grad
    # curl_only_proportions = contract('si,st,ti->i', curl_only, DG_class.G1(), curl_only)
    # curl_only_indices = np.where(curl_only_proportions > 0.5)
    # curl_only_forms = curl_only[:, curl_only_indices].reshape(parameters['n1']*parameters['n2'],-1)
    # forms1_new.append(curl_only_forms)

    if limits['use curvature']:
        # print('computing RCT in frame')
        # Riemann curvature tensor - only upper triangle because it's antisymmetric
        # RCT_frame = DG_class.Riemann()[np.triu_indices(limits['Riemann curvature of eigenfunction frame'])].reshape(-1,n1*n2).T
        # forms1_new.append(RCT_frame)
        # print(RCT_frame.shape)

        # print('computing RCT for eigenforms')
        k = limits['Riemann curvature of eigenforms']
        RCT = contract('in,jm,ijks->nmks', forms1[:,:k], forms1[:,:k], DG_class.Riemann())[np.triu_indices(k)].reshape(-1,n1*n2).T
        forms1_new.append(RCT)
        # print(RCT.shape)

    numbers_new.append(numbers)
    functions_new.append(functions)
    forms1_new.append(forms1)

    ### -1 operations
    # Diffusion eigenvalues at different times
    numbers_new = np.concatenate(numbers_new)
    numbers_new1 = [numbers_new]
    for t in [1,10,50]:
        numbers_new1.append(np.exp(-t*numbers_new))
    numbers_new = np.concatenate(numbers_new1)

    functions_new = np.concatenate(functions_new, axis = 1)
    forms1_new = np.concatenate(forms1_new, axis = 1)
    return numbers_new, functions_new, forms1_new

def feature_iteration_basic(DG_class, features, parameters):
    n0 = parameters['n0']
    n1 = parameters['n1']
    n2 = parameters['n2']
    numbers, functions, forms1 = features
    numbers_new, functions_new, forms1_new = [],[],[]

    ### 0 operations
    # Function product
    function_products = contract('in,jm,ijs->nms', functions, functions, DG_class.C3_000())
    functions_new.append(function_products[np.triu_indices(functions.shape[1])].T)

    # D0
    d0 = DG_class.d0() @ functions
    forms1_new.append(d0)

    ### 01 operations
    # Wedge product
    W01 = contract('in,jm,ijs->nms', forms1, functions, DG_class.wedge_01())
    forms1_new.append(W01.reshape(-1,n1*n2).T)

    ### 1 operations
    # Metric - only upper triangle because it's symmetric
    g1 = contract('in,jm,ijs->nms', forms1, forms1, DG_class.g1())
    functions_new.append(g1[np.triu_indices(forms1.shape[1])].T)

    # Inner product - only upper triangle because it's symmetric
    G1 = contract('in,jm,ij->nm', forms1, forms1, DG_class.G1())
    numbers_new.append(G1[np.triu_indices(forms1.shape[1])].T)

    # Codifferential
    d0_star = DG_class.d0().T @ forms1
    functions_new.append(d0_star)

    # Hodge decomposition
    grad = DG_class.d0() @ DG_class.grad_decomp_matrix() @ forms1
    forms1_new.append(grad)

    numbers_new.append(numbers)
    functions_new.append(functions)
    forms1_new.append(forms1)
    numbers_new = np.concatenate(numbers_new)
    functions_new = np.concatenate(functions_new, axis = 1)
    forms1_new = np.concatenate(forms1_new, axis = 1)
    return numbers_new, functions_new, forms1_new

def numbers_output(DG_class, features, limits):

    numbers, functions, forms1 = features
    numbers_new = []

    ### 0 operations
    # Function metric - only upper triangle because it's symmetric
    function_metric = contract('in,im->nm', functions, functions)
    numbers_new.append(function_metric[np.triu_indices(functions.shape[1])].flatten())

    ### 1 operations
    # Metric - only upper triangle because it's symmetric
    g1 = contract('in,jm,ijs->nms', forms1, forms1, DG_class.g1()[:,:,:limits['metric expansion']])
    numbers_new.append(g1[np.triu_indices(forms1.shape[1])].flatten())

    # Inner product - only upper triangle because it's symmetric
    G1 = contract('in,jm,ij->nm', forms1, forms1, DG_class.G1())
    numbers_new.append(G1[np.triu_indices(forms1.shape[1])].T)

    numbers_new.append(numbers)
    numbers_new.append(functions[:limits['eigenfunctions']].flatten())
    numbers_new.append(forms1.flatten())
    numbers_new = np.concatenate(numbers_new)
    return numbers_new

def feature_vector(DG_class, limits, verbose = False):

    n1 = DG_class.parameters['n1']
    n2 = DG_class.parameters['n2']
    div_threshold = limits['divergence-only threshold']

    DG_class.C3_000()
    DG_class.CdC_020()
    DG_class.G1_VF()
    L1, U1 = DG_class.weak_eigenproblem_1(DG_class.D1())

    ## Hodge decomposition of the spectrum.
    # Filtering the divergence-only forms:
    grad = DG_class.d0() @ DG_class.grad_decomp_matrix() @ U1
    div_only_proportions = contract('si,st,ti->i', grad, DG_class.G1(), grad)
    div_only_indices = np.where(div_only_proportions > div_threshold)[0]
    div_only_forms = U1[:, div_only_indices]
    div_only_eigenvalues = L1[div_only_indices]
    # Filtering the curl-only forms:
    curl_only = U1 - grad
    curl_only_indices = np.where(div_only_proportions <= div_threshold)[0]
    curl_only_forms = U1[:, curl_only_indices]
    curl_only_eigenvalues = L1[curl_only_indices]

    curl_shortfall = limits['1-eigenforms-curl'] - curl_only_forms.shape[1]
    if curl_shortfall > 0:
        curl_only_forms = np.concatenate([curl_only_forms, 
                                          np.zeros((n1*n2, curl_shortfall))], 
                                          axis=1)
        curl_only_eigenvalues = np.concatenate([curl_only_eigenvalues, [L1.max()]*curl_shortfall])

    if verbose:
        print('div only:', div_only_forms.shape[1])
        print('curl only:', curl_only_forms.shape[1])

    ## Form the base collection of data: numbers, functions, and 1-forms.
    base_data = [np.concatenate([DG_class.lam()[:limits['eigenfunctions']], 
                                 div_only_eigenvalues[:limits['1-eigenforms-div']],
                                 curl_only_eigenvalues[:limits['1-eigenforms-curl']]]),
                np.eye(DG_class.parameters['n0'])[:,:limits['eigenfunctions']],
                np.concatenate([div_only_forms[:,:limits['1-eigenforms-div']],
                                curl_only_forms[:,:limits['1-eigenforms-curl']]], axis=1)]
    
    ## Compute arbitrary combinations of the base data using diffusion geometry operators.
    numbers_new, functions_new, forms1_new = feature_iteration_full(DG_class, base_data, DG_class.parameters, limits)
    return numbers_output(DG_class, [numbers_new, functions_new, forms1_new], limits)


def cross_validate_model(clf, pair_features, y, cell1, cell2, numbers, train_split = 0.8, runs = 100):
    
    acc = []
    feats_nbr = []
    for _ in range(runs):

        mask = []

        mask1 = np.array([False] * numbers[cell1])
        mask1[: int(np.floor(numbers[cell1] * train_split))] = True
        np.random.shuffle(mask1)
        mask.append(mask1)

        mask2 = np.array([False] * numbers[cell2])
        mask2[: int(np.floor(numbers[cell2] * train_split))] = True
        np.random.shuffle(mask2)
        mask.append(mask2)

        mask = np.concatenate(mask)

        X_train = pair_features[mask]
        y_train = y[mask]
        X_test = pair_features[np.logical_not(mask)]
        y_test = y[np.logical_not(mask)]

        
        # clf = LinearDiscriminantAnalysis()
        clf.fit(X_train, y_train)

        features_used = np.where(np.absolute(clf.coef_) > 0)[1]
        nbr_features_used = features_used.shape[0]
        feats_nbr.append(nbr_features_used)

        acc.append(accuracy_score(y_test, clf.predict(X_test)))

    acc = np.array(acc)
    feats_nbr = np.array(feats_nbr)

    return acc.mean(), acc.std(), feats_nbr.mean(), feats_nbr.std()

def compute_dg_features(parameters, limits, cells):
    dg_features = []
    for cell_type in cells:
        cell_type_dg = []
        for sample in cell_type:
            vec = feature_vector(DG(sample, parameters), limits)
            cell_type_dg.append(vec)
        dg_features.append(np.array(cell_type_dg))
    return dg_features

def norm_features(dg_features, tol = 1e-8):
    all_features = np.concatenate(dg_features)
    good_features = np.where(all_features.std(axis=0) > tol)[0]
    selected_features = all_features[:,good_features]

    norm_dg_features = []
    for cell_type in dg_features:
        selected = cell_type[:,good_features]
        selected -= selected_features.mean(axis=0)
        selected /= selected_features.std(axis=0)
        norm_dg_features.append(selected)

    return norm_dg_features



def string_feature_iteration_full(features):

    numbers, functions, forms1 = features
    numbers_new, functions_new, forms1_new = [],[],[]

    ### 0 operations
    # Function product
    function_products = np.char.add(np.repeat(functions[:,np.newaxis], functions.shape, axis=1),
                                    np.repeat(functions[:,np.newaxis], functions.shape, axis=1).T)
    functions_new.append(function_products[np.triu_indices(functions.shape[0])].T)

    # D0
    d0 = np.char.add(np.char.add('d_0(', functions),')')
    forms1_new.append(d0)

    ### 01 operations
    # Wedge product
    W01 = np.char.add(np.repeat(functions[:,np.newaxis], forms1.shape, axis=1).T,
                      np.repeat(forms1[:,np.newaxis], functions.shape, axis=1))
    forms1_new.append(W01.flatten())

    # Hessian - only upper triangle because it's symmetric
    form_pairs = np.char.add(np.repeat(np.char.add(np.char.add('(', forms1), ', ')[:,np.newaxis], forms1.shape, axis = 1),
                             np.repeat(np.char.add(forms1, ')')[:,np.newaxis], forms1.shape, axis = 1).T)[np.triu_indices(forms1.shape[0])]
    Hess = np.char.add(np.repeat(np.char.add(np.char.add('H(', functions), ')')[:,np.newaxis], form_pairs.shape, axis = 1).T,
                       np.repeat(form_pairs[:,np.newaxis], functions.shape, axis = 1))
    functions_new.append(Hess.flatten())


    ### 1 operations
    # Metric - only upper triangle because it's symmetric
    g1 = np.char.add(np.repeat(np.char.add(np.char.add('g(', forms1), ', ')[:,np.newaxis], forms1.shape, axis = 1),
                     np.repeat(np.char.add(forms1, ')')[:,np.newaxis], forms1.shape, axis = 1).T)
    functions_new.append(g1[np.triu_indices(forms1.shape[0])].T)

    # Inner product - only upper triangle because it's symmetric
    G1 = np.char.add(np.repeat(np.char.add(np.char.add('G1(', forms1), ', ')[:,np.newaxis], forms1.shape, axis = 1),
                     np.repeat(np.char.add(forms1, ')')[:,np.newaxis], forms1.shape, axis = 1).T)
    numbers_new.append(G1[np.triu_indices(forms1.shape[0])].T)
    
    # Codifferential
    d0_star = np.char.add(np.char.add('d_0^*(', forms1),')')
    functions_new.append(d0_star)

    # Lie bracket - only upper triangle because it's antisymmetric
    LB = np.char.add(np.repeat(np.char.add(np.char.add('[', forms1), ', ')[:,np.newaxis], forms1.shape, axis = 1),
                     np.repeat(np.char.add(forms1, ']')[:,np.newaxis], forms1.shape, axis = 1).T)
    forms1_new.append(LB[np.triu_indices(forms1.shape[0])].T)

    # Levi-Civita connection - only upper triangle because 
    # it's torsion-free and we already have the Lie bracket
    LCC = np.char.add(np.repeat(np.char.add(np.char.add('nabla_', forms1), ' (')[:,np.newaxis], forms1.shape, axis = 1),
                      np.repeat(np.char.add(forms1, ')')[:,np.newaxis], forms1.shape, axis = 1).T)
    forms1_new.append(LCC[np.triu_indices(forms1.shape[0])].T)

    # Hodge decomposition
    # grad = np.char.add(np.char.add('[gradient_part](', forms1),')')
    # forms1_new.append(grad)

    # if limits['use curvature']:
    #     # print('computing RCT in frame')
    #     # Riemann curvature tensor - only upper triangle because it's antisymmetric
    #     # RCT_frame = DG_class.Riemann()[np.triu_indices(limits['Riemann curvature of eigenfunction frame'])].reshape(-1,n1*n2).T
    #     # forms1_new.append(RCT_frame)
    #     # print(RCT_frame.shape)

    #     # print('computing RCT for eigenforms')
    #     k = limits['Riemann curvature of eigenforms']
    #     RCT = contract('in,jm,ijks->nmks', forms1[:,:k], forms1[:,:k], DG_class.Riemann())[np.triu_indices(k)].reshape(-1,n1*n2).T
    #     forms1_new.append(RCT)
    #     # print(RCT.shape)

    numbers_new.append(numbers)
    functions_new.append(functions)
    forms1_new.append(forms1)

    ### -1 operations
    # Diffusion eigenvalues at different times
    numbers_new = np.concatenate(numbers_new)
    numbers_new1 = [numbers_new]
    for t in [1,10,50]:
        numbers_new1.append(np.char.add(np.char.add('exp(-{} '.format(t), numbers_new),')'))
    numbers_new = np.concatenate(numbers_new1)

    functions_new = np.concatenate(functions_new)
    forms1_new = np.concatenate(forms1_new)
    return numbers_new, functions_new, forms1_new


def string_numbers_output(features, parameters, limits):
    n1 = parameters['n1']
    n2 = parameters['n2']

    numbers, functions, forms1 = features
    numbers_new = []

    ### 0 operations
    # Function metric - only upper triangle because it's symmetric
    G0 = np.char.add(np.repeat(np.char.add(np.char.add('G0(', functions), ', ')[:,np.newaxis], functions.shape, axis = 1),
                     np.repeat(np.char.add(functions, ')')[:,np.newaxis], functions.shape, axis = 1).T)
    numbers_new.append(G0[np.triu_indices(functions.shape[0])].flatten())

    ### 1 operations
    # Metric - only upper triangle because it's symmetric
    form_pairs = np.char.add(np.repeat(np.char.add(np.char.add('(', forms1), ', ')[:,np.newaxis], forms1.shape, axis = 1),
                             np.repeat(np.char.add(forms1, ')')[:,np.newaxis], forms1.shape, axis = 1).T)[np.triu_indices(forms1.shape[0])]
    g1 = np.char.add(np.repeat(np.char.add(np.char.add('G0(', ['phi_{}'.format(i) for i in range(limits['metric expansion'])]), ', g')[:,np.newaxis], form_pairs.shape, axis = 1).T,
                     np.repeat(form_pairs[:,np.newaxis], limits['metric expansion'], axis = 1))
    numbers_new.append(g1.flatten())

    # Inner product - only upper triangle because it's symmetric
    G1 = np.char.add(np.repeat(np.char.add(np.char.add('G1(', forms1), ', ')[:,np.newaxis], forms1.shape, axis = 1),
                     np.repeat(np.char.add(forms1, ')')[:,np.newaxis], forms1.shape, axis = 1).T)
    numbers_new.append(G1[np.triu_indices(forms1.shape[0])])

    numbers_new.append(numbers)
    
    eigenfunction_expansion = np.char.add(np.repeat(np.char.add(np.char.add('G0(', ['phi_{}'.format(i) for i in range(limits['metric expansion'])]), ', ')[:,np.newaxis], functions.shape, axis = 1),
                                          np.repeat(functions[:,np.newaxis], limits['metric expansion'], axis = 1).T)
    numbers_new.append(eigenfunction_expansion.flatten())

    form_frame_elements = np.array(['phi_{} d(phi_{})'.format(i,j) for i in range(n1) for j in range(n2)])
    form_frame_expansion = np.char.add(np.repeat(np.char.add(form_frame_elements, ' component of : ')[:,np.newaxis], forms1.shape, axis = 1),
                                            np.repeat(forms1[:,np.newaxis], form_frame_elements.shape, axis = 1).T)
    numbers_new.append(form_frame_expansion.flatten())
    
    numbers_new = np.concatenate(numbers_new)
    return numbers_new

def string_feature_vector(parameters, limits):

    ## Form the base collection of data: numbers, functions, and 1-forms.
    base_data = [np.concatenate([["lam^0_{}".format(i) for i in range(limits['eigenfunctions'])],
                ["lam^1,div_{}".format(i) for i in range(limits['1-eigenforms-div'])],
                ["lam^1,curl_{}".format(i) for i in range(limits['1-eigenforms-curl'])]]),
                np.array(["phi_{}".format(i) for i in range(limits['eigenfunctions'])]),
                np.concatenate([["(alpha^1,div_{})".format(i) for i in range(limits['1-eigenforms-div'])],
                                ["(alpha^1,curl_{})".format(i) for i in range(limits['1-eigenforms-curl'])]])]

    ## Compute arbitrary combinations of the base data using diffusion geometry operators.
    iterated_data = string_feature_iteration_full(base_data)

    return string_numbers_output(iterated_data, parameters, limits)