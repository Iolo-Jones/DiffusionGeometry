import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import diags, coo_matrix, issparse
from scipy.sparse.linalg import eigsh


def knn_graph(data_matrix: np.ndarray, knn_kernel: int = 32):
    """
    Computes k-nearest neighbor graph.

    Parameters
    ----------
    data_matrix : array
        Input data coordinates.
        Shape: [n, d]
    knn_kernel : int
        Number of nearest neighbors for the graph.

    Returns
    -------
    nbr_distances : array
        Distances to neighbors.
        Shape: [n, k]
    nbr_indices : array
        Indices of neighbors.
        Shape: [n, k]
    """
    nbrs = NearestNeighbors(n_neighbors=knn_kernel,
                            algorithm="auto").fit(data_matrix)
    nbr_distances, nbr_indices = nbrs.kneighbors(data_matrix)

    return nbr_distances, nbr_indices


def compute_local_bandwidths(
    nbr_distances: np.ndarray, k_bandwidth: int = 8, bandwidth_type: str = "l2"
):
    """
    Computes local bandwidths from neighbor distances.

    Parameters
    ----------
    nbr_distances : array
        Distances to neighbors.
        Shape: [n, k]
    k_bandwidth : int
        Number of neighbors to use for bandwidth estimation.
    bandwidth_type : str
        Method for bandwidth estimation ('k', 'l1', 'l2').

    Returns
    -------
    local_bandwidths : array
        Estimated local bandwidths.
        Shape: [n]
    """

    if bandwidth_type == "k":
        # Use the distance to the k_bandwidth-th neighbor
        local_bandwidths = nbr_distances[:, k_bandwidth]
    elif bandwidth_type == "l1":
        # Mean distance to the first k_bandwidth neighbors
        local_bandwidths = nbr_distances[:, 1:k_bandwidth].mean(axis=1)
    elif bandwidth_type == "l2":
        # Root mean square distance to the first k_bandwidth neighbors
        local_bandwidths = np.sqrt(
            (nbr_distances[:, 1:k_bandwidth] ** 2).mean(axis=1))
    else:
        raise ValueError(f"Unknown bandwidth_type: {bandwidth_type}")

    return local_bandwidths


def tune_kernel(kernel_entries: np.ndarray, epsilons: np.ndarray):
    """
    Tunes the epsilon parameter for the kernel.

    Parameters
    ----------
    kernel_entries : array
        Precomputed kernel entries (dist² / bw).
        Shape: [n, k]
    epsilons : array
        Range of epsilon values to test.
        Shape: [e]

    Returns
    -------
    epsilon : float
        Optimal epsilon value.
    dim : float
        Estimated intrinsic dimension.
    """

    test_kernel = np.exp(-kernel_entries[:,
                         :, None] / (epsilons[None, None, :]))
    test_kernel_avg = test_kernel.mean(axis=(0, 1))

    criterion = np.diff(np.log(test_kernel_avg)) / np.diff(np.log(epsilons))
    max_index = np.argmax(criterion)

    epsilon = epsilons[max_index]
    dim = 2 * criterion[max_index]

    return epsilon, dim


def markov_chain(
    nbr_distances,
    nbr_indices,
    c=0,
    bandwidth_variability=-0.5,
    knn_bandwidth=8,
    fixed_bandwidth=None,
):
    """
    Computes the diffusion kernel using a specific Markov chain construction.
    K_α(x, y) = k(x, y) / (q(x)^α q(y)^α)

    Parameters
    ----------
    nbr_distances : array
        Distances to neighbors.
        Shape: [n, k]
    nbr_indices : array
        Indices of neighbors.
        Shape: [n, k]
    c : float
        Parameter in the estimated weighted Laplacian Δ(f) + c ∇(f) ⋅ ∇(q)/q.
    bandwidth_variability : float
        Variable bandwidth exponent q^bandwidth_variability.
        0 for fixed (standard Gaussian), -0.5 for variable (Diffusion Maps).
    knn_bandwidth : int
        Number of NNs for bandwidth estimation.
    fixed_bandwidth : float, optional
        Explicit Gaussian length scale. When provided, automatic bandwidth
        estimation and tuning are skipped.

    Returns
    -------
    diffusion_kernel : array
        Row-stochastic diffusion kernel matrix.
        Shape: [n, k]
    bandwidths : array
        Estimated local bandwidths ρ used for the kernel.
        Shape: [n]
    """

    n, knn_kernel = nbr_distances.shape
    assert nbr_indices.shape == (n, knn_kernel)

    if fixed_bandwidth is not None:
        try:
            fixed_bandwidth = float(fixed_bandwidth)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "fixed_bandwidth must be a positive finite scalar"
            ) from exc
        if not np.isfinite(fixed_bandwidth) or fixed_bandwidth <= 0:
            raise ValueError("fixed_bandwidth must be a positive finite scalar")

        kernel = np.exp(-(nbr_distances**2) / fixed_bandwidth**2)
        density_estimate = kernel.sum(axis=1)
        alpha = 1 - c / 2
        density_estimate_alpha = density_estimate**(-alpha)
        kernel_alpha = kernel * (
            density_estimate_alpha[:, None]
            * density_estimate_alpha[nbr_indices]
        )
        row_sums = kernel_alpha.sum(axis=1)
        diffusion_kernel = kernel_alpha / row_sums[:, None]
        bandwidths = np.full(n, fixed_bandwidth**2 / 4)

        return diffusion_kernel, bandwidths

    # 1. Compute the kernel bandwidths rho via kernel density estimation.

    # Define and tune bandwidths for the density estimation kernel.
    bandwidths_A = compute_local_bandwidths(
        nbr_distances, knn_bandwidth, "l2")  # [n]
    kernel_entries_A = nbr_distances**2 / (
        bandwidths_A[nbr_indices] * bandwidths_A[:, None]
    )
    epsilons = 2 ** np.arange(-10, 10, 0.25)
    epsilon_A, dim_A = tune_kernel(kernel_entries_A, epsilons)

    # Compute the kernel density estimate q0 for q.
    kernel_A = np.exp(-kernel_entries_A / epsilon_A) / (
        (np.pi * epsilon_A) ** (dim_A / 2)
    )  # [n, knn_kernel]
    density_estimate_A = kernel_A.sum(
        axis=1) / (n * bandwidths_A**dim_A)  # [n]

    # Define and tune the bandwidths bw with density_estimate_A.
    bandwidths_B = density_estimate_A**bandwidth_variability  # [n]
    bandwidths_B /= np.median(bandwidths_B)
    kernel_entries_B = nbr_distances**2 / (
        bandwidths_B[nbr_indices] * bandwidths_B[:, None]
    )  # [n, knn_kernel]
    epsilon_B, dim_B = tune_kernel(kernel_entries_B, epsilons)

    # 2. Compute the kernel K using the bandwidths bw.

    # Compute K, and an improved kernel density estimate q1.
    kernel_B = np.exp(-kernel_entries_B / epsilon_B)  # [n, knn_kernel]
    density_estimate_B = kernel_B.sum(axis=1) / (bandwidths_B**dim_B)  # [n]

    # Normalise K with the 'alpha' normalisation, in terms of bandwidth_variability and c.
    alpha = 1 - c / 2 + bandwidth_variability * (dim_B / 2 + 1)
    density_estimate_alpha = density_estimate_B ** (-alpha)  # [n]
    kernel_alpha = kernel_B * (
        density_estimate_alpha[:, None] * density_estimate_alpha[nbr_indices]
    )  # [n, k]

    # 3. Compute the Markov chain from K.
    row_sums = kernel_alpha.sum(axis=1)  # [n]
    diffusion_kernel = kernel_alpha / row_sums[:, None]  # [n, k]

    # 4. Compute the bandwidths.
    bandwidths = (epsilon_B * bandwidths_B**2) / 4  # [n]

    return diffusion_kernel, bandwidths


def build_symmetric_kernel_matrix(diffusion_kernel, nbr_indices):
    """
    Constructs a symmetric sparse kernel matrix from the row-stochastic diffusion kernel.

    Parameters
    ----------
    diffusion_kernel : array
        Row-stochastic diffusion kernel matrix.
        Shape: [n, k]
    nbr_indices : array
        Indices of neighbors.
        Shape: [n, k]

    Returns
    -------
    K : sparse matrix
        Symmetric kernel matrix.
        Shape: [n, n]
    row_sums : array
        Row sums of the symmetric matrix.
        Shape: [n]
    """

    n, k = diffusion_kernel.shape

    # Form a sparse kernel matrix K
    K = coo_matrix(
        (
            diffusion_kernel.flatten(),
            (np.repeat(np.arange(n), k), nbr_indices.flatten()),
        ),
        shape=(n, n),
    )
    K = (K + K.T) / 2
    row_sums = K.sum(axis=0).A[0]

    return K, row_sums


def compute_eigenfunction_basis(
    symmetric_kernel_matrix,
    row_sums,
    n0=40,
):
    """
    Computes the functional basis {φ_i} for the diffusion geometry.

    Here we use the Diffusion Maps eigenfunctions as our choice of basis,
    but the geometry can support other function bases.

    Parameters
    ----------
    symmetric_kernel_matrix : sparse matrix
        Symmetric kernel matrix K.
        Shape: [n, n]
    row_sums : array
        Row sums of K, representing the unnormalised measure μ.
        Shape: [n]
    n0 : int
        Number of coefficient functions to compute.

    Returns
    -------
    u : array
        The basis of coefficient functions {φ_i}.
        Shape: [n, n0]
        Ordered by increasing eigenvalue (complexity). φ_0 is constant.
    """

    # Compute eigenfunctions of K via symmetric normalisation.
    sym_normalisation = diags(row_sums ** (-1 / 2))
    K_sym = sym_normalisation @ symmetric_kernel_matrix @ sym_normalisation
    n = K_sym.shape[0]
    n0_eff = int(min(max(1, n0), n))

    if n0_eff < n:
        _, eigenfunctions = eigsh(K_sym, n0_eff, which="LM", tol=1e-2)
    else:
        # eigsh requires k < n; for full bases, use dense Hermitian eigendecomposition.
        K_sym_dense = K_sym.toarray() if issparse(K_sym) else np.asarray(K_sym)
        eigenvalues, eigenvectors = np.linalg.eigh(K_sym_dense)
        top_idx = np.argsort(np.abs(eigenvalues))[-n0_eff:]
        eigenfunctions = eigenvectors[:, top_idx[::-1]]

    u = sym_normalisation @ eigenfunctions

    # Reorder in increasing complexity and normalise so φ_0 = 1.0.
    u = u[:, ::-1]
    u /= u[0, 0]

    return u
