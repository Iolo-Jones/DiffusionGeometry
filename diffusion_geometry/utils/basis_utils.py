from __future__ import annotations
from scipy.special import comb
import numpy as np
from itertools import combinations, combinations_with_replacement
from functools import lru_cache
import math
import string
import itertools
from opt_einsum import contract


@lru_cache(maxsize=None)
def get_symmetric_basis_indices(d):
    """
    Returns the multi-indices for the symmetric (0,2)-tensor basis.

    Parameters
    ----------
    d : int
        Dimension of the underlying space.

    Returns
    -------
    idx : int array
        Array of multi-indices.
        Shape: [d * (d + 1) / 2, 2]
    """
    idx = np.array(list(combinations_with_replacement(range(d), 2)), dtype=int)
    idx.setflags(write=False)  # (C, 2)
    return idx


@lru_cache(maxsize=None)
def get_wedge_basis_indices(d, k):
    """
    Returns the multi-indices for the wedge product basis.

    Parameters
    ----------
    d : int
        Dimension of the underlying space.
    k : int
        Form degree.

    Returns
    -------
    idx : int array
        Array of multi-indices.
        Shape: [comb(d, k), k]
    """
    idx = np.array(list(combinations(range(d), k)), dtype=int)
    idx.setflags(write=False)  # (C, k)
    return idx


@lru_cache(maxsize=None)
def get_wedge_product_indices(d, k1, k2):
    """
    Indices and signs for the wedge product of a k1-form and a k2-form.

    Computes I, J, K such that dx_I ∧ dx_J = sgn * dx_K.

    Parameters
    ----------
    d : int
        Dimension.
    k1 : int
        Degree of first form.
    k2 : int
        Degree of second form.

    Returns
    -------
    target_indices : int array
        Indices of the result form (K).
        Shape: [L]
    left_indices : int array
        Indices of the first factor (I).
        Shape: [L]
    right_indices : int array
        Indices of the second factor (J).
        Shape: [L]
    signs : int8 array
        Parity signs (+1 or -1).
        Shape: [L]

    Where L = comb(d, k1+k2) * comb(k1+k2, k1).
    """
    k_total = k1 + k2
    if k_total > d:
        # Return empty arrays
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=int),
        )

    # 1. Generate all target indices K (combinations of size k_total)
    # shape (n_out, k_total)
    K_basis = get_wedge_basis_indices(d, k_total)
    n_out = K_basis.shape[0]

    # 2. For each K, we need all splits into I (size k1) and J (size k2).
    # We can generate this by taking combinations of size k1 from range(k_total)
    # These are LOCAL indices within the K array.

    # local_I: combinations of indices into K. shape (n_splits, k1)
    # n_splits = comb(k_total, k1)
    local_I = np.array(list(combinations(range(k_total), k1)))

    # Create mask to find local_J
    mask = np.ones((len(local_I), k_total), dtype=bool)
    mask[np.arange(len(local_I))[:, None], local_I] = False

    # local_J: remaining indices. shape (n_splits, k2)
    local_J = np.broadcast_to(np.arange(k_total), (len(local_I), k_total))[
        mask
    ].reshape(len(local_I), k2)

    # 3. Compute signs for these splits
    # Sum of indices in local_I determines parity relative to sorted K
    # Parity = sum(local_I) - sum(0..k1-1)
    # If parity is even, sign is +1. If odd, -1.
    parity = (local_I.sum(axis=1) - k1 * (k1 - 1) // 2) & 1
    split_signs = np.where(parity == 0, 1, -1).astype(np.int8)

    # 4. Map back to global indices.
    # We repeat K_basis for each split.
    # (n_out, n_splits, k_total)
    # But we don't need the full K values, just their RANK in the basis, which is row index.

    # Target indices repeat n_splits times for each K.
    target_indices = np.repeat(np.arange(n_out), len(local_I))  # (n_out * n_splits,)

    # Signs also repeat (tiled)
    signs = np.tile(split_signs, n_out)  # (n_out * n_splits,)

    # Now we need the ranks of I and J in the global basis.
    # I values = K_basis[:, local_I]
    # J values = K_basis[:, local_J]

    # Expand K_basis to (n_out, n_splits, k_total) roughly
    # Actually just simple indexing:
    I_vals = K_basis[:, local_I]  # (n_out, n_splits, k1)
    J_vals = K_basis[:, local_J]  # (n_out, n_splits, k2)

    # Flatten to list of combinations
    I_vals_flat = I_vals.reshape(-1, k1)
    J_vals_flat = J_vals.reshape(-1, k2)

    # Get ranks
    left_indices = lex_rank(I_vals_flat, d)
    right_indices = lex_rank(J_vals_flat, d)

    # Cache protection
    target_indices.setflags(write=False)
    left_indices.setflags(write=False)
    right_indices.setflags(write=False)
    signs.setflags(write=False)

    return target_indices, left_indices, right_indices, signs


@lru_cache(maxsize=None)
def kp1_children_and_signs(d, k):
    """
    Cached helper: for degree k, build
    - idx_k: combinations [Ck, k]
    - idx_kp1: combinations [Ckp1, k+1]
    - children: indices into idx_k where each entry is J' \\ {j'_r}. Shape [Ckp1, k+1]
    - signs: alternating signs (-1)^r for Laplace expansion. Shape [k+1]
    """
    idx_k = get_wedge_basis_indices(d, k)
    idx_kp1 = get_wedge_basis_indices(d, k + 1)
    Ckp1 = idx_kp1.shape[0]

    assert 1 <= k <= d - 1, f"Form degree k={k} must be between 1 and {d - 1}"

    signs = (-1) ** np.arange(k + 1)

    # 1. Generate child combinations.
    cols = np.arange(k + 1)
    # Create a mask where entry (r, c) is True if c != r
    mask = cols[:, np.newaxis] != cols
    # Get the column coordinates of the 'True' entries to form the indices
    indices = np.where(mask)[1].reshape(k + 1, k)
    children_combos = np.take_along_axis(
        idx_kp1[:, np.newaxis, :], indices[np.newaxis, :, :], axis=2
    )

    # 2. Find the index of each child in idx_k using binary search.
    view_dtype = np.dtype((np.void, idx_k.dtype.itemsize * k))
    idx_k_view = idx_k.view(view_dtype).ravel()
    children_combos_flat = children_combos.reshape(-1, k)
    children_view = children_combos_flat.view(view_dtype).ravel()

    children = np.searchsorted(idx_k_view, children_view).reshape(Ckp1, k + 1)

    # Mark arrays read-only to protect cached results from accidental mutation.
    idx_k.setflags(write=False)  # (Ck, k)
    idx_kp1.setflags(write=False)  # (Ck+1, k+1)
    children.setflags(write=False)  # (Ck+1, k+1)
    signs.setflags(write=False)  # (k+1,)

    return idx_k, idx_kp1, children, signs


def expand_symmetric_tensor_coeffs(
    coeffs: np.ndarray, n_coefficients: int, d: int
) -> np.ndarray:
    """Expand symmetric (0,2) tensor coefficients to the full tensor basis.

    Parameters
    ----------
    coeffs : ndarray
        Array whose last axis stores ``n1 * d_sym`` coefficients where
        ``d_sym = d * (d + 1) // 2``. The leading dimensions are treated as
        batch dimensions and are preserved in the output.
    n_coefficients : int
        Number of coefficient functions used in the geometry.
    d : int
        Ambient dimension.

    Returns
    -------
    ndarray
        Array with the same leading dimensions as ``coeffs`` and a last axis of
        length ``n1 * d * d`` containing the expanded coefficients in the full
        tensor basis.
        Shape: [..., n1 * d * d]
    """
    n1 = n_coefficients
    coeffs = np.asarray(coeffs)
    d_sym = d * (d + 1) // 2
    expected = n1 * d_sym
    assert (
        coeffs.shape[-1] == expected
    ), f"Symmetric tensor coefficients must have last dimension {expected}, got {coeffs.shape}"

    flat = coeffs.reshape((-1, n1, d_sym))
    expanded = np.zeros((flat.shape[0], n1, d * d), dtype=flat.dtype)

    sym_idx = get_symmetric_basis_indices(d)
    linear = sym_idx[:, 0] * d + sym_idx[:, 1]
    expanded[..., linear] = flat

    off_diag = sym_idx[:, 0] != sym_idx[:, 1]
    if np.any(off_diag):
        transpose_linear = sym_idx[off_diag, 1] * d + sym_idx[off_diag, 0]
        expanded[..., transpose_linear] = flat[..., off_diag]

    return expanded.reshape(coeffs.shape[:-1] + (n1 * d * d,))


def symmetrise_tensor_coeffs(
    coeffs: np.ndarray, n_coefficients: int, d: int
) -> np.ndarray:
    """
    Project full (0,2)-tensor coefficients onto the symmetric subspace.

    Parameters
    ----------
    coeffs : ndarray
        Array whose last axis stores n1 * d * d coefficients of a (0,2)-tensor.
    n_coefficients : int
        Number of coefficient functions φ_i used for the geometry (n1).
    d : int
        Ambient dimension.

    Returns
    -------
    ndarray
        Array containing coefficients of the symmetrised tensor.
        Shape: [..., n1 * d_sym]
    """
    n1 = n_coefficients

    coeffs = np.asarray(coeffs)
    expected = n1 * d * d
    assert (
        coeffs.shape[-1] == expected
    ), f"Full tensor coefficients must have last dimension {expected}, got {coeffs.shape}"

    flat = coeffs.reshape((-1, n1, d * d))
    sym_idx = get_symmetric_basis_indices(d)
    d_sym = sym_idx.shape[0]
    sym_coeffs = flat[..., sym_idx[:, 0] * d + sym_idx[:, 1]].copy()

    off_diag = sym_idx[:, 0] != sym_idx[:, 1]
    if np.any(off_diag):
        transpose_linear = sym_idx[off_diag, 1] * d + sym_idx[off_diag, 0]
        original_linear = sym_idx[off_diag, 0] * d + sym_idx[off_diag, 1]
        sym_coeffs[..., off_diag] = 0.5 * (
            flat[..., original_linear] + flat[..., transpose_linear]
        )

    return sym_coeffs.reshape(coeffs.shape[:-1] + (n1 * d_sym,))


def lex_rank(idx, n):
    """
    Compute lexicographic ranks for k-combinations in a vectorized way.

    Parameters
    ----------
    idx : array_like, shape (..., k)
        Strictly increasing multi-indices (i1 < ... < ik) drawn from {0,...,n-1}.
        Can be any shape; the last axis is treated as the combination.
    n : int
        Ambient size of the ground set.

    Returns
    -------
    ranks : ndarray, shape (...)
        Lexicographic rank(s), int64.
    """
    idx = np.asarray(idx, dtype=np.int64)
    *batch_shape, k = idx.shape
    m = int(np.prod(batch_shape)) if batch_shape else 1

    # Flatten all but last axis
    flat = idx.reshape(m, k)

    # Shifted version with -1 prepended
    prev = np.concatenate([np.full((m, 1), -1, dtype=np.int64), flat[:, :-1]], axis=1)
    r = (k - np.arange(k))[None, :]

    # Binomial contributions
    partA = comb(n - prev - 1, r, exact=False).astype(np.int64)
    partB = comb(n - flat, r, exact=False).astype(np.int64)

    ranks = (partA - partB).sum(axis=1, dtype=np.int64)

    # Reshape back to batch shape
    return ranks.reshape(batch_shape)


@lru_cache(maxsize=None)
def _perm_tables(d: int, k: int):
    """
    Cached (d,k)-dependent tables for antisymmetric expansion.

    Returns
    -------
    comb_indices : int array
        All combinations i1 < … < ik.
        Shape: [C, k] where C = C(d, k)
    base_perms : int array
        All permutations of {0, …, k-1}.
        Shape: [P, k] where P = k!
    signs : int8 array
        Parity signs (+1/-1) of base_perms.
        Shape: [P]
    """
    comb_indices = np.array(list(itertools.combinations(range(d), k)), dtype=np.int64)
    base_perms = np.array(
        list(itertools.permutations(range(k))), dtype=np.int64
    )  # (P,k)

    # Vectorised permutation parity: sign = (-1)^{#inversions}
    # inversions = sum_{i<j} [p[i] > p[j]]
    P = base_perms.shape[0]
    if k <= 1:
        signs = np.ones((P,), dtype=np.int8)
    else:
        cmp = base_perms[:, :, None] > base_perms[:, None, :]  # (P,k,k) boolean
        tri = np.triu_indices(k, k=1)
        inv = np.sum(cmp[:, tri[0], tri[1]], axis=1)  # (P,)
        signs = np.where((inv & 1) == 0, 1, -1).astype(np.int8)

    return comb_indices, base_perms, signs


def form_to_ambient_polyvector(form) -> np.ndarray:
    """
    Convert k-form coefficients in the wedge basis into an ambient antisymmetric
    polyvector field by acting on the ambient coordinates with the carré du champ Γ.

    Parameters
    ----------
    form : Form
        Form object to convert.
    """
    k = form.degree
    d = form.dg.dim
    n = form.dg.n
    data = form.to_pointwise_basis().reshape(form.dg.n, -1)  # (n, C)
    gamma = np.asarray(form.dg.cache.gamma_ambient)  # (n, d_ambient, d)

    # k = 0: scalar field, nothing to raise
    if k == 0:
        assert data.shape == (
            n,
        ), f"Expected shape (n,)=({n},) for k=0, got {data.shape}."
        return data

    # Build the fully antisymmetric tensor of shape (n, d, ..., d)
    if k == 1:
        # 1-forms already expanded as covariant components (n, d)
        assert data.shape == (
            n,
            d,
        ), f"Expected shape (n,d)=({n},{d}) for k=1, got {data.shape}."
        expanded = data
    else:
        # k >= 2: data is (n, C) with C = comb(d,k). Expand into anti-symmetric tensor.
        expected_C = math.comb(d, k)
        assert data.shape == (
            n,
            expected_C,
        ), f"Expected shape (n, comb(d,k)) = ({n}, {expected_C}) for k={k}, got {data.shape}."

        comb_indices, base_perms, signs = _perm_tables(d, k)  # (C,k), (P,k), (P,)
        C, P = comb_indices.shape[0], base_perms.shape[0]

        # All permuted multi-indices for all combinations: (C,P,k) -> reshape to (C*P, k)
        target_indices = comb_indices[:, base_perms].reshape(C * P, k)  # (C*P, k)

        # Values for those targets: repeat coefficients per P permutations and multiply
        # by the sign of each permutation. data: (n, C) -> (n, C*P)
        signs_row = np.broadcast_to(signs, (C, P)).reshape(C * P)  # (C*P,)
        values = (data.repeat(P, axis=1) * signs_row[None, :]).astype(
            data.dtype
        )  # (n, C*P)

        # Scatter into expanded tensor using flat advanced indexing
        expanded = np.zeros((n,) + (d,) * k, dtype=data.dtype)
        # Build flattened indices for assignment
        batch_idx = np.repeat(np.arange(n, dtype=np.int64), C * P)  # (n*C*P,)
        axis_idxs = [
            np.tile(target_indices[:, t], n) for t in range(k)
        ]  # each (n*C*P,)
        expanded[(batch_idx, *axis_idxs)] = values.reshape(n * C * P)

    # --- Raise all k indices with a single einsum ---
    # Build subscripts: for k=3 -> "...ia,...jb,...kc,...abc->...ijk"
    letters = string.ascii_lowercase
    if 2 * k > len(letters):
        raise ValueError(f"Degree k={k} too large for einsum subscript construction.")
    src = letters[:k]  # a,b,c,...
    tgt = letters[k : 2 * k]  # i,j,k,...

    metric_terms = [f"...{t}{s}" for s, t in zip(src, tgt)]

    # Raising indices: Contract k copies of the carré du champ Γ against the form components.
    # Example (k=2): T^{ij} = Γ^{ia} Γ^{jb} T_{ab}
    # Dynamic subscript construction: "...ia,...jb,...ab->...ij"
    einsum_str = f"{','.join(metric_terms)},...{src}->...{tgt}"

    operands = [gamma] * k + [expanded]
    ambient = contract(einsum_str, *operands)  # (n, d_ambient, ..., d_ambient)

    return ambient
