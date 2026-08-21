import numpy as np
import pytest

import scipy.sparse as sp

from diffusion_geometry.core import DiffusionGeometry, knn_graph, markov_chain
import diffusion_geometry.core.diffusion.diffusion_process as diffusion_process


def _build_knn_inputs(n=96, d=4, knn_kernel=24, knn_bandwidth=8):
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n, d))
    nbr_distances, nbr_indices = knn_graph(data_matrix=data, knn_kernel=knn_kernel)
    kernel, bandwidths = markov_chain(
        nbr_distances=nbr_distances,
        nbr_indices=nbr_indices,
        knn_bandwidth=knn_bandwidth,
    )
    return data, nbr_distances, nbr_indices, kernel, bandwidths


def _assert_valid_geometry(dg, data_shape):
    n, d = data_shape
    assert dg.n == n
    assert dg.dim == d
    assert dg.immersion_coords.shape == (n, d)
    assert dg.measure.shape == (n,)
    assert dg.function_basis.shape[0] == n
    assert np.isfinite(dg.immersion_coords).all()
    assert np.isfinite(dg.measure).all()
    assert np.isfinite(dg.function_basis).all()


def test_markov_chain_fixed_bandwidth_uses_requested_gaussian_scale(monkeypatch):
    nbr_distances = np.array([[0.0, 1.0], [0.0, 2.0]])
    nbr_indices = np.array([[0, 1], [1, 0]])
    fixed_bandwidth = 2.0
    c = 0.5

    monkeypatch.setattr(
        diffusion_process,
        "compute_local_bandwidths",
        lambda *_args, **_kwargs: pytest.fail("automatic bandwidth estimation ran"),
    )
    monkeypatch.setattr(
        diffusion_process,
        "tune_kernel",
        lambda *_args, **_kwargs: pytest.fail("automatic bandwidth tuning ran"),
    )

    kernel, bandwidths = markov_chain(
        nbr_distances=nbr_distances,
        nbr_indices=nbr_indices,
        c=c,
        fixed_bandwidth=fixed_bandwidth,
    )

    unnormalised = np.exp(-(nbr_distances**2) / fixed_bandwidth**2)
    density = unnormalised.sum(axis=1)
    alpha = 1 - c / 2
    expected = unnormalised * (
        density[:, None] ** (-alpha) * density[nbr_indices] ** (-alpha)
    )
    expected /= expected.sum(axis=1, keepdims=True)

    np.testing.assert_allclose(kernel, expected)
    np.testing.assert_allclose(bandwidths, fixed_bandwidth**2 / 4)


def test_markov_chain_explicit_none_matches_automatic_default():
    _, nbr_distances, nbr_indices, _, _ = _build_knn_inputs(
        n=32, d=3, knn_kernel=12, knn_bandwidth=4
    )

    default_kernel, default_bandwidths = markov_chain(
        nbr_distances=nbr_distances,
        nbr_indices=nbr_indices,
    )
    none_kernel, none_bandwidths = markov_chain(
        nbr_distances=nbr_distances,
        nbr_indices=nbr_indices,
        fixed_bandwidth=None,
    )

    np.testing.assert_allclose(none_kernel, default_kernel)
    np.testing.assert_allclose(none_bandwidths, default_bandwidths)


@pytest.mark.parametrize("fixed_bandwidth", [0, -1, np.nan, np.inf])
def test_markov_chain_rejects_invalid_fixed_bandwidth(fixed_bandwidth):
    nbr_distances = np.array([[0.0, 1.0], [0.0, 2.0]])
    nbr_indices = np.array([[0, 1], [1, 0]])

    with pytest.raises(ValueError, match="fixed_bandwidth must be a positive finite scalar"):
        markov_chain(
            nbr_distances=nbr_distances,
            nbr_indices=nbr_indices,
            fixed_bandwidth=fixed_bandwidth,
        )


@pytest.mark.parametrize("constructor", ["point_cloud", "knn_graph"])
def test_constructors_forward_fixed_bandwidth(constructor):
    data, nbr_distances, nbr_indices, _, _ = _build_knn_inputs(
        n=32, d=3, knn_kernel=12, knn_bandwidth=4
    )
    fixed_bandwidth = 0.2

    if constructor == "point_cloud":
        dg = DiffusionGeometry.from_point_cloud(
            data_matrix=data,
            fixed_bandwidth=fixed_bandwidth,
            knn_kernel=12,
            n_function_basis=10,
        )
    else:
        dg = DiffusionGeometry.from_knn_graph(
            nbr_indices=nbr_indices,
            nbr_distances=nbr_distances,
            fixed_bandwidth=fixed_bandwidth,
            data_matrix=data,
            n_function_basis=10,
        )

    np.testing.assert_allclose(
        dg.triple._cdc.keywords["bandwidths"], fixed_bandwidth**2 / 4
    )


@pytest.mark.parametrize("regularisation_method", ["diffusion", "bandlimit", "none"])
def test_from_point_cloud_regularisation_modes_construct(regularisation_method):
    data, _, _, _, _ = _build_knn_inputs()
    dg = DiffusionGeometry.from_point_cloud(
        data_matrix=data,
        regularisation_method=regularisation_method,
        knn_kernel=24,
        knn_bandwidth=8,
        n_function_basis=20,
    )
    _assert_valid_geometry(dg, data.shape)

    probe = np.arange(data.size, dtype=float).reshape(data.shape)
    regularised = dg._regularise(probe)
    assert regularised.shape == probe.shape
    if regularisation_method == "none":
        assert np.allclose(regularised, probe)


@pytest.mark.parametrize("regularisation_method", ["diffusion", "bandlimit", "none"])
def test_from_knn_graph_regularisation_modes_construct(regularisation_method):
    data, nbr_distances, nbr_indices, _, _ = _build_knn_inputs()
    dg = DiffusionGeometry.from_knn_graph(
        nbr_indices=nbr_indices,
        nbr_distances=nbr_distances,
        data_matrix=data,
        regularisation_method=regularisation_method,
        knn_bandwidth=8,
        n_function_basis=20,
    )
    _assert_valid_geometry(dg, data.shape)


@pytest.mark.parametrize("regularisation_method", ["diffusion", "bandlimit", "none"])
def test_from_knn_kernel_regularisation_modes_construct(regularisation_method):
    data, _, nbr_indices, kernel, bandwidths = _build_knn_inputs()
    dg = DiffusionGeometry.from_knn_kernel(
        nbr_indices=nbr_indices,
        kernel=kernel,
        bandwidths=bandwidths,
        immersion_coords=None,
        data_matrix=data,
        regularisation_method=regularisation_method,
        n_function_basis=20,
    )
    _assert_valid_geometry(dg, data.shape)


def test_from_knn_kernel_requires_data_or_immersion_for_none_regularisation():
    _, _, nbr_indices, kernel, bandwidths = _build_knn_inputs()
    with pytest.raises(ValueError, match="data_matrix and/or immersion_coords"):
        DiffusionGeometry.from_knn_kernel(
            nbr_indices=nbr_indices,
            kernel=kernel,
            bandwidths=bandwidths,
            immersion_coords=None,
            regularisation_method="none",
            n_function_basis=20,
        )


def test_from_graph_kernel_respects_n_coefficients():
    n = 9
    sources = np.arange(n)
    targets = np.roll(sources, -1)
    edge_index = np.vstack(
        [np.concatenate([sources, targets]), np.concatenate([targets, sources])]
    )
    kernel = np.ones(edge_index.shape[1], dtype=float)
    rng = np.random.default_rng(1)
    immersion_coords = rng.standard_normal((n, 2))

    dg = DiffusionGeometry.from_graph_kernel(
        edge_index=edge_index,
        kernel=kernel,
        immersion_coords=immersion_coords,
        n_coefficients=3,
    )

    assert dg.n == n
    assert dg.n_function_basis == n
    assert dg.n_coefficients == 3


def test_from_graph_kernel_default_measure_uses_degree_weights():
    n = 4
    edge_index = np.array(
        [
            [0, 0, 1, 2, 2, 2, 3],
            [1, 2, 2, 0, 1, 3, 0],
        ]
    )
    kernel = np.array([1.0, 3.0, 2.0, 4.0, 5.0, 1.0, 2.0])
    immersion_coords = np.arange(n)[:, None].astype(float)

    dg = DiffusionGeometry.from_graph_kernel(
        edge_index=edge_index,
        kernel=kernel,
        immersion_coords=immersion_coords,
    )

    expected = np.bincount(edge_index[0], weights=kernel, minlength=n)
    expected = expected / expected.sum()

    np.testing.assert_allclose(dg.measure, expected)
    np.testing.assert_allclose(dg.measure.sum(), 1.0)


def test_from_edges_respects_n_coefficients():
    n = 10
    sources = np.arange(n)
    targets = np.roll(sources, -1)
    edge_index = np.vstack([sources, targets])
    rng = np.random.default_rng(2)
    immersion_coords = rng.standard_normal((n, 3))

    dg = DiffusionGeometry.from_edges(
        edge_index=edge_index,
        immersion_coords=immersion_coords,
        n_coefficients=4,
    )

    assert dg.n == n
    assert dg.n_function_basis == n
    assert dg.n_coefficients == 4


def test_from_sparse_matrix_constructs_correct_geometry():
    n = 10
    sources = np.arange(n)
    targets = np.roll(sources, -1)
    weights = np.random.rand(n)

    # Create a sparse CSR matrix
    sparse_matrix = sp.csr_matrix((weights, (sources, targets)), shape=(n, n))

    rng = np.random.default_rng(3)
    immersion_coords = rng.standard_normal((n, 3))

    dg = DiffusionGeometry.from_sparse_matrix(
        sparse_matrix=sparse_matrix,
        immersion_coords=immersion_coords,
        n_coefficients=5,
    )

    assert dg.n == n
    assert dg.n_function_basis == n
    assert dg.n_coefficients == 5
    assert dg.immersion_coords.shape == (n, 3)

    # Check that from_sparse_matrix delegates correctly by converting back to from_graph_kernel's inputs
    coo = sparse_matrix.tocoo()
    expected_edge_index = np.vstack((coo.col, coo.row))
    expected_kernel = coo.data

    dg_graph_kernel = DiffusionGeometry.from_graph_kernel(
        edge_index=expected_edge_index,
        kernel=expected_kernel,
        immersion_coords=immersion_coords,
        n_coefficients=5,
    )

    assert dg.n == dg_graph_kernel.n
    assert dg.n_coefficients == dg_graph_kernel.n_coefficients
    assert np.allclose(dg.immersion_coords, dg_graph_kernel.immersion_coords)


def test_graph_constructor_carre_du_champ_normalisation_known_example():
    # Known row-stochastic kernel K[i, j] (rows sum to 1).
    kernel_dense = np.array(
        [
            [0.8, 0.2, 0.0],
            [0.1, 0.6, 0.3],
            [0.0, 0.4, 0.6],
        ],
        dtype=float,
    )
    # Linear function over nodes.
    f = np.array([0.0, 1.0, 2.0], dtype=float)[:, None]
    immersion_coords = np.arange(3, dtype=float)[:, None]

    # Ground truth Γ(f,f)(i) = 1/2 * sum_j K[i, j] * (f_j - f_i)^2
    expected = 0.5 * np.array(
        [
            0.8 * (0.0 - 0.0) ** 2 + 0.2 * (1.0 - 0.0) ** 2,
            0.1 * (0.0 - 1.0) ** 2 + 0.6 * (1.0 - 1.0) ** 2 + 0.3 * (2.0 - 1.0) ** 2,
            0.4 * (1.0 - 2.0) ** 2 + 0.6 * (2.0 - 2.0) ** 2,
        ],
        dtype=float,
    )

    # Build canonical graph representation expected by from_graph_kernel:
    # edge_index = [source=j, target=i] with weights K[i, j].
    rows, cols = np.nonzero(kernel_dense)
    edge_index = np.vstack((cols, rows))
    weights = kernel_dense[rows, cols]

    # Sparse representation stores (row=i, col=j) = K[i, j].
    sparse_matrix = sp.csr_matrix(kernel_dense)

    constructors = {
        "from_graph_kernel": lambda: DiffusionGeometry.from_graph_kernel(
            edge_index=edge_index,
            kernel=weights,
            immersion_coords=immersion_coords,
        ),
        "from_sparse_matrix": lambda: DiffusionGeometry.from_sparse_matrix(
            sparse_matrix=sparse_matrix,
            immersion_coords=immersion_coords,
        ),
        "from_edges": lambda: DiffusionGeometry.from_edges(
            edge_index=edge_index,
            immersion_coords=immersion_coords,
        ),
    }

    for name, build in constructors.items():
        dg = build()
        gamma_ff = dg.triple.cdc(f, f).reshape(-1)

        if name == "from_edges":
            # from_edges ignores provided weights and uses uniform incoming
            # averaging, so only the normalization identity is asserted.
            assert gamma_ff.shape == expected.shape
            assert np.all(gamma_ff >= 0)
            continue

        np.testing.assert_allclose(gamma_ff, expected, rtol=1e-12, atol=1e-12)


def test_knn_constructor_carre_du_champ_normalisation_known_example():
    # Known row-stochastic kernel K[i, j] (rows sum to 1).
    kernel_dense = np.array(
        [
            [0.8, 0.2, 0.0],
            [0.1, 0.6, 0.3],
            [0.0, 0.4, 0.6],
        ],
        dtype=float,
    )
    f = np.array([0.0, 1.0, 2.0], dtype=float)[:, None]
    immersion_coords = np.arange(3, dtype=float)[:, None]

    expected = 0.5 * np.array(
        [
            0.8 * (0.0 - 0.0) ** 2 + 0.2 * (1.0 - 0.0) ** 2,
            0.1 * (0.0 - 1.0) ** 2 + 0.6 * (1.0 - 1.0) ** 2 + 0.3 * (2.0 - 1.0) ** 2,
            0.4 * (1.0 - 2.0) ** 2 + 0.6 * (2.0 - 2.0) ** 2,
        ],
        dtype=float,
    )

    nbr_indices = np.tile(np.arange(3), (3, 1))

    dg = DiffusionGeometry.from_knn_kernel(
        nbr_indices=nbr_indices,
        kernel=kernel_dense,
        immersion_coords=immersion_coords,
        regularisation_method="none",
        n_function_basis=3,
        use_mean_centres=False,
    )

    gamma_ff = dg.triple.cdc(f, f).reshape(-1)
    np.testing.assert_allclose(gamma_ff, expected, rtol=1e-12, atol=1e-12)


def test_from_point_cloud_rejects_unknown_regularisation_method():
    data, _, _, _, _ = _build_knn_inputs()
    with pytest.raises(ValueError, match="Unknown regularisation method"):
        DiffusionGeometry.from_point_cloud(
            data_matrix=data,
            regularisation_method="invalid-method",
            knn_kernel=24,
            knn_bandwidth=8,
            n_function_basis=20,
        )


def test_from_point_cloud_small_dataset_with_default_n_function_basis():
    rng = np.random.default_rng(3)
    data = rng.standard_normal((20, 2))

    dg = DiffusionGeometry.from_point_cloud(
        data_matrix=data,
        knn_kernel=10,
        knn_bandwidth=5,
    )

    assert dg.n == 20
    assert dg.n_function_basis == 20
    assert dg.function_basis.shape == (20, 20)
