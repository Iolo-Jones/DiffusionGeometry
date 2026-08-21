from __future__ import annotations

from functools import cached_property, lru_cache, partial
from typing import Optional, Union

import numpy as np
from opt_einsum import contract

from diffusion_geometry.core import (
    carre_du_champ_graph,
    carre_du_champ_knn,
    knn_graph,
    markov_chain,
    ImmersedMarkovTriple,
    regularise_bandlimit,
    regularise_diffusion,
    SymmetricKernelConstructor,
)

from diffusion_geometry import operators, tensors

from diffusion_geometry.utils.batch_utils import compatible_batches

class DiffusionGeometry:
    """
    A diffusion geometry framework for discrete data.

    This class provides a global coordinate-free representation of differential
    operators (Laplacian, exterior derivative, Hessian, etc.) by expanding all
    objects in a basis of coefficient functions {φ_i} and the derivatives of
    the immersion/embedding coordinates.

    Given coefficient functions {φ_i} of the Laplacian (ordered by ascending
    eigenvalues), the Dirichlet energy E(φ_i) increases with i. Restricting to
    the first n_coefficients allows for a low-rank approximation of the
    geometric structures.
    """

    def __init__(
        self,
        triple: ImmersedMarkovTriple,
        **kwargs,
    ) -> None:
        """
        Initialize the DiffusionGeometry object.

        Parameters
        ----------
        triple : ImmersedMarkovTriple
            The underlying Markov triple containing the kernel, measure, function basis, and immersion coordinates.
        **kwargs : dict, optional
            Additional configuration parameters:
            - n_coefficients (int, optional): Number of coefficients for tensor expansions.
              Defaults to n_function_basis.
            - rcond (float, default=1e-5): Cutoff for spectral operations.
        """
        self.triple = triple
        self.rcond = float(kwargs.get("rcond", 1e-5))
        self.n_function_basis = self.triple.n_function_basis

        # Determine n_coefficients
        n_coefficients = kwargs.get("n_coefficients")
        self.n_coefficients = (
            int(n_coefficients) if n_coefficients is not None else self.n_function_basis
        )
        self.n_coefficients = min(self.n_coefficients, self.n_function_basis)

        # Cache for all the reused objects
        from diffusion_geometry.core import DiffusionGeometryCache

        self.cache = DiffusionGeometryCache(self.triple)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DiffusionGeometry):
            return NotImplemented
        return self is other

    def __hash__(self) -> int:
        return id(self)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def n(self) -> int:
        return self.triple.n

    @property
    def dim(self) -> int:
        return self.triple.dim

    @property
    def immersion_coords(self) -> np.ndarray:
        return self.triple.immersion_coords

    @property
    def function_basis(self) -> np.ndarray:
        return self.triple.function_basis

    @property
    def measure(self):
        return self.triple.measure

    @property
    def _regularise(self):
        return self.triple.regularise

    # -------------------------------------------------------------------------
    # Constructors from different data sources
    # -------------------------------------------------------------------------

    @classmethod
    def from_knn_kernel(
        cls,
        nbr_indices: np.ndarray,
        kernel: np.ndarray,
        immersion_coords: np.ndarray,
        **kwargs,
    ) -> "DiffusionGeometry":
        """
        Construct from a precomputed kernel defined on a neighbour graph.

        Parameters
        ----------
        nbr_indices : np.ndarray
            Indices of the k-nearest neighbours for each point.
        kernel : np.ndarray
            The kernel weights for the edges defined by nbr_indices.
        immersion_coords : np.ndarray
            Coordinates of the data points.
        **kwargs : dict, optional
            Additional configuration parameters:
            - bandwidths (np.ndarray, optional): Bandwidths for the kernel.
            - n_function_basis (int, default=50): Number of coefficient functions to compute.
            - n_coefficients (int, optional): Number of coefficients for tensor expansions.
            - regularisation_method (str, default="diffusion"): Method for regularisation
              ('diffusion', 'bandlimit', or 'none').
            - rcond (float, default=1e-5): Cutoff for spectral operations.
            - measure (np.ndarray, optional): Stationary measure.
            - function_basis (np.ndarray, optional): Precomputed basis of coefficient functions.
            - use_mean_centres (bool, default=True): Whether to use mean centering for CDC.
        """
        bandwidths = kwargs.get("bandwidths")
        n_function_basis = kwargs.get("n_function_basis", 50)
        n_coefficients = kwargs.pop("n_coefficients", None)
        regularisation_method = kwargs.get("regularisation_method", "diffusion")
        rcond = kwargs.pop("rcond", 1e-5)
        measure = kwargs.get("measure")
        function_basis = kwargs.get("function_basis")
        use_mean_centres = kwargs.get("use_mean_centres", True)
        data_matrix = kwargs.get("data_matrix")

        nbr_indices = np.asarray(nbr_indices)
        kernel = np.asarray(kernel)
        bandwidths = np.asarray(bandwidths) if bandwidths is not None else None

        # Resolve any missing data using SymmetricKernelConstructor
        constructor = SymmetricKernelConstructor(
            nbr_indices=nbr_indices,
            kernel=kernel,
        )
        measure = constructor.resolve_measure(measure)
        function_basis = constructor.resolve_function_basis(
            n_function_basis, function_basis
        )

        # Define regularisation
        if regularisation_method == "bandlimit":
            regularise = partial(
                regularise_bandlimit, u=function_basis, measure=measure
            )
        elif regularisation_method == "diffusion":
            regularise = partial(
                regularise_diffusion, kernel=kernel, nbr_indices=nbr_indices
            )
        elif regularisation_method == "none":
            regularise = None
        else:
            raise ValueError(f"Unknown regularisation method: {regularisation_method}")

        immersion_regularise = regularise if regularise is not None else np.asarray
        immersion_coords = constructor.resolve_immersion(
            immersion_regularise, data_matrix, immersion_coords
        )

        # Define carré du champ
        cdc = partial(
            carre_du_champ_knn,
            diffusion_kernel=kernel,
            nbr_indices=nbr_indices,
            bandwidths=bandwidths,
            use_mean_centres=use_mean_centres,
        )


        # Create the ImmersedMarkovTriple with all this data
        triple = ImmersedMarkovTriple(
            function_basis=function_basis,
            measure=measure,
            carre_du_champ=cdc,
            immersion_coords=immersion_coords,
            data_matrix=data_matrix,
            regularise=regularise,
        )

        return cls(
            triple=triple,
            n_coefficients=n_coefficients,
            rcond=rcond,
            **kwargs,
        )

    @classmethod
    def from_knn_graph(
        cls,
        nbr_indices: np.ndarray,
        nbr_distances: np.ndarray,
        fixed_bandwidth: Optional[float] = None,
        **kwargs,
    ) -> "DiffusionGeometry":
        """
        Construct from a pre-built neighbour graph and coordinates.

        Parameters
        ----------
        nbr_indices : np.ndarray
            Indices of the k-nearest neighbours.
        nbr_distances : np.ndarray
            Distances to the k-nearest neighbours.
        **kwargs : dict, optional
            Additional configuration parameters:
            - immersion_coords (np.ndarray, optional): Coordinates of the data points.
            - n_function_basis (int, default=50): Number of coefficient functions.
            - n_coefficients (int, optional): Number of coefficients.
            - regularisation_method (str, default='diffusion'): Regularisation method.
            - c (float, default=0.0): Parameter for Markov chain construction.
            - bandwidth_variability (float, default=-0.5): Parameter for bandwidth variability.
            - knn_bandwidth (int, default=8): Number of neighbours for bandwidth estimation.
            - fixed_bandwidth (float, optional): Explicit Gaussian length scale.
            - rcond (float, default=1e-5): Cutoff for spectral operations.
            - measure (np.ndarray, optional): Stationary measure.
            - function_basis (np.ndarray, optional): Precomputed basis of coefficient functions.
            - use_mean_centres (bool, default=True): Whether to use mean centering.
        """
        kernel, bandwidths = markov_chain(
            nbr_distances=nbr_distances,
            nbr_indices=nbr_indices,
            c=kwargs.get("c", 0.0),
            bandwidth_variability=kwargs.get("bandwidth_variability", -0.5),
            knn_bandwidth=kwargs.get("knn_bandwidth", 8),
            fixed_bandwidth=fixed_bandwidth,
        )
        immersion_coords = kwargs.pop("immersion_coords", None)

        return cls.from_knn_kernel(
            nbr_indices=nbr_indices,
            kernel=kernel,
            bandwidths=bandwidths,
            immersion_coords=immersion_coords,
            **kwargs,
        )

    @classmethod
    def from_point_cloud(
        cls,
        data_matrix: np.ndarray,
        fixed_bandwidth: Optional[float] = None,
        **kwargs,
    ) -> "DiffusionGeometry":
        """
        Construct a diffusion geometry instance directly from raw data.

        Parameters
        ----------
        data_matrix : np.ndarray
            The input data matrix (n_samples, n_features).
        **kwargs : dict, optional
            Additional configuration parameters:
            - n_function_basis (int, default=50): Number of coefficient functions.
            - n_coefficients (int, optional): Number of coefficients.
            - immersion_coords (np.ndarray, optional): Explicit coordinates if different from data.
            - knn_kernel (int, default=32): Number of neighbours for graph construction.
            - c (float, default=0.0): Parameter for Markov chain construction.
            - bandwidth_variability (float, default=-0.5): Parameter for bandwidth variability.
            - knn_bandwidth (int, default=8): Number of neighbours for bandwidth estimation.
            - fixed_bandwidth (float, optional): Explicit Gaussian length scale.
            - regularisation_method (str, default='diffusion'): Regularisation method.
            - rcond (float, default=1e-5): Cutoff for spectral operations.
            - measure (np.ndarray, optional): Stationary measure.
            - function_basis (np.ndarray, optional): Precomputed basis of coefficient functions.
            - use_mean_centres (bool, default=True): Whether to use mean centering.
        """
        nbr_distances, nbr_indices = knn_graph(
            data_matrix=data_matrix, knn_kernel=kwargs.get("knn_kernel", 32)
        )
        kwargs["data_matrix"] = data_matrix

        return cls.from_knn_graph(
            nbr_indices=nbr_indices,
            nbr_distances=nbr_distances,
            fixed_bandwidth=fixed_bandwidth,
            **kwargs,
        )

    @classmethod
    def from_sparse_matrix(
        cls,
        sparse_matrix,
        immersion_coords: np.ndarray,
        **kwargs,
    ) -> "DiffusionGeometry":
        """
        Construct from a scipy.sparse matrix representing a graph.

        This method acts as a bridge for graph libraries that output
        scipy.sparse matrices (like MAGIC or graphtools).

        Parameters
        ----------
        sparse_matrix : scipy.sparse matrix
            The sparse matrix representing the graph kernel/adjacency.
        immersion_coords : np.ndarray
            Coordinates of the nodes.
        **kwargs : dict, optional
            Additional configuration parameters passed to from_graph_kernel.
        """
        coo = sparse_matrix.tocoo()
        # Sparse kernels are typically stored as K[i, j] where i is the point
        # at which local expectations are evaluated and j is the neighbour.
        # The graph carré du champ expects edges encoded as (source=j, target=i),
        # so we transpose the sparse (row, col) convention into (col, row).
        edge_index = np.vstack((coo.col, coo.row))
        kernel = coo.data

        return cls.from_graph_kernel(
            edge_index=edge_index,
            kernel=kernel,
            immersion_coords=immersion_coords,
            **kwargs,
        )

    @classmethod
    def from_graph_kernel(
        cls,
        edge_index: np.ndarray,
        kernel: np.ndarray,
        immersion_coords: np.ndarray,
        **kwargs,
    ) -> "DiffusionGeometry":
        """
        Construct from a precomputed kernel defined on an arbitrary graph.

        Parameters
        ----------
        edge_index : np.ndarray
            The edge indices of the graph (2, n_edges).
        kernel : np.ndarray
            The kernel weights for the edges.
        immersion_coords : np.ndarray
            Coordinates of the nodes.
        **kwargs : dict, optional
            Additional configuration parameters:
            - bandwidths (np.ndarray, optional): Bandwidths for the kernel.
            - rcond (float, default=1e-5): Cutoff for spectral operations.
            - mu (np.ndarray, optional): Stationary measure.
            - function_basis (np.ndarray, optional): Precomputed basis of coefficient functions.
            - use_mean_centres (bool, default=False): Whether to use mean centering.
        """
        bandwidths = kwargs.get("bandwidths")
        rcond = kwargs.pop("rcond", 1e-5)
        n_coefficients = kwargs.pop("n_coefficients", None)

        measure = kwargs.get("measure")
        function_basis = kwargs.get("function_basis")
        use_mean_centres = kwargs.get("use_mean_centres", False)

        n = immersion_coords.shape[0]
        if measure is None:
            measure = np.bincount(edge_index[0], weights=kernel, minlength=n)
            measure = measure / measure.sum()
        if function_basis is None:
            function_basis = np.eye(n)

        def cdc(f: np.ndarray, h: np.ndarray) -> np.ndarray:
            return carre_du_champ_graph(
                f,
                h,
                kernel,
                edge_index,
                bandwidths=bandwidths,
                use_mean_centres=use_mean_centres,
            )

        triple = ImmersedMarkovTriple(
            function_basis=function_basis,
            measure=measure,
            carre_du_champ=cdc,
            immersion_coords=immersion_coords,
        )

        return cls(
            triple=triple,
            n_coefficients=n_coefficients,
            rcond=rcond,
        )

    @classmethod
    def from_edges(
        cls,
        edge_index: np.ndarray,
        **kwargs,
    ) -> "DiffusionGeometry":
        """
        Construct from a graph defined by edge indices with uniform weights.

        The diffusion kernel is defined as w_{ji} = 1/d(i) where d(i) is the in-degree of i.
        The stationary measure mu is set to the in-degrees d(i).

        Parameters
        ----------
        edge_index : np.ndarray
            The edge indices of the graph (2, n_edges).
        **kwargs : dict, optional
            Additional configuration parameters:
            - immersion_coords (np.ndarray, optional): Coordinates, defaults to identity.
            - rcond (float, default=1e-5): Cutoff for spectral operations.
            - Other arguments passed to from_graph_kernel.
        """
        immersion_coords = kwargs.pop("immersion_coords", None)

        edge_index = np.asarray(edge_index)

        # Determine n and immersion_coords if necessary
        if immersion_coords is not None:
            n = immersion_coords.shape[0]
        else:
            n = int(edge_index.max()) + 1
            immersion_coords = np.eye(n)

        # Compute in-degrees (number of incoming edges for each node)
        # edge_index[1] contains the target nodes i
        target_indices = edge_index[1]
        degrees = np.bincount(target_indices, minlength=n)

        # Compute kernel weights: w_{ji} = 1 / d(i)
        # For each edge pointing to i, weight is 1/d(i)
        # Note: edges pointing to nodes with degree 0 is impossible if they appear in target_indices
        edge_weights = 1.0 / degrees[target_indices]

        # Set measure to degrees
        measure = degrees.astype(float)
        kwargs["measure"] = measure

        return cls.from_graph_kernel(
            edge_index=edge_index,
            kernel=edge_weights,
            immersion_coords=immersion_coords,
            **kwargs,
        )

    # -------------------------------------------------------------------------
    # Spaces of functions, vector fields, forms, and tensors
    # -------------------------------------------------------------------------

    @cached_property
    def function_space(self) -> tensors.FunctionSpace:
        """Space of functions."""
        return tensors.FunctionSpace(self)

    @cached_property
    def vector_field_space(self) -> tensors.VectorFieldSpace:
        """Space of vector fields."""
        return tensors.VectorFieldSpace(self)

    @lru_cache(maxsize=None)
    def form_space(
        self, degree: int
    ) -> Union[tensors.FunctionSpace, tensors.FormSpace]:
        """Space of forms."""
        if degree == 0:
            return self.function_space
        return tensors.FormSpace(self, degree)

    @cached_property
    def tensor02_space(self) -> tensors.Tensor02Space:
        """Space of general (0,2)-tensors."""
        return tensors.Tensor02Space(self)

    @cached_property
    def tensor02sym_space(self) -> tensors.Tensor02SymSpace:
        """Space of symmetric (0,2)-tensors."""
        return tensors.Tensor02SymSpace(self)

    # -------------------------------------------------------------------------
    # Riemannian metric and global inner product
    # -------------------------------------------------------------------------

    def g(self, a: "tensors.Tensor", b: "tensors.Tensor") -> np.ndarray:
        """
        Pointwise Riemannian metric of two objects in the same tensor space.

        g_p(a, b)

        Parameters
        ----------
        a, b : Tensor
            Tensors to compute the metric on. They must belong to the same space.

        Returns
        -------
        metric : np.ndarray
            The pointwise Riemannian metric g(a, b) evaluated at each point.
            Shape: [..., n]
        """
        assert a.space == b.space, f"Spaces {a.space} vs {b.space} do not match"
        assert compatible_batches(
            a.batch_shape, b.batch_shape
        ), f"Batch shapes {a.batch_shape} vs {b.batch_shape} are not compatible"
        return a.space.metric_apply(a.coeffs, b.coeffs)

    def pointwise_norm(self, a: "tensors.Tensor") -> np.ndarray:
        """
        Pointwise norm of a tensor field.

        ‖a‖_p = √(g_p(a, a))

        Parameters
        ----------
        a : Tensor
            Tensor object to compute the norm of.

        Returns
        -------
        norm : np.ndarray
            The pointwise norm ‖a‖ evaluated at each data point.
            Shape: [..., n]
        """
        pointwise_norm_squared = np.maximum(self.g(a, a), 0.0)
        return np.sqrt(pointwise_norm_squared)

    def inner(
        self, a: "tensors.Tensor", b: "tensors.Tensor"
    ) -> Union[float, np.ndarray]:
        """
        L² inner product of two tensors in the same space.

        ⟨a, b⟩ = ∫ g(a, b) dμ

        Parameters
        ----------
        a, b : Tensor
            Tensors to compute the inner product on. They must belong to the same space.

        Returns
        -------
        inner_product : float or np.ndarray
            The global L² inner product ⟨a, b⟩.
        """
        assert a.space == b.space, f"Spaces {a.space} vs {b.space} do not match"
        assert compatible_batches(
            a.batch_shape, b.batch_shape
        ), f"Batch shapes {a.batch_shape} vs {b.batch_shape} are not compatible"
        result = contract("AB,...A,...B->...", a.space.gram, a.coeffs, b.coeffs)
        if result.ndim == 0:
            return float(result)
        return result

    def norm(self, a: "tensors.Tensor") -> Union[float, np.ndarray]:
        """
        Global L² norm of a tensor field.

        ‖a‖_L² = √(⟨a, a⟩)

        Parameters
        ----------
        a : Tensor
            Tensor object to compute the norm of.

        Returns
        -------
        norm : float or np.ndarray
            The global L² norm ‖a‖.
        """
        norm_squared = np.maximum(self.inner(a, a), 0.0)
        return np.sqrt(norm_squared)

    # -------------------------------------------------------------------------
    # First-order derivatives: d, grad, codifferential, div
    # -------------------------------------------------------------------------

    @cached_property
    def grad(self) -> operators.LinearOperator:
        """
        Gradient mapping functions to vector fields.

        ∇ : A ↦ 𝔛(M)
        f ↦ ∇f
        """
        weak_matrix = operators.derivative_weak(
            self.triple.function_basis,
            self.cache.gamma_mixed,
            None,
            self.triple.measure,
            0,
            self.n_coefficients,
        )
        return operators.LinearOperator(
            domain=self.function_space,
            codomain=self.vector_field_space,
            weak_matrix=weak_matrix,
        )

    @lru_cache(maxsize=None)
    def d(self, k: int) -> operators.LinearOperator:
        """
        Exterior derivative mapping k-forms to (k+1)-forms.

        d : Ωᵏ(M) ↦ Ωᵏ⁺¹(M)
        """
        assert (
            0 <= k < self.dim
        ), f"Exterior derivative undefined for k={k} in dim {self.dim}."
        if k == 0:
            return operators.LinearOperator(
                domain=self.form_space(k),
                codomain=self.form_space(k + 1),
                weak_matrix=self.grad.weak,
            )
        _, compound_matrices = self.cache.gamma_coords_compound(k)
        weak_matrix = operators.derivative_weak(
            self.triple.function_basis,
            self.cache.gamma_mixed,
            compound_matrices,
            self.triple.measure,
            k,
            self.n_coefficients,
        )
        return operators.LinearOperator(
            domain=self.form_space(k),
            codomain=self.form_space(k + 1),
            weak_matrix=weak_matrix,
        )

    @lru_cache(maxsize=None)
    def codifferential(self, k: int) -> operators.LinearOperator:
        """
        Codifferential mapping k-forms to (k-1)-forms.

        δ : Ωᵏ(M) ↦ Ωᵏ⁻¹(M)
        δ = d*
        """
        assert 1 <= k <= self.dim, f"Codifferential undefined for degree k={k}."
        return self.d(k - 1).adjoint

    @cached_property
    def div(self) -> operators.LinearOperator:
        """
        Divergence mapping vector fields to functions.

        div : 𝔛(M) ↦ A
        div = -δ ∘ ♭
        """
        return -self.grad.adjoint

    # -------------------------------------------------------------------------
    # Laplacians
    # -------------------------------------------------------------------------

    @lru_cache(maxsize=None)
    def up_laplacian(self, k: int) -> operators.LinearOperator:
        """
        Up-Laplacian mapping k-forms to k-forms.

        Δ_up = δ d : Ωᵏ(M) ↦ Ωᵏ(M)
        """
        assert 0 <= k <= self.dim, f"Up-Laplacian undefined for degree k={k}."
        space = self.form_space(k)
        if k == self.dim:
            return operators.zero(space)
        if k == 0:
            weak_matrix = operators.up_delta_weak(
                self.cache.gamma_functions,
                self.cache.gamma_mixed,
                self.cache.gamma_coords,
                None,
                None,
                self.triple.measure,
                k,
            )
            return operators.LinearOperator(
                domain=space,
                codomain=space,
                weak_matrix=weak_matrix,
            )
        gamma_submatrices, compound_matrices = self.cache.gamma_coords_compound(k)
        weak_matrix = operators.up_delta_weak(
            self.cache.gamma_functions,
            self.cache.gamma_mixed,
            self.cache.gamma_coords,
            gamma_submatrices,
            compound_matrices,
            self.triple.measure,
            k,
            self.n_coefficients,
        )
        return operators.LinearOperator(
            domain=space,
            codomain=space,
            weak_matrix=weak_matrix,
        )

    @lru_cache(maxsize=None)
    def down_laplacian(self, k: int) -> operators.LinearOperator:
        """
        Down-Laplacian mapping k-forms to k-forms.

        Δ_down = d δ : Ωᵏ(M) ↦ Ωᵏ(M)
        """
        assert 0 <= k <= self.dim, f"Down-Laplacian undefined for degree k={k}."
        if k == 0:
            return operators.zero(self.function_space)
        return self.d(k - 1) @ self.codifferential(k)

    @lru_cache(maxsize=None)
    def laplacian(self, k: int) -> operators.LinearOperator:
        """
        Hodge Laplacian mapping k-forms to k-forms.

        Δ = δ d + d δ : Ωᵏ(M) ↦ Ωᵏ(M)
        """
        assert 0 <= k <= self.dim, f"Hodge Laplacian undefined for degree k={k}."
        if k == 0:
            return self.up_laplacian(0)
        if k == self.dim:
            return self.down_laplacian(self.dim)
        return self.up_laplacian(k) + self.down_laplacian(k)

    # -------------------------------------------------------------------------
    # Second-order derivatives: Hessian, Lie bracket, Levi-Civita connection
    # -------------------------------------------------------------------------

    @cached_property
    def hessian(self) -> operators.LinearOperator:
        """
        Hessian operator mapping functions to symmetric (0,2)-tensors.

        Hess : A ↦ Sym²(T*M)
        f ↦ Hess(f) ∇df
        """
        hess = operators.hessian_functions(
            self.triple.function_basis,
            self.triple.immersion_coords,
            self.cache.gamma_coords_regularised,
            # this is the only use of gamma_mixed_regularised so we do not cache it
            self.triple.regularise(self.cache.gamma_mixed),
            cdc=self.triple.cdc,
        )
        weak_matrix = operators.hessian_02_sym_weak(
            self.triple.function_basis,
            self._regularise(hess),
            self.triple.measure,
            self.n_coefficients,
        )
        return operators.LinearOperator(
            domain=self.function_space,
            codomain=self.tensor02sym_space,
            weak_matrix=weak_matrix,
        )

    @cached_property
    def lie_bracket(self) -> operators.BilinearOperator:
        """
        Lie bracket bilinear operator on vector fields.
        [ ⋅ , ⋅ ] : 𝔛(M) × 𝔛(M) → 𝔛(M)

        [X, Y] acts on functions f as [X, Y](f) = X(Y(f)) - Y(X(f)).
        """
        weak_tensor = operators.lie_bracket_weak(
            self.triple.function_basis,
            self.triple.immersion_coords,
            self.cache.gamma_coords,
            self.triple.measure,
            self.n_coefficients,
            cdc=self.triple.cdc,
        )
        return operators.BilinearOperator(
            domain_a=self.vector_field_space,
            domain_b=self.vector_field_space,
            codomain=self.vector_field_space,
            weak_tensor=weak_tensor,
        )

    @cached_property
    def levi_civita(self) -> operators.LinearOperator:
        """
        Levi-Civita connection mapping vector fields to (0,2)-tensors.

        ∇ : 𝔛(M) ↦ Ω⁰²(M)
        Y ↦ ∇Y

        The induced operator of ∇Y is the standard covariant derivative
        X ↦ ∇ₓY.
        """
        weak_matrix = operators.levi_civita_02_weak(
            self.triple.function_basis,
            self.cache.gamma_mixed,
            self.cache.gamma_coords,
            self.cache.hessian_coords,
            self.triple.measure,
            self.n_coefficients,
        )
        return operators.LinearOperator(
            domain=self.vector_field_space,
            codomain=self.tensor02_space,
            weak_matrix=weak_matrix,
        )

    def riemann_curvature(
        self,
        X: tensors.VectorField,
        Y: tensors.VectorField,
        Z: tensors.VectorField,
        W: tensors.VectorField,
    ) -> np.array:
        """
        Compute the Riemann curvature tensor R(X, Y, Z, W) as a scalar function.

        Parameters
        ----------
        X, Y, Z, W : VectorField
            Vector fields to evaluate the Riemann curvature tensor on.

        Returns
        -------
        curvature : np.ndarray, shape (..., n)
            The Riemann curvature tensor evaluated at each data point.
        """
        for V in (X, Y, Z, W):
            if V.space != self.vector_field_space:
                raise ValueError(
                    f"Riemann curvature arguments must be VectorFields, but got {V.space}"
                )
        # g(∇_X(∇_Y(Z)), W) = ∇(∇_Y(Z))(X,W)
        term1 = self.levi_civita(self.levi_civita(Z)(Y))(X, W)
        # g(∇_Y(∇_X(Z)), W) = ∇(∇_X(Z))(Y,W)
        term2 = self.levi_civita(self.levi_civita(Z)(X))(Y, W)
        # g(∇_[X,Y](Z), W) = ∇(Z)([X,Y],W)
        term3 = self.levi_civita(Z)(self.lie_bracket(X, Y), W)
        return term1 - term2 - term3

    def sectional_curvature(
        self, X: tensors.VectorField, Y: tensors.VectorField
    ) -> np.ndarray:
        """
        Compute the sectional curvature K(X, Y) for the plane spanned by X and Y.

        Parameters
        ----------
        X, Y : VectorField
            Vector fields defining the plane.

        Returns
        -------
        K : np.ndarray
            The sectional curvature evaluated at each data point.
            Shape: [..., n]
        """
        for V in (X, Y):
            if V.space != self.vector_field_space:
                raise ValueError(
                    f"Sectional curvature arguments must be VectorFields, but got {V.space}"
                )
        R_XYXY = self.riemann_curvature(X, Y, X, Y)
        denominator = self.g(X, X) * self.g(Y, Y) - self.g(X, Y) ** 2
        denominator = np.where(denominator == 0, np.finfo(float).eps, denominator)
        return R_XYXY / denominator

    # -------------------------------------------------------------------------
    # Factory methods to create functions, vector fields, forms, and tensors
    # from data basis values.
    # -------------------------------------------------------------------------

    def function(self, f_data):
        """
        Create a Function object from pointwise values.

        Parameters
        ----------
        f_data : np.ndarray
            Function values at each data point.
            Shape: [..., n]

        Returns
        -------
        f : Function
            Function object expanded in the basis {φ_i}.
        """
        return tensors.Function.from_pointwise_basis(f_data, self)

    def vector_field(self, X_data, mode="pullback"):
        """
        Create a VectorField object from data basis values.

        Parameters
        ----------
        X_data : np.ndarray, shape (..., n, dim)
            Vector field values at each data point, optionally batched over leading axes.
        mode : str, optional (default="pullback")
            Mode of conversion:
            - "pullback": uses the pullback operation to convert to diffusion basis.
            - "reconstruct": uses least-squares reconstruction to convert to diffusion basis.

        Returns
        -------
        X : VectorField
            VectorField object with diffusion basis coefficients.
        """
        X_data = np.asarray(X_data)
        if X_data.ndim < 2 or X_data.shape[-2:] != (self.n, self.dim):
            raise ValueError(
                "Vector field data must have trailing shape "
                f"({self.n}, {self.dim}), got {X_data.shape}"
            )
        if mode == "pullback":
            return tensors.VectorField.from_pointwise_basis(X_data, self)
        elif mode == "reconstruct":
            return tensors.VectorField.from_reconstruction(X_data, self)
        else:
            raise ValueError('Mode must be either "pullback" or "reconstruct".')

    def form(self, form_data, degree):
        """
        Create a Form object from data basis values.

        Parameters
        ----------
        form_data : np.ndarray, shape (..., n, C(d,k))
            Form values at each data point, optionally batched over leading axes.
            When ``degree`` is 0, this argument is interpreted as function values
            and a :class:`Function` is returned instead.
        degree : int
            Degree k of the differential form. Must be at least 0.

        Returns
        -------
        form : Form
            Form object with diffusion basis coefficients.
        """
        if degree == 0:
            return self.function(form_data)

        form_data = np.asarray(form_data)
        return tensors.Form.from_pointwise_basis(form_data, self, degree)

    def tensor02(self, tensor_data):
        """Create a Tensor02 object from data basis values."""

        tensor_data = np.asarray(tensor_data)
        return tensors.Tensor02.from_pointwise_basis(tensor_data, self)

    def tensor02sym(self, tensor_data):
        """Create a symmetric (0,2)-tensor object from data basis values."""

        tensor_data = np.asarray(tensor_data)
        return tensors.Tensor02Sym.from_pointwise_basis(tensor_data, self)
