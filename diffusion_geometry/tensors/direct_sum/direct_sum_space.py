"""
Direct sum tensor space descriptor for diffusion geometry.

Represents the direct sum of multiple tensor spaces.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Sequence

import numpy as np
from scipy.linalg import block_diag

from diffusion_geometry.tensors.base_tensor.base_tensor_space import BaseTensorSpace


if TYPE_CHECKING:
    from diffusion_geometry.core import DiffusionGeometry
    from .direct_sum_element import DirectSumElement


class DirectSumSpace(BaseTensorSpace):
    """
    Direct sum of tensor spaces sharing a :class:`DiffusionGeometry`.

    Combines multiple tensor spaces into a single space whose elements
    are tuples of elements from the constituent spaces. Coefficients
    are concatenated along the last axis.
    """

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------

    def __init__(
        self, dg: "DiffusionGeometry", spaces: Sequence[BaseTensorSpace]
    ) -> None:
        assert spaces, "Direct sum requires at least one space"
        super().__init__(dg)
        flattened: list[BaseTensorSpace] = []
        for space in spaces:
            assert (
                space.dg == dg
            ), "All summands in a direct sum must share the same DiffusionGeometry"
            if isinstance(space, DirectSumSpace):
                flattened.extend(space.spaces)
            else:
                flattened.append(space)
        self._spaces = tuple(flattened)

    @property
    def spaces(self) -> tuple[BaseTensorSpace, ...]:
        """Underlying tensor spaces that form the direct sum."""
        return self._spaces

    @cached_property
    def dim(self) -> int:
        """Dimension of the coefficient space is the sum of the dimensions of the constituent spaces."""
        return sum(space.dim for space in self._spaces)

    @cached_property
    def _coeff_slices(self) -> tuple[slice, ...]:
        """Slices for splitting the coefficient vector into constituent spaces."""
        slices: list[slice] = []
        offset = 0
        for space in self._spaces:
            size = space.dim
            slices.append(slice(offset, offset + size))
            offset += size
        return tuple(slices)

    # -------------------------------------------------------------------------
    # Block diagonal helpers
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Block diagonal helpers
    # -------------------------------------------------------------------------

    def _block_diag_pointwise(self, matrices: Sequence[np.ndarray]) -> np.ndarray:
        """
        Combine pointwise matrices from each summand into a single block-diagonal pointwise matrix.
        """
        assert matrices, "Direct sum with no summands"
        n = matrices[0].shape[0]
        total = self.dim
        dtype = np.result_type(*(matrix.dtype for matrix in matrices))
        result = np.zeros((n, total, total), dtype=dtype)
        offset = 0
        for matrix, space in zip(matrices, self._spaces):
            size = space.dim
            result[:, offset : offset + size, offset : offset + size] = matrix
            offset += size
        return result

    def _block_diag(self, matrices: Sequence[np.ndarray]) -> np.ndarray:
        """
        Combine matrices from each summand into a single block-diagonal matrix.
        """
        assert matrices, "Direct sum with no summands"
        return block_diag(*matrices)

    # -------------------------------------------------------------------------
    # Metric and Gram matrix
    # -------------------------------------------------------------------------

    @cached_property
    def gram(self) -> np.ndarray:
        """
        Block-diagonal Gram matrix, where each block is the Gram matrix of a summand space.
        """
        grams = [space.gram for space in self._spaces]
        return self._block_diag(grams)

    def metric_apply(self, a_coeffs: np.ndarray, b_coeffs: np.ndarray) -> np.ndarray:
        """
        Apply the metric to two direct sum elements by summing the metric contributions
        from each individual tensor component.
        """
        a_arr = np.asarray(a_coeffs)
        b_arr = np.asarray(b_coeffs)
        parts_a = self.split_coeffs(a_arr)
        parts_b = self.split_coeffs(b_arr)
        result = None
        for space, a_part, b_part in zip(self._spaces, parts_a, parts_b):
            term = space.metric_apply(a_part, b_part)
            result = term if result is None else result + term
        assert result is not None, "DirectSumSpace must contain at least one summand"
        return result

    @cached_property
    def gram_inv(self) -> np.ndarray:
        """Pseudoinverse of the block-diagonal Gram matrix."""
        inverses = [space.gram_inv for space in self._spaces]
        return self._block_diag(inverses)

    @cached_property
    def orthonormal_basis(self) -> np.ndarray:
        """The combined orthonormal basis for the direct sum space."""
        bases = [space.orthonormal_basis for space in self._spaces]
        total = self.dim
        widths = [basis.shape[1] for basis in bases]
        total_width = sum(widths)
        if total_width == 0:
            dtype = np.result_type(*(basis.dtype for basis in bases))
            return np.zeros((total, 0), dtype=dtype)
        dtype = np.result_type(*(basis.dtype for basis in bases))
        result = np.zeros((total, total_width), dtype=dtype)
        row_offset = 0
        col_offset = 0
        for space, basis, width in zip(self._spaces, bases, widths):
            rows = space.dim
            if width:
                result[
                    row_offset : row_offset + rows, col_offset : col_offset + width
                ] = basis
                col_offset += width
            row_offset += rows
        return result

    # -------------------------------------------------------------------------
    # Wrapping and packing
    # -------------------------------------------------------------------------

    def wrap(self, coeffs: np.ndarray) -> "DirectSumElement":
        """Wrap a coefficient array into a DirectSumElement."""
        from .direct_sum_element import DirectSumElement

        return DirectSumElement(self, coeffs)

    def pack(self, *tensors) -> "DirectSumElement":
        assert len(tensors) == len(
            self._spaces
        ), f"Expected {len(self._spaces)} tensors, received {len(tensors)}"
        assert tensors, "Direct sum requires at least one tensor to pack"
        batch_shape = tensors[0].coeffs.shape[:-1]
        parts = []
        for space, tensor in zip(self._spaces, tensors):
            # space.validate_tensor(tensor) # Removed as validate_tensor is deprecated
            assert (
                tensor.coeffs.shape[:-1] == batch_shape
            ), "Tensors must share the same batch shape"
            parts.append(np.asarray(tensor.coeffs))
        coeffs = np.concatenate(parts, axis=-1)
        return self.wrap(coeffs)

    def split_coeffs(self, coeffs: np.ndarray) -> tuple[np.ndarray, ...]:
        return tuple(coeffs[..., sl] for sl in self._coeff_slices)

    # -------------------------------------------------------------------------
    # Equality and hashing
    # -------------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, DirectSumSpace):
            return False
        if self.dg != other.dg:
            return False
        if len(self._spaces) != len(other._spaces):
            return False
        return all(a == b for a, b in zip(self._spaces, other._spaces))

    def __hash__(self) -> int:
        return hash((DirectSumSpace, self.dg, self._spaces))

    def __repr__(self) -> str:
        parts = ", ".join(space.__class__.__name__ for space in self._spaces)
        return (
            f"DirectSumSpace(spaces=[{parts}], dim={self.dim})"
        )
