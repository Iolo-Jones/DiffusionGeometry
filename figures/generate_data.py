"""Utility functions for generating example datasets used in notebooks.

This module centralises the small synthetic datasets that appear across the
example notebooks.  The goal is to make it easy to reproduce the examples in a
consistent way and to provide an extensible structure for adding new datasets
later on.

Two entry points are provided:

* :func:`gen_2d_data` exposes the planar datasets indexed by integer ids.
* :func:`gen_3d_data` exposes volumetric datasets keyed by descriptive names.

Both functions accept ``n`` and ``noise`` parameters so that notebooks can
control the size of the sample and the magnitude of the additive Gaussian noise
that is applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Tuple

import numpy as np

try:  # Optional dependency used for one of the datasets
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow is only required for one dataset.
    Image = None  # type: ignore

try:  # sklearn is an optional dependency for the Swiss roll dataset.
    from sklearn.datasets import make_swiss_roll
except ImportError:  # pragma: no cover - avoid hard dependency if sklearn is missing.
    make_swiss_roll = None  # type: ignore


def load_image_point_cloud(
    image_path: str,
    n: int,
    *,
    threshold: float = 0.8,
    intensity_weighted: bool = True,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Load an image and sample a point cloud of non-white pixels, scaled into [-1,1]^2.

    The image is converted to grayscale in [0,1]. Pixels darker than `threshold`
    are included, their bounding box is mapped into the [-1,1]^2 square (preserving
    aspect ratio), and `n` points are sampled—optionally weighted by darkness.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    n : int
        Number of points to sample.
    threshold : float, optional
        Grayscale threshold in [0,1]; pixels below this value are considered non-white.
        Defaults to 0.8.
    intensity_weighted : bool, optional
        If True (default), sample probabilities are proportional to pixel darkness
        (1 - intensity), so darker pixels are more likely to be chosen.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    data : (n, 2) np.ndarray
        Array of sampled coordinates in [-1,1]^2, with y increasing upward.
    """
    from PIL import Image
    import numpy as np

    rng = np.random.default_rng(random_state)

    # Load and convert to grayscale [0,1]
    img = Image.open(image_path).convert("L")
    gray = np.asarray(img, dtype=float) / 255.0

    # Mask for non-white pixels
    mask = gray < threshold
    if not np.any(mask):
        raise ValueError("No non-white pixels found below threshold; adjust it.")

    # Extract coordinates and corresponding intensities
    y_coords, x_coords = np.where(mask)
    intensities = gray[mask]  # in [0,1], smaller = darker

    # Bounding box
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # Normalise within bounding box to [0,1]
    x = (x_coords - x_min) / max(x_max - x_min, 1)
    y = (y_coords - y_min) / max(y_max - y_min, 1)
    y = 1.0 - y  # flip vertically

    # Preserve aspect ratio, map to [-1,1]^2
    width, height = x_max - x_min, y_max - y_min
    aspect = width / height if height > 0 else 1.0
    if aspect >= 1.0:
        x = 2.0 * (x - 0.5)
        y = 2.0 * (y - 0.5) / aspect
    else:
        x = 2.0 * (x - 0.5) * aspect
        y = 2.0 * (y - 0.5)

    data = np.column_stack((x, y))

    # Compute sampling probabilities
    if intensity_weighted:
        weights = 1.0 - intensities  # darker = higher weight
        weights = np.maximum(weights, 1e-8)
        probs = weights / weights.sum()
    else:
        probs = None  # uniform sampling

    # Sample indices with or without replacement
    total = len(data)
    replace = total < n
    idx = rng.choice(total, size=n, replace=replace, p=probs)
    return data[idx]


Array = np.ndarray
Metadata = Dict[str, Any]
Generated = Tuple[Array, Metadata]


def _add_noise(data: Array, noise: float, rng: np.random.Generator) -> Array:
    """Return ``data`` with isotropic Gaussian noise of standard deviation ``noise``."""
    if noise <= 0:
        return data
    return data + noise * rng.standard_normal(data.shape)


@dataclass
class _DatasetSpec:
    """Container describing how to generate a dataset."""

    name: str
    generator: Callable[[int, float, np.random.Generator, Mapping[str, Any]], Generated]
    default_kwargs: MutableMapping[str, Any] = field(default_factory=dict)

    def __call__(
        self,
        n: int,
        noise: float,
        rng: np.random.Generator,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> Generated:
        params = dict(self.default_kwargs)
        if overrides:
            params.update(overrides)
        return self.generator(n, noise, rng, params)


def _linspace_angles(n: int) -> Array:
    """Return ``n`` evenly spaced angles in ``[0, 2π)``."""
    return np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)


# ---------------------------------------------------------------------------
# 2D dataset generators
# ---------------------------------------------------------------------------


def _dataset_warped_circle(
    n: int, noise: float, rng: np.random.Generator, params: Mapping[str, Any]
) -> Generated:
    """Noisy warped circle with a cloud of outliers.

    Mirrors the datasets constructed in
    ``Intro notebook.ipynb``/``circular_coordinates.ipynb``/``Sobolev testing.ipynb``.
    """

    mix = float(params.get("mix", 0.1))
    outlier_ratio = float(params.get("outlier_ratio", 0.3))
    outlier_loc = np.asarray(params.get("outlier_loc", (-0.6, 0.0)), dtype=float)
    outlier_scale = float(params.get("outlier_scale", 0.2))

    t = _linspace_angles(n)
    base = np.column_stack((np.cos(t), np.sin(t)))
    warp = np.column_stack(
        (
            np.cos(t) / (1.0 + np.sin(t) ** 2),
            np.sin(t) * np.cos(t) / (1.0 + np.sin(t) ** 2),
        )
    )

    data = warp + mix * base
    data = _add_noise(data, noise, rng)

    outliers = int(np.round(outlier_ratio * n))
    if outliers > 0:
        cloud = rng.normal(loc=outlier_loc, scale=outlier_scale, size=(outliers, 2))
        data = np.vstack((data, cloud))

    metadata: Metadata = {"angles": t, "description": "warped circle"}
    return data, metadata


def _dataset_double_frequency(
    n: int, noise: float, rng: np.random.Generator, params: Mapping[str, Any]
) -> Generated:
    """Cosine curve with a second harmonic component (Intro notebook cell 3)."""

    amplitude = float(params.get("amplitude", 0.6))
    t = _linspace_angles(n)
    data = np.column_stack((np.cos(t) + amplitude * np.cos(2.0 * t), np.sin(t)))
    data = _add_noise(data, noise, rng)
    metadata: Metadata = {"angles": t, "description": "double-frequency circle"}
    return data, metadata


def _dataset_sine_warp(
    n: int, noise: float, rng: np.random.Generator, params: Mapping[str, Any]
) -> Generated:
    """Curve with a nonlinear denominator and additional outliers.

    Matches the variant used in ``figures/vector_calculus.ipynb``.
    """

    mix = float(params.get("mix", 0.3))
    outlier_ratio = float(params.get("outlier_ratio", 0.5))
    outlier_loc = np.asarray(params.get("outlier_loc", (-0.2, 0.3)), dtype=float)
    outlier_scale = float(params.get("outlier_scale", 0.2))

    t = _linspace_angles(n)
    base = np.column_stack((np.cos(t), np.sin(t)))
    warp = np.column_stack(
        (
            np.cos(t) / (1.0 + np.sin(2.0 * t) ** 2),
            np.sin(t) * np.cos(t) / (1.0 + np.sin(t) ** 2),
        )
    )

    data = warp + mix * base
    data = _add_noise(data, noise, rng)

    outliers = int(np.round(outlier_ratio * n))
    if outliers > 0:
        cloud = rng.normal(loc=outlier_loc, scale=outlier_scale, size=(outliers, 2))
        data = np.vstack((data, cloud))

    metadata: Metadata = {"angles": t, "description": "sine-warped circle"}
    return data, metadata


def _dataset_anisotropic_curve(
    n: int, noise: float, rng: np.random.Generator, params: Mapping[str, Any]
) -> Generated:
    """Anisotropic closed curve (figures/vector_calculus.ipynb)."""

    mix = float(params.get("mix", 0.9))
    phase = float(params.get("phase", 1.0))
    vertical_scale = float(params.get("vertical_scale", 3.0))
    denom_shift = float(params.get("denom_shift", 5.0))

    t = _linspace_angles(n)
    base = np.column_stack((np.cos(t), np.sin(t)))
    warp = np.column_stack(
        (
            np.cos(t + phase) / (1.0 + np.sin(phase * t) ** 2),
            vertical_scale * np.sin(t) * np.cos(t) / (denom_shift + np.sin(t) ** 2),
        )
    )

    data = warp + mix * base
    data = _add_noise(data, noise, rng)
    metadata: Metadata = {"angles": t, "description": "anisotropic closed curve"}
    return data, metadata


def _dataset_uniform_square(
    n: int, noise: float, rng: np.random.Generator, params: Mapping[str, Any]
) -> Generated:
    """Uniformly sampled square (geodesics/vector_calculus notebooks)."""

    side = float(params.get("side", 20.0))
    data = rng.random((n, 2)) * side
    data = _add_noise(data, noise, rng)
    metadata: Metadata = {"description": "uniform square", "side_length": side}
    return data, metadata


def _dataset_image_point_cloud(
    n: int, noise: float, rng: np.random.Generator, params: Mapping[str, Any]
) -> Generated:
    """Point cloud extracted from an image (used in PDE notebooks).

    Parameters expected in ``params``:
        ``image_path``: path to the image file (defaults to ``../data/tissue_slice.jpg``).
        ``threshold``: grayscale threshold in ``[0, 1]`` (defaults to ``0.8``).
    """

    if Image is None:
        raise ImportError(
            "Pillow is required for the image point cloud dataset but is not installed."
        )

    image_path = Path(params.get("image_path", "../data/tissue_slice.jpg"))
    threshold = float(params.get("threshold", 0.8))

    if not image_path.exists():
        raise FileNotFoundError(f"Image path '{image_path}' does not exist")

    image = Image.open(image_path).convert("L")
    gray = np.asarray(image, dtype=float) / 255.0

    mask = gray < threshold
    y_coords, x_coords = np.where(mask)
    if len(x_coords) == 0:
        raise ValueError(
            "No points found below the threshold; adjust the threshold parameter."
        )

    x = x_coords.astype(float) / max(gray.shape[1] - 1, 1)
    y = (gray.shape[0] - 1 - y_coords).astype(float) / max(gray.shape[0] - 1, 1)
    data = np.column_stack((x, y))

    total = len(data)
    if n <= 0:
        raise ValueError("n must be positive for the image point cloud dataset")
    if total >= n:
        indices = rng.choice(total, size=n, replace=False)
        data = data[indices]
    else:
        indices = rng.choice(total, size=n, replace=True)
        data = data[indices]

    data = _add_noise(data, noise, rng)
    metadata: Metadata = {
        "description": "image point cloud",
        "image_path": str(image_path),
        "threshold": threshold,
    }
    return data, metadata


_2D_DATASETS: Dict[int, _DatasetSpec] = {
    0: _DatasetSpec("warped_circle", _dataset_warped_circle),
    1: _DatasetSpec("double_frequency_circle", _dataset_double_frequency),
    2: _DatasetSpec("sine_warped_circle", _dataset_sine_warp),
    3: _DatasetSpec("anisotropic_curve", _dataset_anisotropic_curve),
    4: _DatasetSpec("uniform_square", _dataset_uniform_square),
    5: _DatasetSpec("image_point_cloud", _dataset_image_point_cloud),
}


def available_2d_datasets() -> Mapping[int, str]:
    """Return a mapping from dataset indices to their human readable names."""

    return {idx: spec.name for idx, spec in _2D_DATASETS.items()}


def gen_2d_data(
    index: int,
    n: int = 200,
    noise: float = 0.1,
    *,
    seed: Optional[int] = None,
    **dataset_kwargs: Any,
) -> Generated:
    """Generate one of the 2D datasets used in the example notebooks.

    Parameters
    ----------
    index:
        Identifier of the dataset to generate.  Use :func:`available_2d_datasets`
        to inspect the available options.
    n:
        Number of samples to draw from the dataset.  For closed curves this is
        the number of points along the parametrised curve; for point clouds or
        images this is the number of points to return (with replacement if
        necessary).
    noise:
        Standard deviation of the isotropic Gaussian noise added to the
        generated coordinates.
    seed:
        Optional seed or ``numpy.random.Generator`` to control reproducibility.
    **dataset_kwargs:
        Additional keyword arguments forwarded to the underlying dataset
        generator.  This makes it easy to tweak parameters such as outlier
        ratios or image thresholds from notebooks.

    Returns
    -------
    (data, metadata):
        ``data`` is an ``(n, 2)`` array (or larger if outliers are appended).
        ``metadata`` exposes auxiliary information such as the parameterisation
        angles used to generate the curve.
    """

    if index not in _2D_DATASETS:
        raise KeyError(
            f"Unknown 2D dataset index {index}. Available indices: {sorted(_2D_DATASETS)}"
        )

    spec = _2D_DATASETS[index]
    if "random_state" in dataset_kwargs:
        raise TypeError("random_state was renamed to seed")
    rng = (
        seed
        if isinstance(seed, np.random.Generator)
        else np.random.default_rng(seed)
    )
    return spec(n, noise, rng, dataset_kwargs)


# ---------------------------------------------------------------------------
# 3D dataset generators
# ---------------------------------------------------------------------------


def _dataset_swiss_roll(
    n: int, noise: float, rng: np.random.Generator, params: Mapping[str, Any]
) -> Generated:
    """Swiss roll dataset mirroring the usage in the notebooks."""

    if make_swiss_roll is None:
        raise ImportError(
            "scikit-learn is required for the Swiss roll dataset but is not installed."
        )

    data, color = make_swiss_roll(
        n_samples=n, noise=noise, random_state=rng.integers(0, 1 << 32)
    )
    metadata: Metadata = {"description": "swiss roll", "color": color}
    return data, metadata


def _dataset_torus(
    n: int, noise: float, rng: np.random.Generator, params: Mapping[str, Any]
) -> Generated:
    """Sample points on a torus.

    Parameters recognised in ``params``:
        ``grid_angles`` (bool): when true, sample angles from a regular grid
            instead of drawing them randomly.
    """

    major_radius = float(params.get("major_radius", 2.0))
    minor_radius = float(params.get("minor_radius", 1.0))

    grid_angles = bool(params.get("grid_angles", False))
    if grid_angles:
        # Build a near-square grid so that the first ``n`` samples are used.
        grid_u = int(np.ceil(np.sqrt(n)))
        grid_v = int(np.ceil(n / grid_u))
        u_vals = np.linspace(0.0, 2.0 * np.pi, grid_u, endpoint=False)
        v_vals = np.linspace(0.0, 2.0 * np.pi, grid_v, endpoint=False)
        uu, vv = np.meshgrid(u_vals, v_vals, indexing="ij")
        stacked = np.column_stack((uu.ravel(), vv.ravel()))
        u = stacked[:n, 0]
        v = stacked[:n, 1]
    else:
        u = rng.uniform(0.0, 2.0 * np.pi, size=n)
        v = rng.uniform(0.0, 2.0 * np.pi, size=n)

    x = (major_radius + minor_radius * np.cos(v)) * np.cos(u)
    y = (major_radius + minor_radius * np.cos(v)) * np.sin(u)
    z = minor_radius * np.sin(v)

    data = np.column_stack((x, y, z))
    data = _add_noise(data, noise, rng)
    metadata: Metadata = {
        "description": "torus",
        "major_radius": major_radius,
        "minor_radius": minor_radius,
        "sampling": "grid" if grid_angles else "random",
        "parameters": np.column_stack((u, v)),
    }
    return data, metadata


def _dataset_twoholed_torus(
    n: int, noise: float, rng: np.random.Generator, params: Mapping[str, Any]
) -> Generated:
    """
    Sample points on a smooth two-holed torus built by joining two tori
    along the x=0 plane (genus-2 surface approximation).

    Strategy:
      • Oversample two standard tori centered at x = ±offset.
      • For the left torus, keep only x < 0 region; for the right, keep only x > 0.
      • Merge, add small noise, and randomly downsample to exactly n points.

    Parameters recognised in ``params``:
        ``major_radius`` (float): radius from centre to tube centre (default 2.0)
        ``minor_radius`` (float): radius of the tube (default 1.0)
        ``offset`` (float): half-distance between the two torus centres (default 2.5)
        ``oversample_factor`` (float): how much to oversample before trimming (default 3.0)
        ``grid_angles`` (bool): sample angles on a grid instead of randomly.
    """
    major_radius = float(params.get("major_radius", 2.0))
    minor_radius = float(params.get("minor_radius", 1.0))
    offset = float(params.get("offset", 2.5))
    oversample_factor = float(params.get("oversample_factor", 3.0))
    grid_angles = bool(params.get("grid_angles", False))

    n_each = int(n * oversample_factor // 2)  # oversample per torus

    # --- helper: sample (u,v) angles ---
    def sample_angles(n):
        if grid_angles:
            grid_u = int(np.ceil(np.sqrt(n)))
            grid_v = int(np.ceil(n / grid_u))
            u_vals = np.linspace(0.0, 2.0 * np.pi, grid_u, endpoint=False)
            v_vals = np.linspace(0.0, 2.0 * np.pi, grid_v, endpoint=False)
            uu, vv = np.meshgrid(u_vals, v_vals, indexing="ij")
            stacked = np.column_stack((uu.ravel(), vv.ravel()))
            u = stacked[:n, 0]
            v = stacked[:n, 1]
        else:
            u = rng.uniform(0.0, 2.0 * np.pi, size=n)
            v = rng.uniform(0.0, 2.0 * np.pi, size=n)
        return u, v

    # --- left torus (centered at -offset) ---
    u1, v1 = sample_angles(n_each)
    x1 = (major_radius + minor_radius * np.cos(v1)) * np.cos(u1) - offset
    y1 = (major_radius + minor_radius * np.cos(v1)) * np.sin(u1)
    z1 = minor_radius * np.sin(v1)

    # --- right torus (centered at +offset) ---
    u2, v2 = sample_angles(n_each)
    x2 = (major_radius + minor_radius * np.cos(v2)) * np.cos(u2) + offset
    y2 = (major_radius + minor_radius * np.cos(v2)) * np.sin(u2)
    z2 = minor_radius * np.sin(v2)

    # --- trimming: keep only one side from each ---
    mask_left = x1 < 0.0
    mask_right = x2 > 0.0
    X_left = np.column_stack((x1[mask_left], y1[mask_left], z1[mask_left]))
    X_right = np.column_stack((x2[mask_right], y2[mask_right], z2[mask_right]))

    # --- merge and add small smoothing jitter ---
    data = np.concatenate([X_left, X_right], axis=0)
    data = _add_noise(data, noise, rng)

    # --- random downsample to n points ---
    if data.shape[0] > n:
        idx = rng.choice(data.shape[0], size=n, replace=False)
        data = data[idx]

    metadata: Metadata = {
        "description": "two-holed torus (joined tori with cut at x=0)",
        "major_radius": major_radius,
        "minor_radius": minor_radius,
        "offset": offset,
        "oversample_factor": oversample_factor,
        "sampling": "grid" if grid_angles else "random",
    }
    return data, metadata


def _dataset_sphere_with_handles(
    n: int, noise: float, rng: np.random.Generator, params: Mapping[str, Any]
) -> Generated:
    """
    Sample points on a sphere of radius 1 with two circular handles in the x–y plane,
    each tangent to the sphere at (-1, 0, 0) and (1, 0, 0).

    The handles are simple circles (not tori) of radius `handle_radius` lying in
    the x–y plane, touching the sphere externally.
    """

    sphere_radius = 1.0
    handle_radius = float(params.get("handle_radius", 1.0))
    grid_angles = bool(params.get("grid_angles", False))
    ratio = float(params.get("points_per_component", 0.5))  # fraction for sphere

    # --- Number of samples per part ---
    n_sphere = int(n * ratio)
    n_each_handle = max(1, (n - n_sphere) // 2)

    # --- Sphere points ---
    phi = np.arccos(1.0 - 2.0 * rng.random(n_sphere))
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_sphere)
    x_s = sphere_radius * np.sin(phi) * np.cos(theta)
    y_s = sphere_radius * np.sin(phi) * np.sin(theta)
    z_s = sphere_radius * np.cos(phi)
    sphere_points = np.column_stack((x_s, y_s, z_s))

    # --- Helper for a handle circle ---
    def sample_circle(center_x: float, n_points: int) -> np.ndarray:
        if grid_angles:
            t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
        else:
            t = rng.uniform(0.0, 2.0 * np.pi, size=n_points)
        # circle in x–y plane centered *outside* the sphere
        x = center_x + handle_radius * np.cos(t)
        y = handle_radius * np.sin(t)
        z = np.zeros_like(t)
        return np.column_stack((x, y, z))

    # --- Circle centers positioned so circles are tangent to the sphere ---
    left_center = -sphere_radius - handle_radius  # touches at x = -1
    right_center = sphere_radius + handle_radius  # touches at x = +1

    left_handle = sample_circle(center_x=left_center, n_points=n_each_handle)
    right_handle = sample_circle(center_x=right_center, n_points=n_each_handle)

    # --- Combine all components ---
    data = np.vstack((sphere_points, left_handle, right_handle))
    data = _add_noise(data, noise, rng)

    metadata: Metadata = {
        "description": "sphere with two external circular handles",
        "sphere_radius": sphere_radius,
        "handle_radius": handle_radius,
        "points_per_component": ratio,
        "sampling": "grid" if grid_angles else "random",
    }
    return data, metadata


def _dataset_sphere(
    n: int, noise: float, rng: np.random.Generator, params: Mapping[str, Any]
) -> Generated:
    """Points approximately uniformly distributed on a sphere."""

    radius = float(params.get("radius", 1.0))

    phi = np.arccos(1.0 - 2.0 * rng.random(n))
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    data = np.column_stack((x, y, z))
    data = _add_noise(data, noise, rng)
    metadata: Metadata = {
        "description": "sphere",
        "radius": radius,
        "angles": np.column_stack((phi, theta)),
    }
    return data, metadata


def _dataset_ball(
    n: int, noise: float, rng: np.random.Generator, params: Mapping[str, Any]
) -> Generated:
    """Points uniformly sampled from a solid ball."""

    radius = float(params.get("radius", 1.0))
    center = np.asarray(params.get("center", (0.0, 0.0, 0.0)), dtype=float)

    if center.shape != (3,):
        raise ValueError("center must be a length-3 iterable of floats")

    directions = rng.normal(size=(n, 3))
    norms = np.linalg.norm(directions, axis=1)
    norms[norms == 0.0] = 1.0  # avoid division by zero if a vector is degenerate
    radii = radius * rng.random(n) ** (1.0 / 3.0)

    data = center + directions / norms[:, None] * radii[:, None]
    data = _add_noise(data, noise, rng)
    metadata: Metadata = {
        "description": "solid ball",
        "radius": radius,
        "center": center,
    }
    return data, metadata


def _dataset_cone(
    n: int, noise: float, rng: np.random.Generator, params: Mapping[str, Any]
) -> Generated:
    """Uniformly sampled 3D cone (solid or surface).

    Samples points uniformly inside or on the surface of a right circular cone
    aligned with the z-axis and apex at the origin.

    Parameters recognised in ``params``:
        ``height`` (float): total height of the cone (default 1.0)
        ``radius`` (float): base radius at z = height (default 1.0)
        ``solid`` (bool): if True (default), sample throughout the cone volume;
                         if False, sample only on its surface.

    Returns
    -------
    (data, metadata):
        data: (n, 3) array of XYZ coordinates
        metadata: dictionary with cone parameters
    """

    height = float(params.get("height", 1.0))
    radius = float(params.get("radius", 1.0))
    solid = bool(params.get("solid", True))

    # --- Uniform height distribution ---
    if solid:
        # Uniform in volume → z ∝ u^(1/3)
        u = rng.random(n)
        z = height * u ** (1.0 / 3.0)
        # Radius at that height (linear profile)
        r_max = (z / height) * radius
        # Uniform in disk area → r ∝ sqrt(v)
        v = rng.random(n)
        r = np.sqrt(v) * r_max
    else:
        # Surface only: linear relation between z and r
        z = rng.random(n) * height
        r = (z / height) * radius

    # --- Azimuthal angle ---
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    data = np.column_stack((x, y, z))
    data = _add_noise(data, noise, rng)

    metadata: Metadata = {
        "description": "cone (solid uniform)" if solid else "cone (surface)",
        "height": height,
        "radius": radius,
        "solid": solid,
    }
    return data, metadata


def _dataset_hyperboloid(
    n: int, noise: float, rng: np.random.Generator, params: Mapping[str, Any]
) -> Generated:
    """Sample points on a two-sheet hyperboloid."""

    a = float(params.get("a", 1.0))
    c = float(params.get("c", 1.0))
    max_radius = float(params.get("max_radius", 2.0))

    r = rng.uniform(0.0, max_radius, size=n)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    z = c * np.sqrt(1.0 + (r / a) ** 2)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    data = np.column_stack((x, y, z))
    data = _add_noise(data, noise, rng)
    metadata: Metadata = {
        "description": "hyperboloid",
        "a": a,
        "c": c,
        "max_radius": max_radius,
    }
    return data, metadata


_3D_DATASETS: Dict[str, _DatasetSpec] = {
    "swiss_roll": _DatasetSpec("swiss_roll", _dataset_swiss_roll),
    "torus": _DatasetSpec("torus", _dataset_torus),
    "twoholed_torus": _DatasetSpec("twoholed_torus", _dataset_twoholed_torus),
    "sphere": _DatasetSpec("sphere", _dataset_sphere),
    "sphere_with_handles": _DatasetSpec(
        "sphere_with_handles", _dataset_sphere_with_handles
    ),
    "ball": _DatasetSpec("ball", _dataset_ball),
    "cone": _DatasetSpec("cone", _dataset_cone),
    "hyperboloid": _DatasetSpec("hyperboloid", _dataset_hyperboloid),
}


def available_3d_datasets() -> Mapping[str, str]:
    """Return a mapping from dataset keys to their human readable names."""

    return {key: spec.name for key, spec in _3D_DATASETS.items()}


def gen_3d_data(
    kind: str,
    n: int = 500,
    noise: float = 0.0,
    *,
    seed: Optional[int] = None,
    **dataset_kwargs: Any,
) -> Generated:
    """Generate one of the 3D datasets used in the example notebooks.

    Parameters
    ----------
    kind:
        String identifier of the dataset to generate (e.g. ``"torus"`` or
        ``"swiss_roll"``).  Use :func:`available_3d_datasets` to inspect the
        available options.
    n:
        Number of points to sample.
    noise:
        Standard deviation of the isotropic Gaussian noise added to the
        coordinates (for datasets without intrinsic noise).
    seed:
        Optional seed or ``numpy.random.Generator`` to control reproducibility.
    **dataset_kwargs:
        Additional parameters forwarded to the dataset generator.

    Returns
    -------
    (data, metadata):
        ``data`` is an ``(n, 3)`` array of coordinates.  The metadata dictionary
        may contain additional information such as colouring values used in the
        notebooks.
    """

    kind_key = kind.lower()
    if kind_key not in _3D_DATASETS:
        raise KeyError(
            f"Unknown 3D dataset '{kind}'. Available datasets: {sorted(_3D_DATASETS)}"
        )

    spec = _3D_DATASETS[kind_key]
    if "random_state" in dataset_kwargs:
        raise TypeError("random_state was renamed to seed")
    rng = (
        seed
        if isinstance(seed, np.random.Generator)
        else np.random.default_rng(seed)
    )
    return spec(n, noise, rng, dataset_kwargs)


__all__ = [
    "gen_2d_data",
    "gen_3d_data",
    "available_2d_datasets",
    "available_3d_datasets",
]
