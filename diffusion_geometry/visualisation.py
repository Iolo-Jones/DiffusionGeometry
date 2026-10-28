from __future__ import annotations
import numpy as np
from opt_einsum import contract
import plotly
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.neighbors import NearestNeighbors

import re
import inspect


DEFAULT_COLORSCALE = ["#166dde", "#d3d3d3", "#e32636"]
CYCLIC_COLORSCALE = ["#166dde", "#d3d3d3", "#e32636", "#166dde"]


def plot_scatter_2d(
    points, color=None, size=6, colorscale=None, cyclic=False, opacity=1.0, amax=None
):
    """
    Simple 2D scatter plot with optional colouring.
    Accepts either:
      - scalar/array colour values (with colorscale)
      - a single string (e.g. 'red', '#ff0000') for uniform colour.
    """
    points = np.asarray(points, float)
    if points.shape[1] != 2:
        raise ValueError("points must be (n,2)")

    if colorscale is None:
        colorscale = CYCLIC_COLORSCALE if cyclic else DEFAULT_COLORSCALE

    fig = go.Figure()
    marker = dict(size=size)

    # Handle opacity (scalar or array)
    if np.isscalar(opacity):
        marker["opacity"] = float(opacity)
    else:
        opacity = np.asarray(opacity, float).reshape(-1)
        if opacity.shape[0] != points.shape[0]:
            raise ValueError("opacity must have length n to match points")
        marker["opacity"] = opacity

    # --- Colour handling ---
    if isinstance(color, str):
        # uniform string colour
        marker["color"] = color
    else:
        if color is None:
            color = np.zeros(points.shape[0])
        color = np.asarray(color, float).reshape(-1)
        if color.shape[0] != points.shape[0]:
            raise ValueError("color must have length n to match points")

        # Normalisation: divergent vs cyclic
        if cyclic:
            cmin, cmax, cmid = 0, 2 * np.pi, np.pi
        else:
            if amax is None:
                amax = (
                    float(np.nanmax(np.abs(color)))
                    if np.any(np.isfinite(color))
                    else 1.0
                )
                if amax == 0:
                    amax = 1.0
            cmin, cmax, cmid = -amax, +amax, 0.0

        marker.update(
            color=color,
            colorscale=colorscale,
            showscale=False,
            cmin=cmin,
            cmax=cmax,
        )
        if cmid is not None:
            marker["cmid"] = cmid

    fig.add_trace(
        go.Scatter(x=points[:, 0], y=points[:, 1], mode="markers", marker=marker)
    )
    clean_fig(fig)
    return fig


def plot_scatter_3d(
    points,
    color=None,
    size=3,
    colorscale=None,
    cyclic=False,
    opacity=1.0,
    amax=None,
    *,
    project_to_2d=False,
    camera=None,
    projection="orthographic",  # "orthographic" or "perspective"
    fov_y_deg=45.0,
    aspect=1.0,
    near=0.1,
    far=100.0,
    ortho_scale=1.0,
    fix_ndc_axes=True,  # keep axes to [-1,1] if projecting
):
    """
    3D scatter with optional projection to 2D.
    If project_to_2d=True, project using camera+projection, then call plot_scatter_2d.
    Colour handling mirrors the 3D path (array or single colour string).
    Points are depth-sorted by distance from camera (furthest first)
    to ensure consistent rendering order.
    """
    import numpy as np
    import plotly.graph_objects as go

    points = np.asarray(points, float)
    if points.shape[1] != 3:
        raise ValueError("points must be (n,3)")

    # Validate opacity early so we can reuse for both 3D and projected paths
    if np.isscalar(opacity):
        opacity_scalar = float(opacity)
        opacity_array = None
    else:
        opacity_array = np.asarray(opacity, float).reshape(-1)
        if opacity_array.shape[0] != points.shape[0]:
            raise ValueError("opacity must have length n to match points")
        opacity_scalar = None

    # ---- 3D path ----
    if not project_to_2d:
        # ---------- Depth ordering ----------
        if camera is not None:
            eye = np.array(
                [
                    camera["eye"]["x"],
                    camera["eye"]["y"],
                    camera["eye"]["z"],
                ],
                dtype=float,
            )
            dist = np.linalg.norm(points - eye, axis=1)
            order = np.argsort(dist)[::-1]  # farthest first
            points = points[order]

            if opacity_array is not None:
                opacity_array = opacity_array[order]
            if not isinstance(size, (int, float)):
                size = np.asarray(size)[order]
            if not isinstance(color, str) and color is not None:
                color = np.asarray(color)[order]

        # ---------- Colour + size setup ----------
        if colorscale is None:
            cs = CYCLIC_COLORSCALE if cyclic else DEFAULT_COLORSCALE
        else:
            cs = colorscale

        fig = go.Figure()
        marker = dict(size=size, line=dict(width=0, color="rgba(0,0,0,0)"))

        marker_opacity = opacity_array if opacity_array is not None else opacity_scalar
        marker["opacity"] = marker_opacity

        if isinstance(color, str):
            marker["color"] = color
        else:
            if color is None:
                c = np.zeros(points.shape[0])
            else:
                c = np.asarray(color, float).reshape(-1)
                if c.shape[0] != points.shape[0]:
                    raise ValueError("color must have length n to match points")

            if cyclic:
                cmin, cmax, cmid = 0, 2 * np.pi, None
            else:
                if amax is None:
                    amax = (
                        float(np.nanmax(np.abs(c))) if np.any(np.isfinite(c)) else 1.0
                    )
                    if amax == 0:
                        amax = 1.0
                cmin, cmax, cmid = -amax, +amax, 0.0

            marker.update(color=c, colorscale=cs, showscale=False, cmin=cmin, cmax=cmax)
            if cmid is not None:
                marker["cmid"] = cmid

        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=marker,
            )
        )

        clean_fig(fig)
        if camera is not None:
            fig.update_layout(scene=dict(camera=camera))
        return fig

    # ---- Projected 2D path ----
    if camera is None:
        raise ValueError(
            "camera (eye/center/up) must be provided when project_to_2d=True."
        )

    xy = _project_points(
        points,
        camera,
        mode=projection,
        fov_y_deg=fov_y_deg,
        aspect=aspect,
        near=near,
        far=far,
        ortho_scale=ortho_scale,
    )

    if colorscale is None:
        cs = CYCLIC_COLORSCALE if cyclic else DEFAULT_COLORSCALE
    else:
        cs = colorscale

    fig2d = plot_scatter_2d(
        xy,
        color=(
            color
            if (isinstance(color, str) or color is None)
            else np.asarray(color, float).reshape(-1)
        ),
        size=size,
        colorscale=cs,
        cyclic=cyclic,
        opacity=opacity_array if opacity_array is not None else opacity_scalar,
        amax=amax,
    )

    if fix_ndc_axes:
        fig2d.update_xaxes(range=[-1, 1])
        fig2d.update_yaxes(range=[-1, 1])

    return fig2d


def plot_quiver_2d(
    data,
    quiver,
    scale=1.0,
    arrow_scale=0.4,
    line_width=3,
    color="black",
):
    """
    Simple 2D quiver using Plotly figure_factory.create_quiver.
    The input vectors are pre-scaled by `scale` for predictable behaviour.
    """
    data = np.asarray(data, float)
    quiver = np.asarray(quiver, float)
    if data.shape != quiver.shape or data.shape[1] != 2:
        raise ValueError("data and quiver must be (n,2) arrays.")

    # Pre-scale vectors ourselves so `scale` behaves linearly.
    u = quiver[:, 0] * scale
    v = quiver[:, 1] * scale

    fig = ff.create_quiver(
        data[:, 0],
        data[:, 1],
        u,
        v,
        scale=1.0,  # we've already applied scaling
        arrow_scale=arrow_scale,
        line_width=line_width,
        marker=dict(color=color),
    )
    clean_fig(fig)
    return fig


def plot_quiver_3d(
    data,
    quiver,
    scale=1.0,
    arrow_scale=0.3,
    line_width=5.0,
    shaft_color="black",
    arrow_color="black",
    *,
    project_to_2d=False,
    camera=None,
    projection="orthographic",  # "orthographic" or "perspective"
    fov_y_deg=45.0,
    aspect=1.0,
    near=0.1,
    far=100.0,
    ortho_scale=1.0,  # size of the orthographic window
    quiver2d_color="black",  # forwarded to plot_quiver_2d if projecting
    opacity=None,  # scalar or array-like, per-arrow fade
):
    """
    If project_to_2d is False: render a true 3D quiver (Scatter3d+Mesh3d).
    If True: project (data, data+quiver) to 2D via the given camera & projection,
             then call plot_quiver_2d to get a vector-friendly 2D figure.

    Each arrow runs from data[i] to data[i] + scale * quiver[i]. As in
    plot_quiver_2d, the arrowhead length is arrow_scale times the arrow length,
    so head size is proportional to the pointwise magnitude and does not depend
    on the rest of the data (number of points, spacing, ordering, max norm).

    Parameters
    ----------
    opacity : float or array-like, optional
        Per-arrow fade in [0,1] (scalar or length n). Arrows are blended
        towards white rather than made transparent, which looks the same on a
        white background and needs only one trace for the whole field.
    """
    data = np.asarray(data, float)
    quiver = np.asarray(quiver, float)
    if data.shape != quiver.shape or data.shape[1] != 3:
        raise ValueError("data and quiver must both be (n,3) arrays.")

    if project_to_2d:
        if camera is None:
            raise ValueError(
                "camera (eye/center/up) must be provided when project_to_2d=True."
            )
        xy0 = _project_points(
            data,
            camera,
            mode=projection,
            fov_y_deg=fov_y_deg,
            aspect=aspect,
            near=near,
            far=far,
            ortho_scale=ortho_scale,
        )
        xy1 = _project_points(
            data + quiver,
            camera,
            mode=projection,
            fov_y_deg=fov_y_deg,
            aspect=aspect,
            near=near,
            far=far,
            ortho_scale=ortho_scale,
        )
        q2d = xy1 - xy0
        return plot_quiver_2d(
            xy0,
            q2d,
            scale=scale,
            arrow_scale=arrow_scale,
            line_width=line_width,
            color=quiver2d_color,
        )

    # ---- 3D path ----
    vec = quiver * scale
    lengths = np.linalg.norm(vec, axis=1)
    keep = lengths > 0
    alpha = None
    if opacity is not None:
        alpha = np.clip(np.broadcast_to(np.asarray(opacity, float), len(data)), 0, 1)[keep]
    data, vec, lengths = data[keep], vec[keep], lengths[keep]
    directions = vec / lengths[:, None]
    tips = data + vec
    head_lengths = arrow_scale * lengths
    # Shafts stop at the base of the head so the line never pokes through the apex
    shaft_end = tips - head_lengths[:, None] * directions

    fig = go.Figure()

    # --- Shafts: one Scatter3d, segments separated by NaNs ---
    nan = np.full(len(data), np.nan)
    x, y, z = (np.column_stack((data[:, a], shaft_end[:, a], nan)).ravel() for a in range(3))
    line = dict(color=shaft_color, width=line_width)
    if alpha is not None:
        # Fade towards white via a two-colour scale (one value per segment vertex)
        line.update(color=np.repeat(alpha, 3), colorscale=[[0, "white"], [1, shaft_color]], cmin=0, cmax=1)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=line, showlegend=False))

    # --- Arrowheads: one Mesh3d of cones in data coordinates ---
    verts, i, j, k = _arrowhead_mesh(tips, directions, head_lengths)
    head = dict(color=arrow_color)
    if alpha is not None:
        n_verts = len(verts) // max(len(data), 1)
        head = dict(
            intensity=np.repeat(alpha, n_verts),
            colorscale=[[0, "white"], [1, arrow_color]],
            cmin=0,
            cmax=1,
        )
    fig.add_trace(
        go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=i,
            j=j,
            k=k,
            flatshading=True,
            lighting=dict(ambient=0.7, diffuse=0.4, specular=0.1),
            hoverinfo="skip",
            showscale=False,
            **head,
        )
    )

    clean_fig(fig)
    return fig


def _arrowhead_mesh(tips, directions, lengths, width=0.35, n_sides=12):
    """
    Triangulated cones for a batch of arrowheads, in data coordinates.

    Cone m has its apex at tips[m], points along directions[m] (unit vectors),
    and has height lengths[m] and base radius width * lengths[m]. Returns the
    vertex array (m*(n_sides+2), 3) and triangle index arrays i, j, k for
    go.Mesh3d.
    """
    m = len(tips)
    # Orthonormal frame (e1, e2) perpendicular to each direction
    helper = np.where(
        np.abs(directions[:, [0]]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]
    )
    e1 = np.cross(directions, helper)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(directions, e1)

    angles = 2 * np.pi * np.arange(n_sides) / n_sides
    base = tips - lengths[:, None] * directions  # (m,3)
    ring = base[:, None, :] + width * lengths[:, None, None] * (
        np.cos(angles)[None, :, None] * e1[:, None, :]
        + np.sin(angles)[None, :, None] * e2[:, None, :]
    )  # (m,n_sides,3)

    # Per cone: [apex, base centre, ring_0, ..., ring_{n_sides-1}]
    verts = np.concatenate([tips[:, None], base[:, None], ring], axis=1)
    verts = verts.reshape(-1, 3).astype(np.float32)  # float32/int32 halve the HTML size

    stride = n_sides + 2
    offset = (stride * np.arange(m))[:, None]
    r0 = offset + 2 + np.arange(n_sides)[None, :]  # (m,n_sides)
    r1 = offset + 2 + (np.arange(n_sides)[None, :] + 1) % n_sides
    apex = np.broadcast_to(offset, r0.shape)
    centre = apex + 1
    # Sides (apex, r0, r1) then base cap (centre, r1, r0)
    i = np.concatenate([apex, centre], axis=1)
    j = np.concatenate([r0, r1], axis=1)
    k = np.concatenate([r1, r0], axis=1)
    return verts, *(a.ravel().astype(np.int32) for a in (i, j, k))


def hodge_star_2_form(omega, orientation: int = +1) -> np.ndarray:
    """
    Compute the Hodge dual of a 2-form represented by ambient skew-symmetric
    matrices ω_ij. Works in 2D or 3D Euclidean ambient coordinates.

    Parameters
    ----------
    omega : ndarray
        Shape (n, d, d), where ω_ij = -ω_ji represents the 2-form at n points.
        The input should already be in Euclidean (ambient) coordinates.
    orientation : {+1, -1}, optional
        Overall orientation sign. +1 corresponds to right-handed coordinates,
        -1 to left-handed (reverses sign of the result).

    Returns
    -------
    star_omega : ndarray
        - For d = 2: shape (n,), scalar field (*ω)
        - For d = 3: shape (n, 3), vector field (*ω)
    """
    if omega.ndim != 3:
        raise ValueError("omega must have shape (n, d, d)")
    n, d, _ = omega.shape

    if orientation not in (+1, -1):
        raise ValueError("orientation must be +1 or -1")

    # --- 2D case: *ω is a scalar function ---
    if d == 2:
        # (*ω) = (1/2) ε^{ij} ω_ij = ω_12 for ε^{12}=+1
        star = -orientation * 0.5 * (omega[:, 0, 1] - omega[:, 1, 0])
        return star  # (n,)

    # --- 3D case: *ω is an axial vector field ---
    elif d == 3:
        eps = np.zeros((3, 3, 3), dtype=omega.dtype)
        # Levi-Civita symbol (right-handed)
        eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = +1
        eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1
        # v^i = (1/2) ε^{ijk} ω_{jk}
        star = -orientation * 0.5 * contract("ijk,njk->ni", eps, omega)
        return star  # (n,3)

    else:
        raise ValueError("Only 2D and 3D cases are supported for 2-forms.")


def plot_2form_2d(
    points,
    matrices,
    radius=0.06,
    n_circle=32,
    colorscale=[[0, "#166dde"], [0.5, "#d3d3d3"], [1, "#e32636"]],
    magnitude_scaling=True,
    opacity=1.0,
):
    """
    Visualise 2-forms in 2D as filled circles centered at `points`,
    with no outlines.
    """
    points = np.asarray(points, float)
    matrices = np.asarray(matrices, float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be (n,2)")
    if matrices.ndim != 3 or matrices.shape[1:] != (2, 2):
        raise ValueError("matrices must be (n,2,2) skew-symmetric")

    # --- Scalar from 2-form ---
    star = hodge_star_2_form(matrices)
    if np.any(np.isfinite(star)):
        amax = float(np.nanmax(np.abs(star))) or 1.0
    else:
        amax = 1.0

    t = 0.5 + 0.5 * (star / amax)
    r_scale = np.abs(star) / amax if magnitude_scaling else 1.0
    radii = radius * r_scale

    # --- Circle geometry ---
    theta = np.linspace(0, 2 * np.pi, int(max(3, n_circle)), endpoint=False)
    unit_circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)

    fig = go.Figure()
    for p, rad, ti in zip(points, radii, t):
        circle = p + rad * unit_circle
        fill_col = plotly.colors.sample_colorscale(colorscale, [ti])[0]
        fig.add_trace(
            go.Scatter(
                x=circle[:, 0],
                y=circle[:, 1],
                mode="lines",
                fill="toself",
                line=dict(color="rgba(0,0,0,1)", width=0.0),  # no outline
                fillcolor=fill_col,
                opacity=float(opacity),
                showlegend=False,
            )
        )

    clean_fig(fig)
    return fig


def plot_2form_3d(
    points,
    matrices,
    radius=0.1,
    n_circle=12,
    colorscale=[[0, "#166dde"], [0.5, "lightgray"], [1, "#e32636"]],
    magnitude_scaling=True,
    offset=1e-3,
):
    """
    Plot oriented disk pairs orthogonal to 3D axial vectors (2-forms).
    Each vector defines:
        • +offset disk (positive side)
        • -offset disk (negative side)
    Colour is taken from the colorscale according to |v|/max|v|:
        smallest → middle colour, largest → extreme ends (blue/red).

    Parameters
    ----------
    points : (n,3)
        Disk centres.
    matrices : (n,3,3)
        Axial matrices (normals of disks).
    radius : float
        Base disk radius.
    n_circle : int
        Number of rim vertices per disk (>=3).
    colorscale : list
        Plotly-style diverging colorscale [[0,color1],[0.5,color_mid],[1,color2]].
    magnitude_scaling : bool
        Scale disk radius by |v|.
    offset : float
        Distance separating the ± disks.
    """
    points = np.asarray(points, float)
    matrices = np.asarray(matrices, float)
    vectors = hodge_star_2_form(matrices)  # (n,3)
    n = len(points)

    # === normalise and get magnitudes ===
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    n_hat = vectors / norms
    mag = norms[:, 0]
    max_mag = np.max(mag)

    # === build orthonormal bases via numpy kernel ===
    absn = np.abs(n_hat)
    min_axis = np.argmin(absn, axis=1)
    a = np.zeros_like(n_hat)
    a[np.arange(n), min_axis] = 1.0
    e1 = np.cross(a, n_hat)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(n_hat, e1)
    e2 /= np.linalg.norm(e2, axis=1, keepdims=True)

    # === circle coordinates ===
    theta = np.linspace(0, 2 * np.pi, n_circle, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    r = radius * (mag if magnitude_scaling else np.ones_like(mag))

    rim = points[:, None, :] + r[:, None, None] * (
        cos_t[None, :, None] * e1[:, None, :] + sin_t[None, :, None] * e2[:, None, :]
    )

    offset_vec = offset * n_hat[:, None, :]

    # + and - offset disks
    front = np.concatenate([points[:, None, :] + offset_vec, rim + offset_vec], axis=1)
    back = np.concatenate([points[:, None, :] - offset_vec, rim - offset_vec], axis=1)

    # === triangulation indices (same for all disks) ===
    n_vert = n_circle + 1
    I, J, K = [], [], []
    for b in np.arange(0, n * n_vert, n_vert):
        ring = np.arange(b + 1, b + n_circle + 1)
        I.extend(np.repeat(b, n_circle))
        J.extend(ring)
        K.extend(np.roll(ring, -1))

    # === colour intensity along colourscale ===
    # map |v|/max|v| -> [0.5,1] for front, [0,0.5] for back
    norm_frac = mag / (max_mag + 1e-12)
    C_front = 0.5 + 0.5 * norm_frac  # mid→red
    C_back = 0.5 - 0.5 * norm_frac  # mid→blue

    def mesh_from_disks(verts, C):
        X, Y, Z = verts.reshape(-1, 3).T
        C_rep = np.repeat(C, n_vert)
        return go.Mesh3d(
            x=X,
            y=Y,
            z=Z,
            i=I,
            j=J,
            k=K,
            intensity=C_rep,
            colorscale=colorscale,
            cmin=0,
            cmax=1,
            # flatshading=True,
            # lighting=dict(diffuse=0.9, specular=0.2),
            showscale=False,
        )

    fig = go.Figure()
    fig.add_trace(mesh_from_disks(front, C_front))
    fig.add_trace(mesh_from_disks(back, C_back))

    clean_fig(fig)
    return fig


def plot_3form_3d(
    points,
    tensors,
    base_size=0.0,
    size_scale=30.0,
    colorscale=[[0, "#166dde"], [0.5, "#d3d3d3"], [1, "#e32636"]],
    magnitude_scaling=True,
    opacity=1.0,
    camera=None,
):
    """
    Visualise 3-forms in 3D as variable-size/colour markers using plot_scatter_3d.
    Points are ordered by distance from the camera (furthest first) to ensure
    correct rendering order.
    """
    import numpy as np

    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n,3)")

    tensors = np.asarray(tensors, float)
    if tensors.ndim != 4 or tensors.shape[1:] != (3, 3, 3):
        raise ValueError("tensors must have shape (n,3,3,3)")

    # Extract scalar component ω_{012}
    values = -tensors[:, 0, 1, 2]
    colour = values

    # Determine sizes
    if magnitude_scaling:
        abs_vals = np.abs(values)
        vmax = np.nanmax(abs_vals) if np.any(np.isfinite(abs_vals)) else 1.0
        norm = abs_vals / vmax if vmax != 0 else abs_vals
        sizes = base_size + size_scale * norm
    else:
        sizes = np.full(values.shape, base_size)

    # ---------- Order points by distance from camera ----------
    if camera is not None:
        # Extract camera eye position
        eye = np.array(
            [
                camera["eye"]["x"],
                camera["eye"]["y"],
                camera["eye"]["z"],
            ],
            dtype=float,
        )

        # Compute distances and sort by descending (furthest first)
        dist = np.linalg.norm(points - eye, axis=1)
        order = np.argsort(dist)[::-1]

        points = points[order]
        colour = colour[order]
        sizes = sizes[order]

    # ---------- Plot ----------
    fig = plot_scatter_3d(
        points,
        color=colour,
        size=sizes,
        colorscale=colorscale,
        cyclic=False,
        camera=camera,  # propagate camera if supported
    )

    # Apply marker opacity and remove white outline
    for trace in fig.data:
        trace.marker.opacity = opacity
        trace.marker.line = dict(width=0, color="rgba(0,0,0,0)")

    # ---------- Set camera ----------
    if camera is not None:
        fig.update_layout(scene=dict(camera=camera))

    return fig


def plot_ellipsoids(
    points,
    tensors,
    scale=1.0,
    n_theta=24,
    n_phi=24,
    magnitude_scaling=True,
    line_color="black",
    line_width=0.0,
):
    """
    Plot symmetric (0,2)-tensors as filled ellipses (2D) or ellipsoids (3D).

    Parameters
    ----------
    points : (n,d)
        Centres of the tensors (2D or 3D).
    tensors : (n,d,d)
        Symmetric matrices representing the (0,2)-tensors in ambient coords.
    scale : float
        Global scaling for all ellipsoids.
    n_theta, n_phi : int
        Sampling resolution for the ellipsoid mesh.
    colorscale : list
        Plotly-style diverging colorscale [[0, color1], [0.5, mid], [1, color2]].
    magnitude_scaling : bool
        Scale axis lengths by the eigenvalue magnitudes.
    line_color : str
        Outline colour for 2D ellipses.
    line_width : float
        Outline width for 2D ellipses.

    Returns
    -------
    fig : go.Figure
        Plotly figure showing the ellipses/ellipsoids.
    """
    points = np.asarray(points, float)
    tensors = np.asarray(tensors, float)
    n, d = tensors.shape[:2]
    if d not in (2, 3):
        raise ValueError("Only 2D or 3D tensors are supported.")

    # --- Eigen-decomposition ---
    eigvals, eigvecs = np.linalg.eigh(tensors)  # (n,d), (n,d,d)
    magnitudes = np.linalg.norm(eigvals, axis=1) ** 0.5
    max_mag = np.max(magnitudes) if np.any(magnitudes) else 1.0
    frac = 0.7 * magnitudes / max_mag

    if d == 2:
        # --- 2D Ellipses ---
        theta = np.linspace(0, 2 * np.pi, n_theta)
        unit_circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # (n_theta,2)

        fig = go.Figure()
        for p, V, E, s in zip(points, eigvecs, eigvals, frac):
            radii = scale * np.sqrt(np.abs(E))
            if magnitude_scaling:
                radii *= s
            ellipse = (unit_circle * radii) @ V.T + p
            fig.add_trace(
                go.Scatter(
                    x=ellipse[:, 0],
                    y=ellipse[:, 1],
                    mode="lines",
                    fill="toself",
                    line=dict(color=line_color, width=line_width),
                    fillcolor=line_color,
                    opacity=s,
                    showlegend=False,
                )
            )

        clean_fig(fig)
        return fig

    else:
        # --- 3D Ellipsoids ---
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi)
        st, ct = np.sin(theta), np.cos(theta)
        sp, cp = np.sin(phi), np.cos(phi)
        Xs = np.outer(st, cp)
        Ys = np.outer(st, sp)
        Zs = np.outer(ct, np.ones_like(phi))

        I, J, K = [], [], []
        for i in range(n_theta - 1):
            for j in range(n_phi - 1):
                base = i * n_phi + j
                I.append(base)
                J.append(base + 1)
                K.append(base + n_phi)
                I.append(base + 1)
                J.append(base + n_phi + 1)
                K.append(base + n_phi)

        fig = go.Figure()
        for p, V, E, s in zip(points, eigvecs, eigvals, frac):
            radii = scale * np.sqrt(np.abs(E))
            if magnitude_scaling:
                radii *= s
            ellipsoid = np.stack([Xs, Ys, Zs], axis=-1) @ np.diag(radii) @ V.T + p
            X, Y, Z = (
                ellipsoid[..., 0].ravel(),
                ellipsoid[..., 1].ravel(),
                ellipsoid[..., 2].ravel(),
            )

            fig.add_trace(
                go.Mesh3d(
                    x=X,
                    y=Y,
                    z=Z,
                    i=I,
                    j=J,
                    k=K,
                    color=line_color,
                    opacity=s,
                    flatshading=True,
                    lighting=dict(diffuse=0.9, specular=0.3),
                    showscale=False,
                )
            )

        clean_fig(fig)
        return fig


def plot_hessian_eig_lines(
    points,
    vecs_ambient,
    vals,
    colorscale=[[0, "#166dde"], [0.5, "lightgray"], [1, "#e32636"]],
    scale=1.0,
    line_width=2,
    opacity=0.9,
):
    """
    Plot bidirectional lines along Hessian eigenvectors in R^3,
    with per-point variable opacity.

    Parameters
    ----------
    points : (n,3)
        Base coordinates.
    vecs_ambient : (n,3,2)
        Ambient-space eigenvectors at each point (each column one direction).
    vals : (n,2)
        Eigenvalues corresponding to vecs_ambient.
    colorscale : list
        Plotly colorscale for signed eigenvalues.
    scale : float
        Global scaling factor for line length.
    line_width : float
        Width of the lines.
    opacity : float or (n,)
        Either a single float (uniform opacity) or an array of length n
        giving per-point opacity weights.
    """
    points = np.asarray(points)
    n = points.shape[0]

    # Ensure opacity is broadcastable to length n
    if np.isscalar(opacity):
        opacity_array = np.full(n, float(opacity))
    else:
        opacity_array = np.asarray(opacity, dtype=float)
        if opacity_array.shape != (n,):
            raise ValueError(
                f"opacity must be scalar or length {n}, got {opacity_array.shape}"
            )

    x_all, y_all, z_all, c_all, a_all = [], [], [], [], []

    for i in range(2):  # two eigenvectors per point
        dirs = vecs_ambient[:, :, i]
        mags = np.abs(vals[:, i]) * scale
        colors = vals[:, i]

        for sign in [+1, -1]:
            start = points
            end = points + sign * dirs * mags[:, None]

            # NaN-separated line segments
            x = np.column_stack((start[:, 0], end[:, 0], np.full(n, np.nan))).ravel()
            y = np.column_stack((start[:, 1], end[:, 1], np.full(n, np.nan))).ravel()
            z = np.column_stack((start[:, 2], end[:, 2], np.full(n, np.nan))).ravel()
            c = np.column_stack((colors, colors, np.full(n, np.nan))).ravel()
            a = np.column_stack(
                (opacity_array, opacity_array, np.full(n, np.nan))
            ).ravel()

            x_all.append(x)
            y_all.append(y)
            z_all.append(z)
            c_all.append(c)
            a_all.append(a)

    x_all = np.concatenate(x_all)
    y_all = np.concatenate(y_all)
    z_all = np.concatenate(z_all)
    c_all = np.concatenate(c_all)
    a_all = np.concatenate(a_all)

    # Convert opacity array into RGBA values
    from matplotlib import cm, colors

    cmap = colors.LinearSegmentedColormap.from_list(
        "custom", [c for _, c in colorscale]
    )
    norm = colors.Normalize(vmin=np.nanmin(vals), vmax=np.nanmax(vals))
    rgba = cmap(norm(c_all))
    rgba[:, 3] = a_all  # replace alpha channel with pointwise opacity
    rgba_hex = [colors.to_hex(rgba[i]) for i in range(len(rgba))]

    fig = go.Figure(
        [
            go.Scatter3d(
                x=x_all,
                y=y_all,
                z=z_all,
                mode="lines",
                line=dict(
                    width=line_width,
                    color=rgba_hex,
                ),
                hoverinfo="none",
                connectgaps=False,
            ),
        ]
    )

    fig.update_layout(
        scene=dict(aspectmode="data"),
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    clean_fig(fig)
    return fig


def clean_fig(fig):
    """
    Completely clean 2D and 3D Plotly figures:
      • Removes all colorbars (both trace-level and global coloraxes)
      • Hides axes, ticks, gridlines, labels, and backgrounds
      • Enforces equal aspect ratios for 2D plots
      • Works with multi-subplot figures containing 3D scenes
    """
    # --- Remove colorbars from traces ---
    for trace in fig.data:
        if hasattr(trace, "showscale"):
            trace.showscale = False
        if hasattr(trace, "colorbar"):
            trace.colorbar = None
        # Some 3D traces may carry shared coloraxis references
        if hasattr(trace, "coloraxis"):
            trace.coloraxis = None

    # --- Basic layout cleanup ---
    fig.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        title="",
        margin=dict(l=0, r=0, t=0, b=0),
    )

    # --- Handle all 2D subplots ---
    x_axes = [k for k in fig.layout if k.startswith("xaxis")]
    y_axes = [k for k in fig.layout if k.startswith("yaxis")]

    for k in x_axes + y_axes:
        fig.layout[k].update(
            showgrid=False,
            showticklabels=False,
            showline=False,
            zeroline=False,
            title="",
        )

    # --- Disable automatic margins ---
    for axis in fig.layout:
        if axis.startswith("xaxis") or axis.startswith("yaxis"):
            fig.layout[axis].automargin = False

    # # --- Enforce 1:1 aspect ratio for each 2D subplot ---
    for i, xk in enumerate(x_axes, start=1):
        suffix = "" if i == 1 else str(i)
        yref = f"y{suffix}"
        if f"yaxis{suffix}" in fig.layout:
            fig.layout[f"yaxis{suffix}"].update(scaleanchor=f"x{suffix}", scaleratio=1)

    # --- Handle all 3D scenes (scene, scene2, ...) ---
    for k in fig.layout:
        if k.startswith("scene"):
            fig.layout[k].update(
                xaxis=dict(
                    visible=False,
                    showbackground=False,
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    title="",
                ),
                yaxis=dict(
                    visible=False,
                    showbackground=False,
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    title="",
                ),
                zaxis=dict(
                    visible=False,
                    showbackground=False,
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    title="",
                ),
                aspectmode="data",
            )

    return fig


def plot_heatmap_2d(X, Y, Z, colorscale=None):
    """
    Plot a smooth 2D heatmap for scalar function values on a grid.
    X, Y, Z : 2D arrays (same shape)
    """
    if colorscale is None:
        colorscale = DEFAULT_COLORSCALE

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=X[0, :],
            y=Y[:, 0],
            z=Z,
            colorscale=colorscale,
            zmin=-np.nanmax(np.abs(Z)),
            zmax=np.nanmax(np.abs(Z)),
            showscale=False,
        )
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


# ---------- Camera + projection helpers ----------


def _normalize(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def _build_view(camera):
    # camera = {"eye":{x,y,z}, "center":{x,y,z}, "up":{x,y,z}}
    E = np.array([camera["eye"][k] for k in ("x", "y", "z")], dtype=float)
    C = np.array([camera["center"][k] for k in ("x", "y", "z")], dtype=float)
    U = np.array([camera["up"][k] for k in ("x", "y", "z")], dtype=float)

    f = _normalize(C - E)  # forward
    r = _normalize(np.cross(f, U))  # right
    u = np.cross(r, f)  # true up

    R = np.stack([r, u, -f], axis=0)  # rows: r, u, -f
    t = -R @ E

    V = np.eye(4)
    V[:3, :3] = R
    V[:3, 3] = t
    return V


def _perspective_matrix(fov_y_deg=45.0, aspect=1.0, near=0.1, far=100.0):
    f = 1.0 / np.tan(0.5 * np.deg2rad(fov_y_deg))
    P = np.zeros((4, 4))
    P[0, 0] = f / aspect
    P[1, 1] = f
    P[2, 2] = (far + near) / (near - far)
    P[2, 3] = (2 * far * near) / (near - far)
    P[3, 2] = -1.0
    return P


def _ortho_matrix(l, r, b, t, n=0.1, f=100.0):
    P = np.eye(4)
    P[0, 0] = 2.0 / (r - l)
    P[1, 1] = 2.0 / (t - b)
    P[2, 2] = -2.0 / (f - n)
    P[0, 3] = -(r + l) / (r - l)
    P[1, 3] = -(t + b) / (t - b)
    P[2, 3] = -(f + n) / (f - n)
    return P


def _project_points(
    points3d,
    camera,
    mode="orthographic",
    fov_y_deg=45.0,
    aspect=1.0,
    near=0.1,
    far=100.0,
    ortho_scale=1.0,
):
    """
    Returns Nx2 projected coordinates in NDC ([-1,1]x[-1,1]).
    """
    V = _build_view(camera)
    if mode == "perspective":
        P = _perspective_matrix(fov_y_deg=fov_y_deg, aspect=aspect, near=near, far=far)
    else:
        # symmetric ortho window around the target
        l = -ortho_scale
        r = ortho_scale
        b = -ortho_scale
        t = ortho_scale
        P = _ortho_matrix(l, r, b, t, n=near, f=far)

    X = np.c_[np.asarray(points3d, float), np.ones(len(points3d))]
    clip = (P @ (V @ X.T)).T
    if mode == "perspective":
        w = np.maximum(np.abs(clip[:, 3]), 1e-12)
        ndc = clip[:, :3] / w[:, None]
    else:
        ndc = clip[:, :3]
    return ndc[:, :2]


def overpic_labels(
    fig,
    label_function,
    *,
    stretch_x=1.0,
    stretch_y=1.0,
    offset_x=0.0,
    offset_y=0.0,
    tol=1e-6,
    include_2d=True,
    include_3d=True,
    return_strings=False,
):
    """
    Emit Overpic \\put(...) labels centred on each subplot/scene in a Plotly figure.

    Works for 2D (xaxis*/yaxis*) and 3D (scene*) subplots.
    """

    # --- Figure size → Overpic %
    fig_w = fig.layout.width or 1000
    fig_h = fig.layout.height or 800
    if fig_w >= fig_h:
        horiz_span = 100.0
        vert_span = 100.0 * (fig_h / fig_w)
    else:
        vert_span = 100.0
        horiz_span = 100.0 * (fig_w / fig_h)

    panels = []
    layout_dict = fig.layout.to_plotly_json()  # <-- safe dict view

    # --- 2D domains (xaxisN/yaxisN pairs)
    if include_2d:
        x_domains, y_domains = {}, {}
        for k, v in layout_dict.items():
            if not isinstance(v, dict):
                continue
            if k.startswith("xaxis"):
                m = re.match(r"xaxis(\d*)$", k)
                idx = int(m.group(1)) if m and m.group(1) else 1
                dom = v.get("domain")
                if dom:
                    x_domains[idx] = tuple(dom)
            elif k.startswith("yaxis"):
                m = re.match(r"yaxis(\d*)$", k)
                idx = int(m.group(1)) if m and m.group(1) else 1
                dom = v.get("domain")
                if dom:
                    y_domains[idx] = tuple(dom)
        for idx in sorted(set(x_domains) & set(y_domains)):
            x0, x1 = x_domains[idx]
            y0, y1 = y_domains[idx]
            panels.append(
                dict(
                    kind="2d",
                    key=f"xaxis{'' if idx==1 else idx}",
                    cx=0.5 * (x0 + x1),
                    cy=0.5 * (y0 + y1),
                )
            )

    # --- 3D scenes
    if include_3d:
        for k, v in layout_dict.items():
            if not (k.startswith("scene") and isinstance(v, dict)):
                continue
            dom = v.get("domain")
            if dom and "x" in dom and "y" in dom:
                x0, x1 = dom["x"]
                y0, y1 = dom["y"]
                panels.append(
                    dict(kind="3d", key=k, cx=0.5 * (x0 + x1), cy=0.5 * (y0 + y1))
                )

    if not panels:
        raise ValueError("No subplot or scene domains found in figure layout.")

    # --- Group into rows/cols by centre positions
    panels.sort(key=lambda p: (-p["cy"], p["cx"]))
    rows = []
    for p in panels:
        for row in rows:
            if abs(row[0]["cy"] - p["cy"]) < tol:
                row.append(p)
                break
        else:
            rows.append([p])
    for row in rows:
        row.sort(key=lambda p: p["cx"])

    ordered = []
    for r, row in enumerate(rows, start=1):
        for c, p in enumerate(row, start=1):
            p["row"], p["col"] = r, c
            ordered.append(p)

    wants_two_args = len(inspect.signature(label_function).parameters) >= 2
    out = []
    for k, p in enumerate(ordered, start=1):
        xc = 0.5 + (p["cx"] - 0.5) * stretch_x
        yc = 0.5 + (p["cy"] - 0.5) * stretch_y
        x_pct = xc * horiz_span - offset_x
        y_pct = yc * vert_span - offset_y
        label = (
            label_function(p["row"], p["col"]) if wants_two_args else label_function(k)
        )
        out.append(f"\\put({x_pct:.1f},{y_pct:.1f}){{\\makebox(0,0)[c]{{{label}}}}}")

    if return_strings:
        return out
    for s in out:
        print(s)


def plot_tangent_planes_3d(tangent_bundle, data_input, angle=(3, 3, 4), zoom=None):
    data = data_input[:, :3]
    nbrs = NearestNeighbors(n_neighbors=4, algorithm='auto').fit(data)
    distances, _ = nbrs.kneighbors(data)
    rho = distances[:, 1:].mean(axis=1)
    tangent_bundle_rescaled = 0.4 * np.median(rho) * tangent_bundle[:, :3]

    n = data.shape[0]

    p1 = data + tangent_bundle_rescaled[:, :, 0] + tangent_bundle_rescaled[:, :, 1]
    p2 = data + tangent_bundle_rescaled[:, :, 1] - tangent_bundle_rescaled[:, :, 0]
    p3 = data - tangent_bundle_rescaled[:, :, 0] - tangent_bundle_rescaled[:, :, 1]
    p4 = data - tangent_bundle_rescaled[:, :, 1] + tangent_bundle_rescaled[:, :, 0]
    all_p = np.concatenate((p1, p2, p3, p4), axis=0)

    indices = np.arange(n)
    simplices_1 = np.stack((indices, indices + n, indices + 2 * n)).T
    simplices_2 = np.stack((indices, indices + 2 * n, indices + 3 * n)).T
    simplices = np.concatenate((simplices_1, simplices_2), axis=0)
    simplices = simplices.reshape(2 * n, 3)

    fig = ff.create_trisurf(x=all_p[:, 0],
                            y=all_p[:, 1],
                            z=all_p[:, 2],
                            colormap=["#8ab6ee", "#e53b4a"],
                            simplices=simplices,
                            show_colorbar=False,
                            plot_edges=False,
                            title=None)

    angle_a = np.array(angle, dtype=float)
    angle_a /= np.linalg.norm(angle_a)

    scale = np.linalg.norm(data, axis=1).max()
    if zoom is None:
        zoom = 1 / scale
    scale *= zoom
    camera = dict(
        eye=dict(x=6 * angle_a[0] / scale,
                 y=6 * angle_a[1] / scale,
                 z=8 * angle_a[2] / scale),
        center=dict(x=0, y=0, z=0),
        up=dict(x=0, y=0, z=1)
    )

    fig.update_layout(width=1000, height=700, margin=dict(l=0, r=0, t=0, b=0))

    fig.update_layout(scene_camera=camera,
                      scene=dict(aspectmode='data',
                                 xaxis=dict(visible=False),
                                 yaxis=dict(visible=False),
                                 zaxis=dict(visible=False)))

    return fig
