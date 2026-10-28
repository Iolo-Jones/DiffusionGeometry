import numpy as np

from diffusion_geometry.visualisation import (
    linked_html,
    match_scene_ranges,
    plot_quiver_3d,
    plot_scatter_3d,
)


def test_match_scene_ranges_uses_union_of_data():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(50, 3))
    a = plot_scatter_3d(data)
    b = plot_quiver_3d(data, rng.normal(size=(50, 3)), scale=2.0)
    match_scene_ranges(a, b)
    for axis in "xyz":
        assert a.layout.scene[f"{axis}axis"].range == b.layout.scene[f"{axis}axis"].range
    # The quiver tips reach beyond the points, so the shared range must cover them
    assert a.layout.scene.xaxis.range[1] >= data[:, 0].max()


def test_linked_html_groups_figures_and_loads_plotly_once():
    data = np.random.default_rng(1).normal(size=(20, 3))
    divs = linked_html([plot_scatter_3d(data), plot_scatter_3d(data)], group="g")
    assert len(divs) == 2
    assert all('data-camera-group="g"' in d for d in divs)
    assert "cdn.plot.ly" in divs[0] and "cdn.plot.ly" not in divs[1]
    assert all("plotly_relayouting" in d for d in divs)
