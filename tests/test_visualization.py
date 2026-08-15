import numpy as np
import pytest
from matplotlib.axes import Axes

from src import visualization
from src.visualization import save_intensity_image


def test_visualization_requires_shared_scale_uses_equal_aspect_and_writes_png(tmp_path, monkeypatch):
    image = np.array([[0.0, 1.0], [2.0, 3.0]])
    output = tmp_path / "map.png"
    with pytest.raises(ValueError, match="shared vmin/vmax"):
        save_intensity_image([0, 1], [0, 1], image, output, "title", "x", "z")
    aspects = []
    subplot_options = []
    original_imshow = Axes.imshow
    original_subplots = visualization.plt.subplots
    def recording_subplots(*args, **kwargs):
        subplot_options.append(kwargs)
        return original_subplots(*args, **kwargs)
    def recording_imshow(self, *args, **kwargs):
        aspects.append(kwargs.get("aspect"))
        return original_imshow(self, *args, **kwargs)
    monkeypatch.setattr(visualization.plt, "subplots", recording_subplots)
    monkeypatch.setattr(Axes, "imshow", recording_imshow)
    save_intensity_image(
        np.array([0, 1]), np.array([0, 1]), image, output,
        "title", "x [mm]", "z [mm]", vmin=0.0, vmax=3.0,
    )
    assert output.is_file()
    assert output.stat().st_size > 0
    assert aspects == ["equal"]
    assert subplot_options == [{"figsize": (7.2, 5.0), "layout": "constrained"}]
