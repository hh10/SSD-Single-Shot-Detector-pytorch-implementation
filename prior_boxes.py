import torch

from math import sqrt
from collections import namedtuple
from typing import Dict

SsdFeatureMapParams = namedtuple("SsdFeatureMapParams", ["scale", "extra_scale", "size", "ratios"])


class PriorBox(object):
    """
    Compute priorbox coords in normalized center-offset form, i.e., [c_x, c_y, w, h] for each
    feature map clamped to [0, 1] (note some people don't do this clipping).
    """
    def __init__(self, img_dim: int, fm_params: Dict[str, SsdFeatureMapParams]) -> torch.Tensor:
        super(PriorBox, self).__init__()
        self.image_size = img_dim
        self.fm_params = fm_params

    def forward(self):
        mean = []
        for map_id, params in self.fm_params.items():
            map_size, map_scale = params.size, params.scale
            for ratio in params.ratios:
                for map_pixel in range(map_size * map_size):
                    # Get cartesian xy-coordinates in this feature map.
                    y, x = divmod(map_pixel, map_size)
                    unit_scaled_center_x = (x + 0.5) / map_size
                    unit_scaled_center_y = (y + 0.5) / map_size
                    unit_scaled_w = map_scale * sqrt(ratio)
                    unit_scaled_h = map_scale / sqrt(ratio)
                    mean += [unit_scaled_center_x, unit_scaled_center_y, unit_scaled_w, unit_scaled_h]
            # Add extra_scale for this feature map.
            extra_scale = params.extra_scale
            for map_pixel in range(map_size * map_size):
                # Get cartesian xy-coordinates in this feature map.
                y, x = divmod(map_pixel, map_size)
                unit_scaled_center_x = (x + 0.5) / map_size
                unit_scaled_center_y = (y + 0.5) / map_size
                unit_scaled_w = extra_scale
                unit_scaled_h = extra_scale
                mean += [unit_scaled_center_x, unit_scaled_center_y, unit_scaled_w, unit_scaled_h]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        output.clamp_(max=1, min=0)  # comment later
        return output


def test_priors():
    SSD_FEATURE_MAPS_SETTINGS = {
        "x128": SsdFeatureMapParams(scale=0.01, extra_scale=sqrt(0.01 * 0.07), size=128, ratios=[1.5, 0.75]),
        "x64": SsdFeatureMapParams(scale=0.07, extra_scale=sqrt(0.07 * 0.12), size=64, ratios=[2.5, 1.25, 0.75]),
        "x32": SsdFeatureMapParams(scale=0.12, extra_scale=sqrt(0.12 * 0.25), size=32, ratios=[2.5, 1.25, 0.75, 4]),
        "x16": SsdFeatureMapParams(scale=0.25, extra_scale=sqrt(0.25 * 0.45), size=16, ratios=[2.5, 1.25, 0.75, 4]),
        "x8": SsdFeatureMapParams(scale=0.45, extra_scale=sqrt(0.45 * 0.6), size=8, ratios=[2.5, 1.25, 0.75, 4]),
    }
    priors = PriorBox(512, SSD_FEATURE_MAPS_SETTINGS).forward()
    assert priors.shape == torch.Size([72256, 4]), print(f"Expected prior shape: {[72256, 4]}\nGot: {priors.shape}")


if __name__ == "__main__":
    test_priors()
