from .datasets import load_dataset
from .utils import jaccard, draw_gt_bounding_boxes, draw_pred_bounding_boxes, get_colors_and_legend
from .augmentations import denormalize, normalization_transform  # todo(hh): remove and add to datasets later

import torch
from typing import List, Tuple


def detection_data_collate(batch) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Custom collate fn for batches of images in detection datasets as they each 
    have different number of associated object annotations.
    Args:
        batch: A tuple of tensor images and lists of annotations
    Returns: (images, targets)
        images: batch of images stacked on their 0 dim
        targets: [tensor([bbox_coords, label]), ...] for images stacked on 0 dim
    """
    return torch.stack([sample[0] for sample in batch], 0), [sample[1] for sample in batch]
