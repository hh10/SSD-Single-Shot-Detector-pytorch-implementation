# dataset independent utilities
import torch
import torchvision.transforms as transforms

import numpy as np
from typing import Tuple, Union, Dict, List
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


# util modules
class RandomCrop(object):
    """
    Generates random image crops with minimum jaccard overlaps. Useful for training
    model to identify even partially captured object in images.
    Read section Data augmentation in https://arxiv.org/pdf/1512.02325.pdf.
    """
    def __init__(self, min_jaccard_overlaps: list = []):
        jaccard_overlaps = min_jaccard_overlaps or [(0.1, np.inf), (0.3, np.inf), (0.7, np.inf), (0.9, np.inf)]
        self.sample_options = np.array([
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            *jaccard_overlaps,
            # randomly sample a patch
            (-np.inf, np.inf),
        ], dtype=object)

    def __call__(self, image, boxes: torch.Tensor, labels: torch.Tensor):
        """
        Return random crops with adjusted bounding boxes and labels subset as applicable for the sampled crop.
        Args: (image, boxes, labels)
            boxes: bboxes coords w.r.t. to image in integer pt form [tl_x, tl_y, br_x, br_y].
        Returns: (cropped_image, boxes_subset, labels_subset)
            boxes: bboxes coords w.r.t. to cropped_image in integer pt form [tl_x, tl_y, br_x, br_y].
        """
        w_i, h_i = image.size
        image_arr = np.array(image)
        while True:
            # randomly choose a mode
            mode = np.random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            for _ in range(50):
                current_image = image_arr
                w_c = np.random.uniform(0.3 * w_i, w_i)
                h_c = np.random.uniform(0.3 * h_i, h_i)
                # note that here the image can have aspect ratio neq 1.
                # random crop aspect ratio must not be wild, keep btw .5 & 2
                if h_c / w_c < 0.5 or h_c / w_c > 2:
                    continue
                # pick a upper left crop corner such that a crop of [w_c, h_c] can be made.
                left_c = np.random.uniform(w_i - w_c)
                top_c = np.random.uniform(h_i - h_c)
                # convert crop to integer rect [x1, y1, x2, y2]
                rect = np.array([int(left_c), int(top_c), int(left_c + w_c), int(top_c + h_c)])
                # calculate IoU (jaccard overlap) btw the cropped and gt boxes
                overlap = jaccard(boxes, rect)
                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue
                # cut the crop from the image
                crop = current_image[rect[1]:rect[3], rect[0]:rect[2], :]
                
                # Now find GT boxes that have there center in this sampled crop
                # - find centers of the gt bounding boxes
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
                # - mask all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                # - mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                # - mask in that both m1 and m2 are true
                mask = m1 * m2
                # - have any GT boxes with center in crop? if not, try another random crop
                if not mask.any():
                    continue
                # take only matching gt boxes and labels
                crop_boxes = boxes[mask, :].copy()
                crop_labels = labels[mask]

                # Note that the crop_boxes coords are still w.r.t. to the full image, next changing that.
                # - some GT boxes may be partially outside the image crop on the left and top, fixing that.
                crop_boxes[:, :2] = np.maximum(crop_boxes[:, :2], rect[:2])
                # - shift crop_boxes top-left coords by the top-left of the crop.
                crop_boxes[:, :2] -= rect[:2]
                # - some GT boxes may be partially outside the image crop on the right and bottom, fixing that.
                crop_boxes[:, 2:] = np.minimum(crop_boxes[:, 2:], rect[2:])
                # - shift crop_boxes bottom-right coords by the top-left of the crop.
                crop_boxes[:, 2:] -= rect[:2]
                return Image.fromarray(crop, 'RGB'), crop_boxes, crop_labels


# util functions
def jaccard(boxes_a: Union[np.array, torch.Tensor], box_b: Union[np.array, torch.Tensor]) -> np.array:
    """
    Computes the jaccard overlap of two sets of boxes, i.e., the intersection
    over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        boxes_a: Multiple bounding boxes, Shape: [num_boxes, 4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: array of overlaps between boxes_a and box_b.
            Shape: [boxes_a.shape[0], boxes_a.shape[1]]
    """
    assert torch.is_tensor(boxes_a) == torch.is_tensor(box_b), print("Inputs type mismatch")
    intersect_func = intersect_ttensor if torch.is_tensor(boxes_a) else intersect_numpy
    inter = intersect_func(boxes_a, box_b)
    areas_a = ((boxes_a[:, 2]-boxes_a[:, 0]) * (boxes_a[:, 3]-boxes_a[:, 1]))
    if torch.is_tensor(boxes_a):
        areas_a = areas_a.unsqueeze(1).expand_as(inter)
        area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)
    else:
        area_b = ((box_b[2]-box_b[0]) * (box_b[3]-box_b[1]))
    union = areas_a + area_b - inter
    return inter / union  # todo(hh): elaborate on output shape here


def intersect_ttensor(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """
    We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def intersect_numpy(box_a: np.array, box_b: np.array) -> np.array:
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def to_absolute_coords(boxes: Union[np.array, torch.Tensor], image_size: Tuple[int, int]):
    width, height = image_size
    out_boxes = torch.empty_like(boxes) if torch.is_tensor(boxes) else np.empty_like(boxes)
    out_boxes[:, 0] = boxes[:, 0] * width
    out_boxes[:, 2] = boxes[:, 2] * width
    out_boxes[:, 1] = boxes[:, 1] * height
    out_boxes[:, 3] = boxes[:, 3] * height
    return out_boxes


def to_percent_coords(boxes: Union[np.array, torch.Tensor], image_size: Tuple[int, int]):
    width, height = image_size
    out_boxes = torch.empty_like(boxes) if torch.is_tensor(boxes) else np.empty_like(boxes)
    out_boxes[:, 0] = boxes[:, 0] / width
    out_boxes[:, 2] = boxes[:, 2] / width
    out_boxes[:, 1] = boxes[:, 1] / height
    out_boxes[:, 3] = boxes[:, 3] / height
    return out_boxes


# visualization functions
def get_colors_and_legend(c):
    cmap, N = plt.get_cmap(c), 10  # cmap.N
    cmap_a = cmap(np.arange(N))
    cmap_a[:, -1] = [0.3] * N
    cmap_a = np.vstack((cmap_a, [[0, 0, 0, 1]]*N))
    colors = [(cmap_a[i]*255).astype(int) for i in range(0, N)]
    return colors


def draw_gt_bounding_boxes(images_t: torch.Tensor, targets_t: List[torch.Tensor], category_map: Dict[int, str], corner_radius: int=0, black_line: bool=False) -> torch.Tensor:
    # todo(hh): check for len first
    assert targets_t[0].dim() == 2 and targets_t[0].shape[1] == 5, print("Targets should be [BS, 5] in shape, found:", targets_t[0].shape)
    colors = get_colors_and_legend('tab10')
    font = ImageFont.load_default()
    bimages_t = []
    for image_t, target_t in zip(images_t, targets_t):
        image = transforms.ToPILImage()(image_t)
        draw = ImageDraw.Draw(image)
        aboxes = to_absolute_coords(target_t[:, :4], image.size).detach().cpu().numpy()
        for ti, (abox, label) in enumerate(zip(aboxes, target_t[:, 4])):
            color = tuple(colors[ti % len(colors)])
            draw.rounded_rectangle(((abox[0], abox[1]), (abox[2], abox[3])), radius=corner_radius, outline="black" if black_line else color, width=1 if black_line else 2 if corner_radius else 3)
            draw.text((abox[0]+10, abox[1]+10), f'{int(label.item())}{category_map[label.item()]}', fill="black" if black_line else color, font=font)
        bimages_t.append(transforms.ToTensor()(image))
    return torch.stack(bimages_t)


def draw_pred_bounding_boxes(images_t: torch.Tensor, preds_t: torch.Tensor, category_map: Dict[int, str]) -> torch.Tensor:
    colors = get_colors_and_legend('tab10')
    font = ImageFont.load_default()
    bimages_t = []
    for image_t, pred_t in zip(images_t, preds_t):
        image = transforms.ToPILImage()(image_t)
        draw = ImageDraw.Draw(image)
        for class_id, label in category_map.items():
            dets = pred_t[class_id, :]  # preds from detector do not have background class
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            # dets is the [num_boxes_of_class, score+bbox_coords]
            aboxes = to_absolute_coords(dets[:, 1:], image.size)
            for ti, abox in enumerate(aboxes):
                color = tuple(colors[ti % len(colors)])
                draw.rectangle(((abox[0], abox[1]), (abox[2], abox[3])), outline=color, width=3)
                draw.text((abox[0]+10, abox[1]+10), label, fill="black", font=font)
        bimages_t.append(transforms.ToTensor()(image))
    return torch.stack(bimages_t)
