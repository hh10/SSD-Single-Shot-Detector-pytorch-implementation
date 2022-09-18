import torch
import torchvision.datasets as datasets
from torchvision.utils import make_grid, save_image

from .augmentations import AugmentedBoxes, denormalize
from .utils import draw_gt_bounding_boxes

import numpy as np
import os
from PIL import Image
from typing import List, Tuple, Dict
from enum import Enum

class COCOObjectTypes(Enum):
    TRAFFIC = ["bicycle\n", "car\n", "motorcycle\n", "airplane\n", "bus\n", "train\n", "truck\n", "boat\n", "bird\n"]

# dataset config
COCO_ROOT = 'data/coco'
IMAGES = 'train2017'
INSTANCES_SET = f'annotations/instances_{IMAGES}.json'  
LABELS = 'coco_labels.txt'


class COCODetection(datasets.CocoDetection):
    """MS Coco Detection Dataset <http://mscoco.org/dataset/#detections-challenge2016>.
    Args:
        image_transform: the transform object that should be applied to dataset's PIL images.
        class_list: classes to allow in the dataset. If empty, all classes are allowed; else
                    images with only the provided classes are kept and bboxes not belonging to
                    any of the provided classes are removed.

    """
    def __init__(self, image_transform, class_list, load_only_metadata:bool = False):
        if not load_only_metadata:
            super(COCODetection, self).__init__(os.path.join(COCO_ROOT, IMAGES), os.path.join(COCO_ROOT, INSTANCES_SET))
        self.image_transform = image_transform
        self.label_map, self._category_map = self.get_obj_to_label_category(os.path.join(COCO_ROOT, LABELS), class_list)
        self.classes = list(self.category_map.values())
        if class_list:
            assert len(set(self.classes) - set(class_list)) == 0, print(f"Expected classes: {class_list}\nLoaded classes: {self.classes}")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ]:
        """
        Returns: (image, target)
            target: bbox (in normalized corner pt form) and labels.
        """
        target = None
        while target is None:
            # get PIL images and annotations dict from json
            image, anns = super(COCODetection, self).__getitem__(index)
            # transform them into torch types suitable for dataloading
            image_t, target = self.__prepare_item(image, anns)
            index = (index + 1) % len(self.ids)
        return image_t, target

    def __prepare_item(self, image: Image, anns: dict):
        """
        Returns: (image, target).
            image: normalized image in CxWxH format
            target: hstacked tensor with normalized corner pt form bbox and labels such that
                    all_bboxes_coords = target[:, :4] and all_labels = target[:, :4].
        """
        target = self.annotation_transform(anns, image.size, self.label_map)
        if not target:
            return None, None
        target = np.array(target)
        image, boxes, labels = self.image_transform(image, target[:, :4], target[:, 4])
        target = torch.hstack((boxes, labels.unsqueeze(1)))
        assert target.shape[-1] == 5, print("Target of shape:", target.shape)
        return image, target

    @property
    def category_map(self):
        return self._category_map

    @staticmethod
    def annotation_transform(anns: dict, image_size: Tuple[int, int], label_map: Dict[int, int]):
        """
        Transform for the annotations dict to get bounding box coords and label in desired format.
        Args:
            anns: COCO json annotation as a python dict.
            image_shape: width, height
        Returns:
            list containing lists of normalized bounding box coords [bbox coords, label_idx] of
            alllowed object categories. bbox coords are in integer pt form [tl_x, tl_y, br_x, br_y].
        """
        width, height = image_size
        scale = np.array([width, height, width, height])
        labeled_bboxes = []
        for ann in anns:
            if 'bbox' not in ann or ann['category_id'] not in label_map:  
                # no bbox in image or bbox category not among the allowed classes
                continue
            bbox = ann['bbox']  # bbox in [xmin, ymin, width, height] format.
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            label_id = label_map[ann['category_id']]
            labeled_bboxes.append(list(np.array(bbox)/scale) + [label_id])
        return labeled_bboxes  # [[xmin, ymin, xmax, ymax, label_idx], ...]

    @staticmethod
    def get_obj_to_label_category(label_file: str, class_list: List[str]) -> Tuple[Dict[int, int], Dict[int, str]]:
        """
        Each file in label_file contains [cat_id (as found in annotations.json), label_id, cat_name].
        The cat_ids and label_ids are different because COCO defines 91 classes (cat_ids) but the
        datasets contain only a subset (80 label_ids) classes.
        After data loading, only label_ids are used.
        Args:
            class_list: categories to allow as keys in returning maps. If empty, all classes are allowed.
        Returns:
            label_map: Dict from cat_id (can have gaps) to label_id (continuous integers for softmax).
            category_map: Dict from label_id to cat_name
        """
        label_map, category_map, label_id = {}, {}, 0
        lines = open(label_file, 'r')
        for line in lines:
            ids = line.split(',')
            if not class_list:
                # consider all classes, class_ids range = [0, 79]
                label_id = int(ids[1]) - 1
                label_map[int(ids[0])] = label_id
                category_map[label_id] = ids[2]
            elif ids[2] in class_list:
                # class_list is provided, so only consider categories in the list
                label_map[int(ids[0])] = label_id
                category_map[label_id] = ids[2]
                label_id += 1
        return label_map, category_map

    # public: getter functions
    def get_classes(self):
        return self.classes


def preview_dataset(class_list: List[str] = []):
    img_size, num_images = 128, 32
    cocoDet = COCODetection(image_transform=AugmentedBoxes(img_size, True), class_list=class_list)
    print(f"Dataset #classes: {len(cocoDet.get_classes())} #images: {len(cocoDet)}")
    images_t, targets_t = [], []
    for bi, (img, gt) in enumerate(cocoDet):
        assert img.shape == torch.Size([3, img_size, img_size])
        if bi > num_images - 1:
            print(f"Dataset images shape: {img.shape}, target shape: {gt.shape}")
            break
        images_t.append(img)
        targets_t.append(gt)
    images_t = torch.stack(images_t)
    pmetadata = '_'.join([cl[:-1] for cl in class_list])
    save_image(make_grid(images_t), f"/tmp/coco_dataset_images_normalized_{pmetadata}.png")
    images_t = denormalize(images_t)  # denormalize
    bimages = draw_gt_bounding_boxes(images_t, targets_t, cocoDet.category_map)
    save_image(make_grid(images_t), f"/tmp/coco_dataset_images_{pmetadata}.png")
    save_image(make_grid(bimages), f"/tmp/coco_dataset_annotated_images_{pmetadata}.png")


if __name__ == '__main__':
    # from parent folder run: python3 -m data.coco
    preview_dataset()
    preview_dataset(["person\n", "zebra\n", "airplane\n", "car\n"])
