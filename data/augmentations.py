import torch
import torchvision.transforms as transforms

from .utils import RandomCrop, to_absolute_coords, to_percent_coords


class AugmentedBoxes(object):
    """
    Module to provide image crops with containing bounding box and labels info.
    """
    def __init__(self, image_size: int, random_crops: bool):
        self.random_crop_sampler = RandomCrop() if random_crops else None
        image_transforms = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomAutocontrast(p=0.5),
            transforms.RandomAdjustSharpness(sharpness_factor=1, p=0.5),
            transforms.ToTensor()]
        # todo(hh): refactor to get change normalization from dataset type - URGENT
        image_transforms.append(normalization_transform())
        self.image_transforms = transforms.Compose(image_transforms)

    def __call__(self, image, boxes, labels):
        if self.random_crop_sampler:
            boxes = to_absolute_coords(boxes, image.size)
            # randomly crop images -> convert the boxes to pixel coords before this!
            image, boxes, labels = self.random_crop_sampler(image, boxes, labels)
            # renormalize the box coords as per cropped image shape
            boxes = to_percent_coords(boxes, image.size)
        # resize images and photometrically augment images, no geometric transforms from now on!
        image = self.image_transforms(image)
        return image, torch.FloatTensor(boxes), torch.FloatTensor(labels)


# todo(hh): make (de/)normalize dataset dependent
def normalization_transform():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

def denormalize(images):
    means,  = torch.tensor((0.5, 0.5, 0.5)).reshape(1, 3, 1, 1)
    std_devs = torch.tensor((0.5, 0.5, 0.5)).reshape(1, 3, 1, 1)
    return images.to('cpu') * std_devs + means
