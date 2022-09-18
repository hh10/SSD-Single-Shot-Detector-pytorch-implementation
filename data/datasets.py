from .coco import COCODetection, COCOObjectTypes
from .augmentations import AugmentedBoxes

def load_dataset(dataset_name: str, image_size: int, random_crops: bool, obj_class_category: str = "", load_only_metadata:bool=False):
    image_transforms = AugmentedBoxes(image_size, random_crops)
    if dataset_name == 'COCO':
        supp_obj_cats_dict, objs_subset = COCOObjectTypes.__members__, []
        if obj_class_category:
            assert obj_class_category in supp_obj_cats_dict, print(f'{obj_class_category} not specified in COCOObjectTypes. \
                                                                    Please create a category with desired object classes in data/coco.py and give its name')
            objs_subset = supp_obj_cats_dict[obj_class_category].value
        dataset = COCODetection(image_transforms, objs_subset, load_only_metadata)
    else:
        raise NotImplementedError(f"{dataset_name} not supported!")
    return dataset
