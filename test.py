import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import numpy as np
import os
from datetime import datetime
import enlighten
import argparse

from data import load_dataset, detection_data_collate
from ssd import build_ssd_and_priors
from detector import Detector
from metrics import detection_eval, plot_PRcurve

device = torch.device("cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")


def test(model_path: str, dataset_name: str, image_size: int, obj_class_category: str = "", test_run: bool = False):
    eval_time = datetime.now()
    eval_dir = os.path.join(os.path.dirname(model_path), "evals", f'{eval_time.strftime("%Y-%m-%d")}_{eval_time.strftime("%H:%M:%S")}')
    os.makedirs(eval_dir)

    dataset = load_dataset(dataset_name, image_size, random_crops=True, obj_class_category=obj_class_category)
    num_object_classes = len(dataset.get_classes()) + 1  # +1 for background or no object class
    dcat_map = dataset.category_map

    # get model
    ssd_net, priors = build_ssd_and_priors(image_size, num_object_classes)
    if model_path:
        print("Loading model from", model_path)
        ckpt = torch.load(model_path, map_location=device)
        ssd_net.load_state_dict(ckpt)
    ssd_net, priors = ssd_net.to(device), priors.to(device)  # transfer model to device
    detection_head = Detector(num_object_classes, top_k=10, conf_thresh=0.1, nms_thresh=0.45, bkg_label=0)
    ssd_net.eval()

    test_dl = DataLoader(dataset, 256, num_workers=8, shuffle=False, collate_fn=detection_data_collate,
                         pin_memory=True)
    recalls, precisions = [], []
    progress_manager = enlighten.get_manager()
    batch_progress = progress_manager.counter(total=len(test_dl), desc="\tBatches", unit="images", leave=False)
    for bi, (timages, ttargets) in enumerate(test_dl):
        if test_run and bi > 2:
            break
        eval_results = detection_eval(timages, ttargets, priors, num_object_classes, ssd_net, detection_head, dcat_map, device)
        recalls.append(eval_results.recall)
        precisions.append(eval_results.precision)
        batch_progress.update()
    progress_manager.stop()
    # save first batch images
    save_image(eval_results.images_with_GT, os.path.join(eval_dir, "GT_images.png"))
    save_image(eval_results.images_with_preds, os.path.join(eval_dir, "Pred_images.png"))
    # print map table in a text file and plot accumulated prec-recall curve
    recalls = np.mean(np.array(recalls), axis=0)
    precisions = np.mean(np.array(precisions), axis=0)
    plot_PRcurve(recalls, precisions, dcat_map, eval_dir)
    with open(os.path.join(eval_dir, "mAPs.txt"), "w") as fw:
        fw.write(f"mAP: {np.array(eval_results.mAP).mean()} \n")
        for mi, v in enumerate(eval_results.mAP):
            fw.write(f"{dcat_map[mi]}: {float(v)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Program to test a trained SSD model against a dataset of given size.')
    parser.add_argument("model_path", help="SSD model (.pth) to finetune.")
    parser.add_argument("--dataset", default="COCO", help="Dataset to train SSD for, currently only accepts {COCO}.")
    parser.add_argument("--image_size", type=int, default=128, help="Size (width==height in pixels) of the SSD input.")
    parser.add_argument("--dataset_subcategory", default="TRAFFIC", help='''Subset of specified dataset classes to train the SSD for.
                                                       Currently, allowing {TRAFFIC} for COCO dataset.
                                                       Specify more subcategories in data/coco/coco.py for allowing more.''')
    parser.add_argument("--test_run", action="store_true", help="Whether to run a minimal (5 epochs with 5 batches) training run for testing end-to-end functionality.")
    args = parser.parse_args()

    test(args.model_path, args.dataset, args.image_size, args.dataset_subcategory, args.test_run)
