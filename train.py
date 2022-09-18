import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torch.onnx
from torch.utils.data import random_split

from data import load_dataset, detection_data_collate, denormalize
from ssd import build_ssd_and_priors
from loss import MultiBoxLoss
from detector import Detector
from metrics import plot_PRcurve, detection_eval

import enlighten
from datetime import datetime
import os
import numpy as np
import argparse

device = torch.device("cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")


def train(dataset_name: str, image_size: int, obj_class_category: str = "", model_path: str = "", test_run: bool = False):
    exp_time = datetime.now()
    exp_dir = os.path.join("/tmp" if test_run else "experiments",
                           f"{dataset_name}_{image_size}x{image_size}", exp_time.strftime("%Y-%m-%d"), exp_time.strftime("%H:%M:%S"))

    dataset = load_dataset(dataset_name, image_size, random_crops=True, obj_class_category=obj_class_category)
    num_object_classes = len(dataset.get_classes()) + 1  # +1 for background or no object class

    # get model
    ssd_net, priors = build_ssd_and_priors(image_size, num_object_classes)
    if model_path:
        print("Loading model from", model_path)
        ckpt = torch.load(model_path, map_location=device)
        ssd_net.load_state_dict(ckpt)
    ssd_net, priors = ssd_net.to(device), priors.to(device)  # transfer model to device
    detection_head = Detector(num_object_classes, top_k=10, conf_thresh=0.1, nms_thresh=0.45, bkg_label=0)

    # training params
    batch_size, num_epochs = 32, 60 if not test_run else 5
    lr, momentum, weight_decay = 1e-3, 0.9, 5e-4
    optimizer = optim.SGD(ssd_net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = MultiBoxLoss(priors, num_object_classes, overlap_thresh=0.5, negs_over_pos=3, device=device)
    
    # get dataset
    data_len, tratio = len(dataset), 0.05
    train_dataset, test_dataset = random_split(dataset, [int(np.ceil((1-tratio)*data_len)), int(np.floor(tratio*data_len))])
    train_dl = DataLoader(train_dataset, batch_size, num_workers=8, shuffle=True, collate_fn=detection_data_collate, pin_memory=True)
    test_dl = DataLoader(test_dataset, 256, num_workers=8, shuffle=False, collate_fn=detection_data_collate, pin_memory=True)

    # start training
    summary_dir = os.path.join(exp_dir, "summary")
    pr_curves_dir = os.path.join(exp_dir, "PRcurves")
    os.makedirs(summary_dir)
    os.makedirs(pr_curves_dir)
    sw, step, num_batches = SummaryWriter(summary_dir), 0, len(train_dl)
    viz_update_fred = int(num_batches/10) if not test_run else 2  # atleast around 10 batches visualized in tensorboard per epoch
    progress_manager = enlighten.get_manager()
    epoch_progress = progress_manager.counter(total=num_epochs, desc="Epochs", unit="batches")
    batch_progress = progress_manager.counter(total=num_batches, desc="\tBatches", unit="images", leave=False)
    random_input_onnx = torch.randn((1, 3, image_size, image_size)).to(device)
    for epoch in range(num_epochs):
        for bi, (images, targets) in enumerate(train_dl):
            if test_run and bi > 4:
                break
            ssd_net.train()
            images, targets = images.to(device), [ann.to(device) for ann in targets]
            out = ssd_net(images)  # shape: [batch_size, num_priors, 5]
            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            sw.add_scalar("LocLoss", loss_l.item(), global_step=step)
            sw.add_scalar("ConfLoss", loss_c.item(), global_step=step)
            if bi % viz_update_fred == 0:
                dcat_map = dataset.category_map
                for timages, ttargets in test_dl:
                    eval_results = detection_eval(timages, ttargets, priors, num_object_classes, ssd_net, detection_head, dcat_map, device)
                    if epoch == 0:
                        sw.add_image("NormalizedImages", make_grid(timages), global_step=step)
                        sw.add_image("Images", make_grid(denormalize(timages)), global_step=step)
                        sw.add_image("ImagesWithGT", eval_results.images_with_GT, global_step=step)
                    sw.add_image("ImagesWithPreds", eval_results.images_with_preds, global_step=step)
                    plot_PRcurve(eval_results.recall, eval_results.precision, dcat_map, pr_curves_dir, f"_epoch{epoch}")
                    for mi, v in enumerate(eval_results.mAP):
                        sw.add_scalar(f"mAP_{dcat_map[mi]}", float(v), global_step=step)
                    sw.add_scalar("mAPs", np.array(eval_results.mAP).mean(), global_step=step)
                    break
            step += 1
            batch_progress.update()
        batch_progress.count = 0
        epoch_progress.update()
        # todo(hh): change to saving the best one after comparing metrics
        torch.save(ssd_net.state_dict(), os.path.join(exp_dir, f'epoch{epoch}.pth'))
        torch.onnx.export(ssd_net, random_input_onnx, os.path.join(exp_dir, f'epoch{epoch}.onnx'),
                          export_params=True, opset_version=12, do_constant_folding=False, input_names=['input'],
                          output_names=['output'])
    progress_manager.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Program to train a SSD for a dataset of given size.')
    parser.add_argument("--dataset", default="COCO", help="Dataset to train SSD for, currently only accepts {COCO}.")
    parser.add_argument("--image_size", type=int, default=128, help="Size (width==height in pixels) of the SSD input.")
    parser.add_argument("--dataset_subcategory", default="TRAFFIC", help='''Subset of specified dataset classes to train the SSD for.
                                                       Currently, allowing {TRAFFIC} for COCO dataset.
                                                       Specify more subcategories in data/coco/coco.py for allowing more.''')
    parser.add_argument("--model_path", default="", help="SSD model (.pth) to finetune.")
    parser.add_argument("--test_run", action="store_true", help="Whether to run a minimal (5 epochs with 5 batches) training run for testing end-to-end functionality.")
    args = parser.parse_args()
    train(args.dataset, args.image_size, args.dataset_subcategory, args.model_path, args.test_run)
