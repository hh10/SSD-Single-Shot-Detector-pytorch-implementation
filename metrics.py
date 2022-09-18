import torch
from torchvision.utils import make_grid

import numpy as np
import matplotlib.pyplot as plt
import os
from collections import namedtuple

from data import jaccard, get_colors_and_legend, draw_gt_bounding_boxes, draw_pred_bounding_boxes, denormalize
from box_utils import decode

EvalResults = namedtuple("EvalResults", ["images_with_GT", "images_with_preds", "mAP", "precision", "recall"])

def detection_eval(timages, ttargets, priors, nclasses, ssd_net, detection_head, category_map, device) -> EvalResults:
    timages, ttargets = timages.to(device), [ann.to(device) for ann in ttargets]
    gt_bbox_images = draw_gt_bounding_boxes(denormalize(timages), ttargets, category_map)
    ssd_net.eval()
    with torch.no_grad():
        pred_box_offsets_confs = ssd_net(timages)
        pred_lboxes = detection_head(pred_box_offsets_confs, priors)
        recall, precision, mAP = avg_precision(pred_lboxes, ttargets, nclasses - 1)
    pred_bbox_images = draw_pred_bounding_boxes(denormalize(timages), pred_lboxes, category_map)
    return EvalResults(images_with_GT=make_grid(gt_bbox_images),
                       images_with_preds=make_grid(pred_bbox_images),
                       mAP=mAP, precision=precision, recall=recall)

def voc_ap(rec, prec):
    # append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]
    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def avg_precision(preds, targets, num_classes, ovthresh=0.5):
    """
    Takes pred bboxes and target for an image:
        - preds: [bs, num_classes, top_k, 5] (bboxes in normalized corner pt form)
        - targets: [bs, #objs, 5]
    and computes the average precision. 
    """
    true_pos, false_pos, total_gt = [[] for _ in range(num_classes)], [[] for _ in range(num_classes)], [0]*num_classes
    for (ipreds, itargets) in zip(preds, targets):
        for c, cpreds in enumerate(ipreds):
            # check if the class is present in the image
            ctargets = itargets[itargets[:, -1] == c].cpu()  # filter based on label
            total_gt[c] += ctargets.shape[0]
            # check if preds have this class
            cpreds = cpreds[cpreds.sum(axis=1) != 0]  # nonzero preds after NMS
            num_cdetections = cpreds.shape[0]
            if ctargets.shape[0] == 0:
                true_pos[c].append(0)
                false_pos[c].append(num_cdetections)
                continue
            if num_cdetections == 0:
                continue
            # class present in image and also predicted
            sorted_indices = np.argsort(-cpreds[:, 0])
            cpreds = cpreds[sorted_indices, 1:]
            
            overlaps = jaccard(ctargets[:, :-1], cpreds)  # should be [#objs, #preds]
            overlaps = overlaps.t()
            for overlap in overlaps:
                if overlap[overlap > ovthresh].sum() > 0:
                    true_pos[c].append(1)
                    false_pos[c].append(0)
                else:
                    true_pos[c].append(0)
                    false_pos[c].append(1)

    # compute precision recall
    recall, precision, mAP = [], [], []
    x_intervals = 25
    for tp, fp, gt in zip(true_pos, false_pos, total_gt):
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        if gt == 0:
            prec_std = [prec[int(x)] for x in np.linspace(0, len(prec), x_intervals)]
            precision.append(prec_std); recall.append([0]*x_intervals); mAP.append(0)
            continue
        rec = tp / gt
        ap = voc_ap(rec, prec)
        # standardize the recall and precision here
        rec_std = [rec[int(x)] for x in np.linspace(0, len(rec)-1, x_intervals)]
        prec_std = [prec[int(x)] for x in np.linspace(0, len(prec)-1, x_intervals)]
        precision.append(prec_std); recall.append(rec_std); mAP.append(ap)  # noqa: E702
    return np.array(recall), np.array(precision), mAP

def closest_prior_accuracy(priors, conf_t, num_classes, ssd_out=None):
    # todo(hh): adapt this func for training time statistics (maybe)
    corr_incorr_out_indices = {}
    if ssd_out is not None:
        ssd_out = ssd_out.view(-1, )  # ensure flatten
    for pi, prior in enumerate(priors):
        out_prior_classes_offset = pi * (4 + num_classes) + 4
        tclass = int(conf_t[pi])
        if tclass < 1:  # background class
            continue
        out_tclass_index = out_prior_classes_offset + tclass
        corr_incorr_out_indices[out_tclass_index] = []
        for j in range(num_classes):
            if j != tclass:
                corr_incorr_out_indices[out_tclass_index].append(out_prior_classes_offset + j)
                # check that the specification for correct class is satisfied by the network output
                if (ssd_out is not None and ssd_out[out_prior_classes_offset + j] >= ssd_out[out_tclass_index]):  # incorrect prediction
                    return False, corr_incorr_out_indices
    return True, corr_incorr_out_indices


def plot_PRcurve(recall, precision, category_map, dir, suffix=""):
    fig = plt.figure(figsize=(10, 6))
    fig.set_tight_layout(True)
    colors = get_colors_and_legend('tab10')
    for pi, (rec, prec) in enumerate(zip(recall, precision)):
        plt.plot(rec, prec, color=colors[pi % 10]/255., marker="s", markersize=3)
    plt.legend(list(category_map.values()))
    plt.grid()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(os.path.join(dir, f"PRcurve{suffix}.png"), dpi=300)

def plot_ssd_outputs(gt, priors, conf_t, num_classes, ssd_out, image_t, category_map):
    # todo(hh): adapt this func for training time statistics (maybe)
    image_t, gt = image_t.clone().unsqueeze(0), gt.unsqueeze(0)
    # drawing GT box
    image_t = draw_gt_bounding_boxes(image_t, gt, category_map=category_map, black_line=True)
    for pi, prior in enumerate(priors):
        tclass = int(conf_t[pi])
        if tclass < 1:  # background class
            continue
        # drawing prior and correct class
        prior_box = torch.empty_like(prior)
        # normalized center offset to normalized corner pt form for priors
        prior_box[:2] = prior[:2] - prior[2:]/2
        prior_box[2:] = prior[:2] + prior[2:]/2
        prior_box = torch.hstack((prior_box, torch.tensor(tclass-1))).unsqueeze(0).unsqueeze(0)
        image_t = draw_gt_bounding_boxes(image_t, prior_box, category_map=category_map, corner_radius=12)
        # draw ssd_out coords/box
        pred_boxes = decode(ssd_out[pi][:4].unsqueeze(0), priors, [0.1, 0.2])
        pred_box = torch.hstack((pred_boxes[pi], torch.tensor(tclass - 1))).unsqueeze(0)
        image_t = draw_gt_bounding_boxes(image_t, pred_box.unsqueeze(0), category_map=category_map)
    return image_t
