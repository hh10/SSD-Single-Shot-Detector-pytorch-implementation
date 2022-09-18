import torch

from data import jaccard

from typing import List


# main functions
def match_prior_gt(truths: torch.Tensor, priors: torch.Tensor, threshold: float, variances: List[float], labels: torch.Tensor):
    """
    Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        truths: GT boxes per image, shape: [num_obj, 4].
        priors: prior boxes, shape: [#priors, 4].
        threshold: overlap threshold used when mathing boxes.
        variances: variances corresponding to each prior coord, list [v_w, v_h].
        labels: class labels for objects in image, shape: [num_obj].
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
        1) [num_priors, 4] encoded offsets to learn
        2) [num_priors] top class label for each prior
    """
    # overlaps has with rows corresponding to truths and columns to priors
    overlaps = jaccard(truths, center_to_point_form(priors))
    # (Bipartite Matching)
    # [#objects, 1] best prior for each GT (take max along dim1)
    _, best_prior_idx = overlaps.max(1, keepdim=True)
    best_prior_idx.squeeze_(1)  # shape: [#objects]
    # [1, #priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0); best_truth_overlap.squeeze_(0)  # shape: [#priors]  # noqa: E702
    
    # ensures that those GTs that have a best prior are kept in output, else they may be removed
    # based on threshold check below.
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    # assign the best GT to each prior
    for j, bpi in enumerate(best_prior_idx):
        best_truth_idx[bpi] = j

    conf = labels[best_truth_idx] + 1         # shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background
    matched_gt_boxes = truths[best_truth_idx]  # shape: [num_priors, 4]
    loc_est = encode(matched_gt_boxes, priors, variances)
    assert loc_est.shape == priors.shape and conf.shape == priors.shape[:1], print(priors.shape, loc_est.shape, conf.shape)
    return loc_est, conf


def nms(boxes: torch.Tensor, scores: torch.Tensor, overlap: float = 0.5, top_k: int = 200) -> torch.Tensor:
    """
    Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: location preds for an image, shape: [<num_priors, 4].
        scores: class predscores for the image, shape: [<num_priors].
        overlap: overlap thresh for suppressing unnecessary boxes.
        top_k: maximum number of box preds to consider.
    Return:
        Indices of the kept boxes with respect to num_priors, shape: [reduced_#boxes].
    """

    keep = scores.new(scores.shape[0]).zero_().long()
    if boxes.numel() == 0:
        return keep
    v, idx = scores.sort(0)  # sort in ascending order
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1, yy1, xx2, yy2 = boxes.new(), boxes.new(), boxes.new(), boxes.new()
    w, h = boxes.new(), boxes.new()
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)

    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]
        # load bboxes of next highest vals
        xx1 = torch.index_select(x1, 0, idx)
        yy1 = torch.index_select(y1, 0, idx)
        xx2 = torch.index_select(x2, 0, idx)
        yy2 = torch.index_select(y2, 0, idx)
        # compute intersection of the current largest with all others
        # - element-wise max with next highest scores
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w, h = xx2 - xx1, yy2 - yy1
        # Note(hh): not all people clip
        w, h = torch.clamp(w, min=0.0), torch.clamp(h, min=0.0)
        inter = w*h
        # compute IoU = i / (area(a) + area(b) - i)
        # - load remaining areas)
        rem_areas = torch.index_select(area, 0, idx)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union
        # only keep elements with an IoU <= overlap as more overlap means
        # prediction for the same object.
        idx = idx[IoU.le(overlap)]
    return keep[:count]


# utils
def encode(matched: torch.Tensor, priors: torch.Tensor, variances: List[float]) -> torch.Tensor:
    """
    Encode the variances from the priorbox layers into GT boxes
    matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: GT coords for each prior in normalized pt-form, shape: [num_priors, 4].
        priors: Boxes in center-offset form, shape: [num_priors, 4].
        variances: variances of priorboxes
    Return:
        encoded boxes, shape: [num_priors, 4].
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors, 4]


def decode(loc: torch.Tensor, priors: torch.Tensor, variances: List[float]) -> torch.Tensor:
    """
    Decode locations from predictions using priors to undo
    the offset regression at train time.
    Args:
        loc: location predictions, shape: [num_priors, 4].
        priors: prior boxes in normalized center-offset form, shape: [num_priors, 4].
        variances: variances of priorboxes
    Return:
        decoded bounding box predictions in normalized corner pt form, shape: [num_priors, 4].
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def center_to_point_form(boxes: torch.Tensor) -> torch.Tensor:
    """
    Args:
        boxes: center-size boxes.
    Returns:
        boxes: converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax
