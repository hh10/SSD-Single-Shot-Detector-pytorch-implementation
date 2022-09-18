import torch
from box_utils import decode, nms

import torch.nn as nn


class Detector():
    """
    At test time, Detect is the final layer of SSD. Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a 'top_k' number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes: int, top_k: int, conf_thresh: float, nms_thresh: float, bkg_label: int = 0):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        assert nms_thresh > 0, print(f'nms_threshold {nms_thresh} should have been positive.')
        self.conf_thresh = conf_thresh
        self.variance = [0.1, 0.2]  # todo(hh): change to be provided by args
        self.softmax = nn.Softmax(dim=-1)

    def __call__(self, ssd_head_out: torch.Tensor, prior_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ssd_head_out: [batch, num_priors, 4 + self.num_classes]
            prior_data: Prior boxes [1, num_priors, 4]
        Returns:
            Tensor with 'top_k' #scores+boxes in integer pt format for every class and image,
            shape: [bs, num_classes-1, top_k, 5]. Bboxes are in normalized corner pt form.
        """
        batch_size, num_priors = ssd_head_out.shape[0], prior_data.size(0)
        output = torch.zeros(batch_size, self.num_classes-1, self.top_k, 5)
        loc_data, conf_data = ssd_head_out[:, :, :4], self.softmax(ssd_head_out[:, :, 4:])
        conf_preds = conf_data.view(batch_size, num_priors, self.num_classes).transpose(2, 1)
        
        # Decode predictions into bboxes.
        for i in range(batch_size):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)  # [num_priors, 4]
            conf_scores = conf_preds[i].clone()  # [classes, num_priors]
            # For each class, perform NMS
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    # no box with cla conf for cl class good enough
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of the highest scoring and non-overlapping boxes per class
                ids = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl-1, :ids.shape[0]] = torch.cat((scores[ids].unsqueeze(1), boxes[ids]), 1)
        flt = output.contiguous().view(batch_size, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output
