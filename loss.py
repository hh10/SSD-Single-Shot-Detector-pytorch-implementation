import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from box_utils import match_prior_gt


class MultiBoxLoss(nn.Module):
    """
    SSD Weighted Loss Function
    Compute Targets:
        1) Produce confidence target indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, priors: torch.Tensor, num_classes: int, overlap_thresh: float,
                 negs_over_pos: int, device: torch.device):
        super(MultiBoxLoss, self).__init__()
        self.device = device
        self.priors = priors  # prior boxes are in center-offset form (cx, cy, w, h)
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.negpos_ratio = negs_over_pos
        self.variance = [0.1, 0.2]  # todo(hh): change setting of variance from arg

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Multibox Loss
        Args:
            predictions: shape [batch_size, num_priors, bbox_offset_preds + num_classes].
            targets: GT boxes and labels for a batch, shape: [batch_size, num_objs, 5] (last idx is the label).
                     target boxes are in (x, y, w, h)
        """
        loc_pred, conf_pred = predictions[:, :, :4], predictions[:, :, 4:]  # [batch_size, num_boxes, ...]
        batch_size, num_priors = loc_pred.shape[0], self.priors.shape[0]
        
        # for each image, match gt boxes to priors (default boxes) 
        loc_t = torch.Tensor(batch_size, num_priors, 4)  # bbox offsets for each prior
        conf_t = torch.LongTensor(batch_size, num_priors)  # best target label for each prior
        for bi in range(batch_size):
            img_truths = targets[bi][:, :-1].data
            labels = targets[bi][:, -1].data
            loc_t[bi], conf_t[bi] = match_prior_gt(img_truths, self.priors, self.threshold, self.variance, labels)

        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False).to(self.device)
        conf_t = Variable(conf_t, requires_grad=False).to(self.device)

        pos = conf_t > 0  # [batch_size, num_priors]
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1), shape: [batch, num_priors, 4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_pred)
        loc_p = loc_pred[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_pred.view(-1, self.num_classes)  # [batch_size*num_priors, num_c]
        batch_conf_max = torch.max(batch_conf)
        # below punishes high conf for wrong classes. Loss_c shape: [batch_size*num_priors]
        loss_c = torch.logsumexp(batch_conf - batch_conf_max, dim=1, keepdim=True) + batch_conf_max - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=num_priors-1)
        loss_c[pos.view(-1, 1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(batch_size, -1)  # [batch_size, num_priors]
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        targets_weighted = conf_t[(pos+neg).gt(0)]
        pos_idx = pos.unsqueeze(2).expand_as(conf_pred)
        neg_idx = neg.unsqueeze(2).expand_as(conf_pred)
        conf_p = conf_pred[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        batch_pos = num_pos.sum()
        return loss_l/batch_pos, loss_c/batch_pos
