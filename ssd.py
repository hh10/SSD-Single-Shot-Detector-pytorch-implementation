import torch
import torch.nn as nn
from torchvision import models
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import ResNet18_Weights

from typing import List, Dict
from collections import OrderedDict
import numpy as np

from prior_boxes import SsdFeatureMapParams, PriorBox

ALLOW_DYNAMIC_RESHAPING = False
USE_INTERPOLATE_FUNC = False


class FeaturePyramidNetwork(nn.Module):
    """
    FPN takes the different numbered features of different sizes, makes the number of features same with 1x1
    convolutions and adds them bottom-up (deepest to shallower) after upscaling the deepest map 
    (usually factor of 2 used).
    Args:
        fpn_layer_mapping: must be layers of the backbone -> desired output names for them
    """
    def __init__(self, backbone: nn.Module, fpn_layers_mapping: Dict[str, str], out_channels: int):
        super().__init__()
        self.branches = nn.ModuleDict()
        for intermediate_layer in list(fpn_layers_mapping.keys())[:-1]:
            module = getattr(backbone, intermediate_layer)
            if isinstance(module, nn.MaxPool2d):
                in_channels = backbone.bn1.num_features
            elif isinstance(module, nn.Sequential):
                if isinstance(module[0], models.resnet.BasicBlock):
                    in_channels = module[-1].bn2.num_features
                else:
                    raise NotImplementedError(f'Unhandled module {module[0]}')
            # save the backbone layer with the intended feature map name
            layer_name = backbone.return_layers[intermediate_layer]
            # make the number of channels while keeping the size same
            self.branches[layer_name] = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                bias=True,
                kernel_size=(1, 1),
                stride=1,
            )
        # It is convenient to reverse the feature map IDs such that we start
        # with the deepest return layer (i.e. x16).
        # ALTERNATIVE if using Python <3.8 as dict values reverse not allowed in Python 3.7 and below
        # self.layer_names_reversed = list(fpn_layers_mapping.values())
        # self.layer_names_reversed.reverse()
        self.layer_names_reversed = list(reversed(fpn_layers_mapping.values()))
        self.interpolate_conv = nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2)

    def forward(self, backbone_output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        fpn_output = OrderedDict()
        # start and upsample the "deepest" output (= x16) + it won't be convoluted in FPN.
        name_from = self.layer_names_reversed[0]
        tensor_from = backbone_output[name_from]
        if USE_INTERPOLATE_FUNC:
            upsampled_from = nn.functional.interpolate(tensor_from, scale_factor=2, mode="bilinear", align_corners=False)
        else:
            upsampled_from = self.interpolate_conv(tensor_from)
        # Note: it is assumed that the out_channels is equal to name_from layer channels
        
        # Iterate over the layers that have FPN branches.
        for name_to in self.layer_names_reversed[1:]:
            # Fetch the output and its target FPN branch convolution
            tensor_to = backbone_output[name_to]
            conv2d_to = self.branches[name_to](tensor_to)
            # Add the upsampled tensor and FPN branch tensor to the convolution output
            output = upsampled_from + conv2d_to
            # Register the result in the output dict
            name_output = "{}_and_{}".format(name_from, name_to)
            fpn_output[name_output] = output
            # Upsample the result to add to the next result
            if USE_INTERPOLATE_FUNC:
                upsampled_from = nn.functional.interpolate(output, scale_factor=2, mode="bilinear", align_corners=False)
            else:
                upsampled_from = self.interpolate_conv(output)
            name_from = name_to
        return fpn_output

    def out_channels(self) -> int:
        # all convolutions have the same output channel size
        return list(self.branches.values())[0].out_channels
    
    def featuremap_names(self) -> List[str]:
        return list(self.layer_names_reversed)


class BackboneWithFPN(nn.Module):
    def __init__(self, backbone_base, backbone_extra, bbase_layers_mapping: Dict[str, str], out_channels: int, img_size: int):
        super().__init__()
        self.img_size = img_size
        self.backbone = IntermediateLayerGetter(backbone_base, return_layers=bbase_layers_mapping)
        self.backbone["extra1"] = backbone_extra
        bbase_layers_mapping["extra1"] = f"x{int(img_size/64)}"
        self.fpn = FeaturePyramidNetwork(self.backbone, fpn_layers_mapping=bbase_layers_mapping, out_channels=out_channels)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        backbone_output = self.backbone(x)
        fpn_output = self.fpn(backbone_output)
        return {
            f"x{int(self.img_size/4)}": fpn_output[f"x{int(self.img_size/8)}_and_x{int(self.img_size/4)}"],
            f"x{int(self.img_size/8)}": fpn_output[f"x{int(self.img_size/16)}_and_x{int(self.img_size/8)}"],
            f"x{int(self.img_size/16)}": fpn_output[f"x{int(self.img_size/32)}_and_x{int(self.img_size/16)}"],
            f"x{int(self.img_size/32)}": fpn_output[f"x{int(self.img_size/64)}_and_x{int(self.img_size/32)}"],
            f"x{int(self.img_size/64)}": backbone_output[f"x{int(self.img_size/64)}"],
        }

    def out_channels(self) -> int:
        return self.fpn.out_channels()

    def featuremap_names(self) -> int:
        return self.fpn.featuremap_names()


class SSDHead(nn.Module):
    def __init__(self, backbone_with_fpn, ssd_feature_map_settings, num_classes: int, img_size: int):
        super(SSDHead, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.ssd_feature_map_settings = ssd_feature_map_settings
        
        self.ssd_bbox_layers = nn.ModuleDict()
        # apply convolution + reshape the width and height into a single dimension.
        for map_id in backbone_with_fpn.featuremap_names():
            map_dict = nn.ModuleDict()
            for ratio_id in range(len(self.ssd_feature_map_settings[map_id].ratios) + 1):
                conv_id = "{}_{}".format(map_id, ratio_id)
                conv_op = nn.Conv2d(
                    in_channels=backbone_with_fpn.out_channels(),
                    out_channels=4 + num_classes,
                    kernel_size=(3, 3),
                    padding=1,
                )
                map_dict[conv_id] = conv_op
            self.ssd_bbox_layers[map_id] = map_dict
    
    def forward(self, feature_maps: Dict[str, torch.Tensor]) -> torch.Tensor:
        output: List[torch.Tensor] = []
        for i, (map_id, tensor) in enumerate(feature_maps.items()):
            ssd_bbox_layer = self.ssd_bbox_layers[map_id]
            for layer in ssd_bbox_layer.values():
                # Pass through SSD convolution
                out = layer(tensor)
                # Turn NCHW -> NHWC
                out = out.transpose(1, 2).transpose(2, 3)
                # Reshape to [batch, mapsize ** 2, 4 + num_classes)
                if ALLOW_DYNAMIC_RESHAPING:
                    map_size = self.ssd_feature_map_settings[map_id].size
                    out = out.view(out.shape[0], map_size ** 2, out.shape[-1])
                else:
                    out = out.view(-1, int((self.img_size/(2**(2+i)))**2), self.num_classes + 4)
                output.append(out)
        # Output tensor shape of SSD network: [batch, num_priors, 4 + flags.num_classes]
        return torch.cat(output, 1)


class SSD(nn.Module):
    def __init__(self, backbone_with_fpn, head, num_classes):
        super(SSD, self).__init__()
        self.backbone_with_fpn = backbone_with_fpn  # input: endpoints, output: backbone_feature_maps
        self.head = head  # input: backbone_feature_maps, output: locations and class confs for each prior box

    def forward(self, x, test_phase=False):
        backbone_feature_maps = self.backbone_with_fpn(x)
        return self.head(backbone_feature_maps)


def build_priors(img_size):
    SSD_FEATURE_MAPS_SETTINGS = {
        f"x{int(img_size/4)}": SsdFeatureMapParams(scale=0.01, extra_scale=np.sqrt(0.01 * 0.07), size=int(img_size/4), ratios=[0.75]),
        f"x{int(img_size/8)}": SsdFeatureMapParams(scale=0.07, extra_scale=np.sqrt(0.07 * 0.12), size=int(img_size/8), ratios=[0.75]),
        f"x{int(img_size/16)}": SsdFeatureMapParams(scale=0.12, extra_scale=np.sqrt(0.12 * 0.25), size=int(img_size/16), ratios=[2, 0.75]),
        f"x{int(img_size/32)}": SsdFeatureMapParams(scale=0.25, extra_scale=np.sqrt(0.25 * 0.45), size=int(img_size/32), ratios=[2, 0.75]),
        f"x{int(img_size/64)}": SsdFeatureMapParams(scale=0.45, extra_scale=np.sqrt(0.45 * 0.6), size=int(img_size/64), ratios=[2, 0.75]),
    }
    return PriorBox(img_size, SSD_FEATURE_MAPS_SETTINGS).forward(), SSD_FEATURE_MAPS_SETTINGS


def build_ssd_and_priors(img_size, num_object_classes):
    FEATURE_MAP_ID_MAPPING = {"maxpool": f"x{int(img_size/4)}",
                              "layer2": f"x{int(img_size/8)}",
                              "layer3": f"x{int(img_size/16)}",
                              "layer4": f"x{int(img_size/32)}"}
    priors, SSD_FEATURE_MAPS_SETTINGS = build_priors(img_size)

    out_channels = 16
    backbone_base = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    backbone_base.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    backbone_extra = nn.Sequential(nn.Conv2d(512, out_channels, kernel_size=(3, 3), stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(num_features=out_channels),
                                   nn.ReLU(inplace=True))
    backbone_with_fpn = BackboneWithFPN(backbone_base, backbone_extra, FEATURE_MAP_ID_MAPPING, out_channels, img_size)
    ssd_head = SSDHead(backbone_with_fpn, SSD_FEATURE_MAPS_SETTINGS, num_object_classes, img_size)
    return SSD(backbone_with_fpn, ssd_head, num_object_classes), priors
