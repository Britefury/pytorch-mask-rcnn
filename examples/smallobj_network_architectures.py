import functools
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init as nninit
import torch.utils.model_zoo as model_zoo
import torchvision.models as models, torchvision.models.vgg as vgg, torchvision.models.resnet as tv_resnet
from maskrcnn.model import mask_rcnn, rcnn, rpn, config


class ArchitectureRegistry (object):
    def __init__(self):
        self._registry = {}

    def register(self, name):
        """
        Decorator to register an architecture;

        Use like so:

        >>> @backbones.register('my_architecture')
        ... def build_backbone(n_classes, gaussian_noise_std):
        ...     # Build network
        ...     return net
        """

        def decorate(fn):
            self._registry[name] = fn
            return fn

        return decorate

    def get_builder(self, name):
        """
        Get network building function and expected sample shape:

        For example:
        >>> fn = get_build_fn_for_architecture('my_architecture')
        """
        return self._registry[name]


backbones = ArchitectureRegistry()
heads = ArchitectureRegistry()




class SmallObjectMaskRCNNConfig (config.Config):
    # Configuration for small object detection networks
    # Suitable for Kaggle Data Science Bowl 2018 (nucleus segmentation)
    # and ellipses dataset.

    BACKBONE_STRIDES = [2, 4, 8, 16]
    RPN_ANCHOR_SCALES = [16, 32, 64, 128]

    NUM_PYRAMID_LEVELS = len(BACKBONE_STRIDES)

    def __init__(self, use_focal_loss=False, mask_size=28, mask_box_enlarge=1.0, mask_box_border_min=0.0,
                 rpn_train_anchors_per_image=256, detection_max_instances=1024,
                 pre_nms_limit=6000, pre_nms_limit_unsup=6000,
                 rpn_nms_threshold=0.7, rpn_nms_threshold_unsup=0.3,
                 post_nms_rois_training=2048, post_nms_rois_training_unsup=512,
                 post_nms_rois_inference=1024, train_rois_per_image=200, detection_min_confidence=0.7,
                 detection_min_confidence_unsup=0.7, detection_nms_threshold=0.3, num_classes=2):
        super(SmallObjectMaskRCNNConfig, self).__init__()

        self.NUM_CLASSES = num_classes


        if use_focal_loss:
            self.RPN_TRAIN_ANCHORS_PER_IMAGE = None
            self.RPN_OBJECTNESS_FUNCTION = 'focal'
        else:
            self.RPN_TRAIN_ANCHORS_PER_IMAGE = rpn_train_anchors_per_image

        self.RPN_PRE_NMS_LIMIT_TRAIN = pre_nms_limit
        self.RPN_PRE_NMS_LIMIT_TRAIN_UNSUP = pre_nms_limit_unsup
        self.RPN_PRE_NMS_LIMIT_TEST = pre_nms_limit

        self.RPN_NMS_THRESHOLD = rpn_nms_threshold
        self.RPN_NMS_THRESHOLD_UNSUP = rpn_nms_threshold_unsup

        self.RPN_POST_NMS_ROIS_TRAINING = post_nms_rois_training
        self.RPN_POST_NMS_ROIS_TRAINING_UNSUP = post_nms_rois_training_unsup
        self.RPN_POST_NMS_ROIS_INFERENCE = post_nms_rois_inference


        self.ROI_CANONICAL_SCALE = 28
        self.ROI_CANONICAL_LEVEL = 1
        self.ROI_MIN_PYRAMID_LEVEL = 1
        self.ROI_MAX_PYRAMID_LEVEL = 4


        self.RCNN_DETECTION_MAX_INSTANCES = detection_max_instances

        self.RCNN_TRAIN_ROIS_PER_IMAGE = train_rois_per_image

        self.RCNN_DETECTION_MIN_CONFIDENCE = detection_min_confidence
        self.RCNN_DETECTION_MIN_CONFIDENCE_UNSUP = detection_min_confidence_unsup

        self.RCNN_DETECTION_NMS_THRESHOLD = detection_nms_threshold

        self.MINI_MASK_SHAPE = (mask_size * 2, mask_size * 2)  # (height, width) of the mini-mask
        self.MASK_POOL_SIZE = mask_size // 2
        self.MASK_SHAPE = [mask_size, mask_size]
        self.MASK_BOX_ENLARGE = mask_box_enlarge
        self.MASK_BOX_BORDER_MIN = mask_box_border_min




def _wn_init(layer):
    nninit.xavier_normal(layer.weight)
    nn.utils.weight_norm(layer, 'weight')


class Encoder (nn.Module):
    def __init__(self, chn_in, chn_out, n_layers, use_wn=False):
        super(Encoder, self).__init__()
        layers = []
        for i in range(n_layers):
            layers.append(nn.Conv2d(chn_in, chn_out, (3, 3), padding=1, bias=use_wn))
            if use_wn:
                _wn_init(layers[-1])
            else:
                layers.append(nn.BatchNorm2d(chn_out))
            layers.append(nn.ReLU())
            chn_in = chn_out

        self.encode = nn.Sequential(*layers)


    def forward(self, x):
        y = self.encode(x)
        y_half = F.max_pool2d(y, kernel_size=2, stride=2)
        return y, y_half



class Decoder (nn.Module):
    def __init__(self, enc_chn_in, dec_chn_in, chn_out, n_layers, use_wn=False):
        super(Decoder, self).__init__()
        layers = []
        chn_in = enc_chn_in + dec_chn_in

        for i in range(n_layers):
            layers.append(nn.Conv2d(chn_in, chn_out, (3, 3), padding=1, bias=use_wn))
            if use_wn:
                _wn_init(layers[-1])
            else:
                layers.append(nn.BatchNorm2d(chn_out))
            layers.append(nn.LeakyReLU(0.1))
            chn_in = chn_out

        self.decode = nn.Sequential(*layers)

        self.chn_out = chn_out


    def forward(self, x_enc, x_dec):
        if x_enc is not None:
            _, _, H, W = x_enc.size()
        else:
            _, _, H, W = x_dec.size()
            H *= 2
            W *= 2
        x_us = F.interpolate(x_dec, (H, W), mode='bilinear')
        if x_enc is not None:
            y = torch.cat([x_enc, x_us], 1)
        else:
            y = x_us
        return self.decode(y)


def _centre_block(chn_in, chn_out, use_wn):
    centre_layers = [
        nn.Conv2d(chn_in, chn_out, kernel_size=(3, 3), bias=use_wn)
    ]
    if use_wn:
        _wn_init(centre_layers[-1])
    else:
        centre_layers.append(nn.BatchNorm2d(chn_out))
    centre_layers.append(nn.ReLU())
    return nn.Sequential(*centre_layers)



@heads.register('mask_rcnn')
class MRCNNNet (mask_rcnn.AbstractMaskRCNNModel):
    def __init__(self, config, BackboneFac):
        self.BackboneFac = BackboneFac
        super(MRCNNNet, self).__init__(config)


    @property
    def BLOCK_SIZE(self):
        return self.fpn.BLOCK_SIZE


    def build_backbone_fpn(self, config, out_channels):
        return self.BackboneFac()


    def pretrained_parameters(self):
        return []

    def new_parameters(self):
        return self.parameters()


@heads.register('faster_rcnn')
class FasterRCNNNet (rcnn.AbstractFasterRCNNModel):
    def __init__(self, config, BackboneFac):
        self.BackboneFac = BackboneFac
        super(FasterRCNNNet, self).__init__(config)


    @property
    def BLOCK_SIZE(self):
        return self.fpn.BLOCK_SIZE


    def build_backbone_fpn(self, config, out_channels):
        return self.BackboneFac()


    def pretrained_parameters(self):
        return []

    def new_parameters(self):
        return self.parameters()


@heads.register('rpn')
class RPNNet (rpn.AbstractRPNNModel):
    def __init__(self, config, BackboneFac):
        self.BackboneFac = BackboneFac
        super(RPNNet, self).__init__(config)


    @property
    def BLOCK_SIZE(self):
        return self.fpn.BLOCK_SIZE


    def build_backbone_fpn(self, config, out_channels):
        return self.BackboneFac()


    def pretrained_parameters(self):
        return []

    def new_parameters(self):
        return self.parameters()



class UNet5Backbone (nn.Module):
    BLOCK_SIZE = (32, 32)

    def __init__(self, use_wn=True, channels=None, **kwargs):
        super(UNet5Backbone, self).__init__()

        if channels is None:
            channels = [48, 96, 192, 256, 512]
        self.channels = channels

        self.drop = nn.Dropout2d(0.5)

        self.enc1 = Encoder(3, channels[0], 2, use_wn=use_wn)
        self.enc2 = Encoder(channels[0], channels[1], 2, use_wn=use_wn)
        self.enc3 = Encoder(channels[1], channels[2], 3, use_wn=use_wn)
        self.enc4 = Encoder(channels[2], channels[3], 3, use_wn=use_wn)
        self.enc5 = Encoder(channels[3], channels[4], 3, use_wn=use_wn)

        self.centre = _centre_block(channels[4], channels[4], use_wn=use_wn)

        self.dec5 = Decoder(channels[4], channels[4], channels[4], 2, use_wn=use_wn)
        self.dec4 = Decoder(channels[4], channels[3], channels[3], 2, use_wn=use_wn)
        self.dec3 = Decoder(channels[3], channels[2], channels[2], 2, use_wn=use_wn)
        self.dec2 = Decoder(channels[2], channels[1], channels[1], 2, use_wn=use_wn)

        self.fpn5 = nn.Conv2d(channels[4], 256, (1, 1))
        self.fpn4 = nn.Conv2d(channels[3], 256, (1, 1))
        self.fpn3 = nn.Conv2d(channels[2], 256, (1, 1))
        self.fpn2 = nn.Conv2d(channels[1], 256, (1, 1))


    def forward(self, x):
        e1, x = self.enc1(x)
        e2, x = self.enc2(x)
        e4, x = self.enc3(x)
        e8, x = self.enc4(x)
        e16, x = self.enc5(x)

        x = self.centre(x)

        d16 = x = self.dec5(e16, x)
        d8 = x = self.dec4(e8, x)
        d4 = x = self.dec3(e4, x)
        d2 = x = self.dec2(e2, x)

        d16 = self.drop(d16)
        d8 = self.drop(d8)
        d4 = self.drop(d4)
        d2 = self.drop(d2)
        # d1 = self.drop(d1)

        p16_out = self.fpn5(d16)
        p8_out = self.fpn4(d8)
        p4_out = self.fpn3(d4)
        p2_out = self.fpn2(d2)
        # p1_out = self.fpn1(d1)

        rpn_feature_maps = [p2_out, p4_out, p8_out, p16_out]
        mrcnn_feature_maps = [p2_out, p4_out, p8_out, p16_out]

        return rpn_feature_maps, mrcnn_feature_maps


@backbones.register('unet5_bn')
def unet5_bn(*args, **kwargs):
    return UNet5Backbone(*args, use_wn=False, **kwargs)

@backbones.register('unet5_wn')
def unet5_wn(*args, **kwargs):
    return UNet5Backbone(*args, use_wn=True, **kwargs)


#
# ResNet
#


class AbstractResidualBlock (nn.Module):
    def __init__(self, chn_in, chn, chn_out, stride=1):
        super(AbstractResidualBlock, self).__init__()

        if stride != 1 or chn_in != chn_out:
            layers = [nn.Conv2d(chn_in, chn_out, kernel_size=1, stride=stride, bias=False)]
            layers.append(nn.BatchNorm2d(chn_out))
            self.downsample = nn.Sequential(*layers)
        else:
            self.downsample = None


class BasicBlock(AbstractResidualBlock):
    def __init__(self, chn_in, chn, chn_out, stride=1):
        super(BasicBlock, self).__init__(chn_in, chn, chn_out, stride=stride)
        self.conv1 = nn.Conv2d(chn_in, chn_out, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(chn_out)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(chn_out, chn_out, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(chn_out)
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(AbstractResidualBlock):
    def __init__(self, chn_in, chn, chn_out, stride=1):
        super(Bottleneck, self).__init__(chn_in, chn, chn*4, stride=stride)
        self.conv1 = nn.Conv2d(chn_in, chn, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(chn)

        self.conv2 = nn.Conv2d(chn, chn, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(chn)

        self.conv3 = nn.Conv2d(chn, chn_out, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(chn_out)

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNetFeatures(nn.Module):
    BLOCK_SIZE = (32, 32)

    def __init__(self, block, layers, base_channels=64):
        super(ResNetFeatures, self).__init__()
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=7, stride=2,
                               padding=3, bias=False)
        chn = base_channels
        self.base_channels = base_channels
        self.bn1 = nn.BatchNorm2d(chn)
        self.relu = nn.ReLU(inplace=True)
        self.channels_out = [0] * len(layers)
        self.layer1, self.channels_out[0] = self._make_layer(block, chn, chn, chn*4, layers[0])
        self.layer2, self.channels_out[1] = self._make_layer(block, chn*4, chn*2, chn*8, layers[1], stride=2)
        self.layer3, self.channels_out[2] = self._make_layer(block, chn*8, chn*4, chn*16, layers[2], stride=2)
        self.layer4, self.channels_out[3] = self._make_layer(block, chn*16, chn*8, chn*32, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, chn_in, chn, chn_out, n_blocks, stride=1):
        layers = []
        layers.append(block(chn_in, chn, chn_out, stride))
        for i in range(1, n_blocks):
            layers.append(block(chn_out, chn, chn_out))

        return nn.Sequential(*layers), chn_out

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        r2 = x = self.layer1(x)
        r4 = x = self.layer2(x)
        r8 = x = self.layer3(x)
        r16 = x = self.layer4(x)

        return [r16, r8, r4, r2]





class ResNetBackboneFPN(nn.Module):
    """
    Feature Pyramid Network (FPN)
    """
    def __init__(self, resnet_main, out_channels):
        super(ResNetBackboneFPN, self).__init__()
        self.resnet_main = resnet_main
        self.out_channels = out_channels
        # self.P6 = nn.MaxPool2d(kernel_size=1, stride=2)
        self.P4_conv1 = nn.Conv2d(self.resnet_main.channels_out[3], self.out_channels, kernel_size=1, stride=1)
        self.P4_conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.P3_conv1 =  nn.Conv2d(self.resnet_main.channels_out[2], self.out_channels, kernel_size=1, stride=1)
        self.P3_conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.P2_conv1 = nn.Conv2d(self.resnet_main.channels_out[1], self.out_channels, kernel_size=1, stride=1)
        self.P2_conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.P1_conv1 = nn.Conv2d(self.resnet_main.channels_out[0], self.out_channels, kernel_size=1, stride=1)
        self.P1_conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

    @property
    def BLOCK_SIZE(self):
        return self.resnet_main.BLOCK_SIZE


    def forward(self, x):
        r16, r8, r4, r2 = self.resnet_main(x)
        p16_out = self.P4_conv1(r16)
        p8_out = self.P3_conv1(r8) + F.interpolate(p16_out, scale_factor=2)
        p4_out = self.P2_conv1(r4) + F.interpolate(p8_out, scale_factor=2)
        p2_out = self.P1_conv1(r2) + F.interpolate(p4_out, scale_factor=2)

        p16_out = self.P4_conv2(p16_out)
        p8_out = self.P3_conv2(p8_out)
        p4_out = self.P2_conv2(p4_out)
        p2_out = self.P1_conv2(p2_out)

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [p2_out, p4_out, p8_out, p16_out]
        mrcnn_feature_maps = [p2_out, p4_out, p8_out, p16_out]

        return rpn_feature_maps, mrcnn_feature_maps


@backbones.register('resnet50')
def resnet50():
    features_net = ResNetFeatures(Bottleneck, [3, 4, 6, 3], 64)
    return ResNetBackboneFPN(features_net, 256)




def build_network(backbone_arch_name, head_arch_name, config_params=None, **kwargs):
    backbone_fac = backbones.get_builder(backbone_arch_name)
    head_fac = heads.get_builder(head_arch_name)

    if config_params is None:
        config_params = {}
    config = SmallObjectMaskRCNNConfig(**config_params)
    make_backbone = lambda: backbone_fac(**kwargs)
    return head_fac(config, make_backbone)



def robust_binary_crossentropy(pred, tgt):
    inv_tgt = -tgt + 1.0
    inv_pred = -pred + 1.0 + 1e-6
    return -(tgt * torch.log(pred + 1.0e-6) + inv_tgt * torch.log(inv_pred))

def sqr_diff(p, q):
    d = p - q
    return d * d

