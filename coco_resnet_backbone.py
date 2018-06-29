import numpy as np
import torch.nn as nn, torch.nn.functional as F
from maskrcnn.model.utils import SamePad2d



############################################################
#  Resnet Graph
############################################################

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, config, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=config.COMPAT_MATTERPORT)
        self.bn1 = nn.BatchNorm2d(planes, eps=config.BN_EPS, momentum=0.01)
        if config.TORCH_PADDING:
            self.padding2 = None
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, bias=config.COMPAT_MATTERPORT, padding=1)
        else:
            self.padding2 = SamePad2d(kernel_size=3, stride=1)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, bias=config.COMPAT_MATTERPORT)
        self.bn2 = nn.BatchNorm2d(planes, eps=config.BN_EPS, momentum=0.01)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=config.COMPAT_MATTERPORT)
        self.bn3 = nn.BatchNorm2d(planes * 4, eps=config.BN_EPS, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.padding2 is not None:
            out = self.padding2(out)
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

class ResNet(nn.Module):

    def __init__(self, config, architecture, stage5=False):
        super(ResNet, self).__init__()
        assert architecture in ["resnet50", "resnet101"]
        self.inplanes = 64
        self.layers = [3, 4, {"resnet50": 6, "resnet101": 23}[architecture], 3]
        self.block = Bottleneck
        self.stage5 = stage5

        if config.TORCH_PADDING:
            self.C1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=config.COMPAT_MATTERPORT),
                nn.BatchNorm2d(64, eps=config.BN_EPS, momentum=0.01),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        else:
            self.C1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=config.COMPAT_MATTERPORT),
                nn.BatchNorm2d(64, eps=config.BN_EPS, momentum=0.01),
                nn.ReLU(inplace=True),
                SamePad2d(kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
        self.C2 = self.make_layer(config, self.block, 64, self.layers[0])
        self.C3 = self.make_layer(config, self.block, 128, self.layers[1], stride=2)
        self.C4 = self.make_layer(config, self.block, 256, self.layers[2], stride=2)
        if self.stage5:
            self.C5 = self.make_layer(config, self.block, 512, self.layers[3], stride=2)
        else:
            self.C5 = None

    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        x = self.C3(x)
        x = self.C4(x)
        x = self.C5(x)
        return x


    def stages(self):
        return [self.C1, self.C2, self.C3, self.C4, self.C5]

    def make_layer(self, config, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=config.COMPAT_MATTERPORT),
                nn.BatchNorm2d(planes * block.expansion, eps=config.BN_EPS, momentum=0.01),
            )

        layers = []
        layers.append(block(config, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(config, self.inplanes, planes))

        return nn.Sequential(*layers)


############################################################
#  FPN Graph
############################################################

class CocoResNetBackboneFPN(nn.Module):
    """
    Feature Pyramid Network (FPN)
    """
    def __init__(self, config, C1, C2, C3, C4, C5, out_channels):
        super(CocoResNetBackboneFPN, self).__init__()
        self.out_channels = out_channels
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4
        self.C5 = C5
        self.P6 = nn.MaxPool2d(kernel_size=1, stride=2)
        self.P5_conv1 = nn.Conv2d(2048, self.out_channels, kernel_size=1, stride=1)
        if config.TORCH_PADDING:
            self.P5_conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.P5_conv2 = nn.Sequential(
                SamePad2d(kernel_size=3, stride=1),
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
            )

        self.P4_conv1 =  nn.Conv2d(1024, self.out_channels, kernel_size=1, stride=1)
        if config.TORCH_PADDING:
            self.P4_conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.P4_conv2 = nn.Sequential(
                SamePad2d(kernel_size=3, stride=1),
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
            )
        self.P3_conv1 = nn.Conv2d(512, self.out_channels, kernel_size=1, stride=1)
        if config.TORCH_PADDING:
            self.P3_conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.P3_conv2 = nn.Sequential(
                SamePad2d(kernel_size=3, stride=1),
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
            )
        self.P2_conv1 = nn.Conv2d(256, self.out_channels, kernel_size=1, stride=1)
        if config.TORCH_PADDING:
            self.P2_conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.P2_conv2 = nn.Sequential(
                SamePad2d(kernel_size=3, stride=1),
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
            )

    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        c2_out = x
        x = self.C3(x)
        c3_out = x
        x = self.C4(x)
        c4_out = x
        x = self.C5(x)
        p5_out = self.P5_conv1(x)
        p4_out = self.P4_conv1(c4_out) + F.upsample(p5_out, scale_factor=2)
        p3_out = self.P3_conv1(c3_out) + F.upsample(p4_out, scale_factor=2)
        p2_out = self.P2_conv1(c2_out) + F.upsample(p3_out, scale_factor=2)

        p5_out = self.P5_conv2(p5_out)
        p4_out = self.P4_conv2(p4_out)
        p3_out = self.P3_conv2(p3_out)
        p2_out = self.P2_conv2(p2_out)

        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        p6_out = self.P6(p5_out)

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [p2_out, p3_out, p4_out, p5_out, p6_out]
        mrcnn_feature_maps = [p2_out, p3_out, p4_out, p5_out]

        return rpn_feature_maps, mrcnn_feature_maps


    def detectron_weight_mapping(self):
        det_map = {}
        orphans = []

        norm_suffix = '_bn'

        def res_block(dst_name, src_name):
            module_ref = getattr(self, dst_name)
            num_blocks = len(module_ref)
            for block_i in range(num_blocks):
                detectron_prefix = '{}_{}'.format(src_name, block_i)
                my_prefix = '{}.{}'.format(dst_name, block_i)

                # residual branch (if downsample is not None)
                if module_ref[block_i].downsample is not None:
                    dtt_bp = detectron_prefix + '_branch1'  # short for "detectron_branch_prefix"
                    det_map[my_prefix + '.downsample.0.weight'] = dtt_bp + '_w'
                    orphans.append(dtt_bp + '_b')
                    det_map[my_prefix + '.downsample.1.weight'] = dtt_bp + norm_suffix + '_s'
                    det_map[my_prefix + '.downsample.1.bias'] = dtt_bp + norm_suffix + '_b'
                    det_map[my_prefix + '.downsample.1.running_mean'] = lambda shape, src_blobs: np.zeros(shape)
                    det_map[my_prefix + '.downsample.1.running_var'] = lambda shape, src_blobs: np.ones(shape)

                # conv branch
                for i, c in zip([1, 2, 3], ['a', 'b', 'c']):
                    dtt_bp = detectron_prefix + '_branch2' + c
                    det_map[my_prefix + '.conv%d.weight' % i] = dtt_bp + '_w'
                    orphans.append(dtt_bp + '_b')
                    det_map[my_prefix + '.' + norm_suffix[1:] + '%d.weight' % i] = dtt_bp + norm_suffix + '_s'
                    det_map[my_prefix + '.' + norm_suffix[1:] + '%d.bias' % i] = dtt_bp + norm_suffix + '_b'
                    det_map[my_prefix + '.' + norm_suffix[1:] + '%d.running_mean' % i] = lambda shape, src_blobs: np.zeros(shape)
                    det_map[my_prefix + '.' + norm_suffix[1:] + '%d.running_var' % i] = lambda shape, src_blobs: np.ones(shape)

        def fpn_block(dst_name, src_name, lat):
            if lat:
                det_map['{}_conv1.weight'.format(dst_name)] = 'fpn_inner_{}_sum_lateral_w'.format(src_name)
                det_map['{}_conv1.bias'.format(dst_name)] = 'fpn_inner_{}_sum_lateral_b'.format(src_name)
            else:
                det_map['{}_conv1.weight'.format(dst_name)] = 'fpn_inner_{}_sum_w'.format(src_name)
                det_map['{}_conv1.bias'.format(dst_name)] = 'fpn_inner_{}_sum_b'.format(src_name)
            det_map['{}_conv2.weight'.format(dst_name)] = 'fpn_{}_sum_w'.format(src_name)
            det_map['{}_conv2.bias'.format(dst_name)] = 'fpn_{}_sum_b'.format(src_name)

        det_map['C1.0.weight'] = 'conv1_w'
        det_map['C1.1.weight'] = 'res_conv1_bn_s'
        det_map['C1.1.bias'] = 'res_conv1_bn_b'
        det_map['C1.1.running_mean'] = lambda shape, src_blobs: np.zeros(shape)
        det_map['C1.1.running_var'] = lambda shape, src_blobs: np.ones(shape)

        res_block('C2', 'res2')
        res_block('C3', 'res3')
        res_block('C4', 'res4')
        res_block('C5', 'res5')

        fpn_block('P5', 'res5_2', False)
        fpn_block('P4', 'res4_5', True)
        fpn_block('P3', 'res3_3', True)
        fpn_block('P2', 'res2_2', True)

        return det_map, orphans
