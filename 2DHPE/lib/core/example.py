import torch 
import torch.nn as nn

from typing import List, Sequence, Optional
from einops import rearrange
from functools import partial

import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class CA(nn.Module):
    def __init__(self, inp, reduction = 16):
        super(CA, self).__init__()
        # h:height(行)   w:width(列)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # (b,c,h,w)-->(b,c,h,1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # (b,c,h,w)-->(b,c,1,w)
 
 
         # mip = max(8, inp // reduction)  论文作者所用
        mip =  inp // reduction  # 博主所用   reduction = int(math.sqrt(inp))
 
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace = True)
 
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
 
    def forward(self, x):
        identity = x
 
        n, c, h, w = x.size()
        x_h = self.pool_h(x)  # (b,c,h,1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (b,c,w,1)
 
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
 
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
 
        out = identity * a_w * a_h
 
        return out

        
class Stem(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size = 3,
        stride = 1,
        groups = 1,
        dilation = 1,      
        norm_layer = nn.BatchNorm2d,
        activation = 'RE',
    ):
        super(Stem, self).__init__()
        self.activation = activation
        padding = (kernel_size - 1)//2 * dilation
    
        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, kernel_size, 
            stride = stride, padding = padding, dilation = dilation, 
            groups = groups, bias = False)

        self.norm_layer = norm_layer(out_channels, eps=0.01, momentum=0.01)
        if activation == 'PRE':
            self.acti_layer = nn.PReLU()
        elif activation == 'HS':
            self.acti_layer = nn.Hardswish(inplace = True)  
        else:
            self.acti_layer = nn.ReLU(inplace = True)
    
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.norm_layer(x)
        if self.activation is not None:
            x = self.acti_layer(x)
        return x


class BlockConfig:
    def __init__(
        self,
        in_channels,
        exp_ratio,
        out_channels,
        kernel_size,
        stride,
        dilation,
        activation,
        use_se,
    ):
        self.in_channels = in_channels
        self.exp_channels = exp_ratio * in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.activation = activation
        self.use_se = use_se
        
class Block(nn.Module):
    def __init__(
        self,
        cnf: BlockConfig,
        ):
        super(Block, self).__init__()
        self.use_res_connect = cnf.stride == 1 and cnf.in_channels == cnf.out_channels

        layers = []
        # expand
        if cnf.exp_channels != cnf.in_channels:
            layers.append(
                Stem(
                    cnf.in_channels, 
                    cnf.exp_channels, 
                    kernel_size=1,
                    activation=cnf.activation
                )
            )

        # depthwise
        layers.append(
            Stem(
                cnf.exp_channels,
                cnf.exp_channels,
                kernel_size=cnf.kernel_size,
                stride=cnf.stride,
                groups=cnf.exp_channels,
                dilation=cnf.dilation,
                activation=cnf.activation,
            )
        )

        if cnf.use_se == True:
            layers.append(
                CA(cnf.exp_channels)
            )
        
        layers.append(
            Stem(
                cnf.exp_channels,
                cnf.out_channels,
                kernel_size=1,
                activation=None))

        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        result = self.block(x)
        if self.use_res_connect == True:
            result = result + x
        return result

class TBlockConfig:
    def __init__(
        self,
        in_channels,
        H_channels,
        L_channels,
        exp_ratio,
        out_channels,
        kernel_size,
        dilation,
        activation,
        out_type,
    ):
        self.in_channels = in_channels
        self.H_channels = H_channels
        self.L_channels = L_channels
        self.exp_ratio = exp_ratio 
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.activation = activation
        self.out_type = out_type
class TBlock(nn.Module):
    def __init__(
        self,
        size,
        cnf: TBlockConfig):
        super().__init__()
        self.out_type = cnf.out_type
        self.use_res_connect = cnf.in_channels == cnf.out_channels and cnf.out_type == 'H'
        HLayers = []
        LLayers = []
        if cnf.exp_ratio != 1:
            exp_channels = cnf.exp_ratio * cnf.in_channels
            HLayers.append(
                Stem(
                    cnf.in_channels,
                    exp_channels,
                    1,
                    activation=cnf.activation
                )
            )
            LLayers.append(
                Stem(
                    cnf.in_channels,
                    exp_channels,
                    1,
                    activation=cnf.activation
                )
            )
        else:
            exp_channels = cnf.in_channels
        
        HLayers.append(nn.Sequential(
            Stem(
                exp_channels,
                exp_channels,
                kernel_size=cnf.kernel_size,
                stride=1,
                groups=exp_channels,
                activation=cnf.activation
            ),
            CA(exp_channel),
            Stem(
                exp_channels,
                exp_channels,
                kernel_size=cnf.kernel_size,
                stride=1,
                groups=exp_channels,
                activation=cnf.activation
            ),
            Stem(
                exp_channels,
                cnf.H_channels,
                kernel_size=1,
                activation=None
            ))
        )
        LLayers.append(nn.Sequential(
            Stem(
                exp_channels,
                exp_channels,
                kernel_size=cnf.kernel_size,
                stride=2,
                groups=exp_channels,
                activation=cnf.activation
            ),
            CA(exp_channel),
            Stem(
                exp_channels,
                exp_channels,
                kernel_size=cnf.kernel_size,
                stride=1,
                groups=exp_channels,
                activation=cnf.activation
            ),
            Stem(
                exp_channels,
                cnf.L_channels,
                kernel_size=1,
                activation=None
            ))
        )
        

        if cnf.exp_ratio != 1:
            H_exp_channels = cnf.exp_ratio * cnf.H_channels
            L_exp_channels = cnf.exp_ratio * cnf.L_channels
        
            HLayers.append(
                Stem(
                    cnf.H_channels,
                    H_exp_channels,
                    kernel_size=1,
                    activation=cnf.activation
                )
            )
            LLayers.append(
                Stem(
                    cnf.L_channels,
                    L_exp_channels,
                    kernel_size=1,
                    activation=cnf.activation
                )
            )
        else:
            H_exp_channels = cnf.H_channels
            L_exp_channels = cnf.L_channels

        if cnf.out_type == 'H':
            HLayers.append(nn.Sequential(
                Stem(H_exp_channels,
                    H_exp_channels,
                    kernel_size=cnf.kernel_size,
                    stride=1,
                    groups=H_exp_channels,
                    activation=cnf.activation),
                    CA(H_exp_channels),
                    Stem(H_exp_channels,
                    H_exp_channels,
                    kernel_size=cnf.kernel_size,
                    stride=1,
                    groups=H_exp_channels,
                    activation=cnf.activation),
                    Stem(H_exp_channels,
                    cnf.out_channels,
                    kernel_size=1,
                    activation=None))
            )
            LLayers.append(nn.Sequential(
                Stem(L_exp_channels,
                L_exp_channels,
                kernel_size=cnf.kernel_size,
                groups=L_exp_channels,
                activation=cnf.activation),

                CA(L_exp_channels),
                
                Stem(L_exp_channels,
                    L_exp_channels,
                    kernel_size=cnf.kernel_size,
                    stride=1,
                    groups=L_exp_channels,
                    activation=cnf.activation),
                
                Stem(L_exp_channels,
                cnf.out_channels,
                kernel_size=1,
                activation=None),
                ))

        elif cnf.out_type == 'L':
            HLayers.append(nn.Sequential(
                Stem(H_exp_channels,
                    H_exp_channels,
                    kernel_size=cnf.kernel_size,
                    stride=2,
                    groups=H_exp_channels,
                    activation=cnf.activation),

                    CA(H_exp_channels),

                    Stem(H_exp_channels,
                    H_exp_channels,
                    kernel_size=cnf.kernel_size,
                    stride=1,
                    groups=H_exp_channels,
                    activation=cnf.activation),

                    Stem(H_exp_channels,
                    cnf.out_channels,
                    kernel_size=1,
                    activation=None))
                )
            LLayers.append(nn.Sequential(
                Stem(L_exp_channels,
                L_exp_channels,
                kernel_size=cnf.kernel_size,
                groups=L_exp_channels,
                activation=cnf.activation),

                CA(L_exp_channels),

                Stem(L_exp_channels,
                    L_exp_channels,
                    kernel_size=cnf.kernel_size,
                    stride=1,
                    groups=L_exp_channels,
                    activation=cnf.activation),

                Stem(L_exp_channels,
                cnf.out_channels,
                kernel_size=1,
                activation=None)))
        self.uppool = nn.UpsamplingBilinear2d(size = size)
        self.H_line = nn.Sequential(*HLayers)
        self.L_line = nn.Sequential(*LLayers)
        self.fusion = nn.Hardswish(inplace = True)
    def forward(self, x):
        x_H = x[0]
        x_L = x[1]
        x_H_ = self.H_line(x_H)
        x_L = self.L_line(x_L)
        if self.out_type == 'H':
            x_L_ = self.uppool(x_L)
        else:
            x_L_ = x_L
        # print(x_H.shape)
        # print(x_L.shape)
        # print(x.shape)
        if self.use_res_connect:
            x_H = self.fusion(x_H_ + x_L_ + x_H)
        else:
            x_H = self.fusion(x_H_ + x_L_)
        if self.out_type == 'H':
            x[0] = x_H
            x[1] = x_L
        else:
            x[0] = x[1] = x_H
        return x

class MobilePosNet(nn.Module):
    def __init__(
        self,
        BlockSetting: List[BlockConfig],
        TBlockSetting: List[TBlockConfig],
        num_joints,
        unit_size,
        pmap_size,
        dropout = 0.2,
    ):
        super(MobilePosNet, self).__init__()
        
        self.num_joints = num_joints
        self.dropout = dropout
        self.pad = (unit_size[0] // 2, unit_size[1] // 2)
        self.padded_patch = (pmap_size[1] + 1, pmap_size[0] + 1)
   
        layers = []
        layers_ = []
        # building first layer
        first_output_channels = BlockSetting[0].in_channels
        layers.append(
            Stem(
                3,
                first_output_channels,
                kernel_size=3,
                stride=2,
                activation='HS'
            )          
        )

        # building stage1 other blocks
        for cnf in BlockSetting:
            layers.append(Block(cnf))
        
        for cnf in TBlockSetting:
            layers_.append(TBlock(self.padded_patch, cnf))
        

        last_output_channel = TBlockSetting[-1].out_channels
    
        self.Net = nn.Sequential(*layers)
        self.Net_ = nn.Sequential(*layers_)
        self.classifier = nn.Sequential(
            # nn.Conv2d(last_output_channel, last_channel, kernel_size = 1),
            # nn.ReLU(inplace=False),
            # nn.Dropout(p = self.dropout, inplace = False),
            nn.Conv2d(last_output_channel, num_joints, kernel_size = 1),
            nn.ReLU(inplace = False)) 
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                # nn.init.normal_(m.weight, 0, 0.01)
                # if m.bias is not None:
                #     nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
            
    def forward(self, x):
        # print(x.shape)
        x = F.pad(x, (self.pad[0], self.pad[0], self.pad[1], self.pad[1]))
        # print(x.shape)
        x = self.Net(x)
        x = [x, x]
        y = self.Net_(x)
        x = y[0]
        # print(x.shape)
        # [-1, num_joints, h, w]
        x = self.classifier(x)
        # print(x.shape)
        # x = self.pmaper(x)
        # print(x.shape)
        x = x.flatten(2)
        # print(x.shape)
        x = F.normalize(x, p = 1, dim = 2)
        # print(x)
        # x = F.softmax(x, dim = 2)
        # x_adjusted = self.get_adjusted(x)
        return x
        # return x , x_adjusted

def _mobileposnet_conf(arch: str):
    # norm_layer = nn.BatchNorm2d(eps=0.01, momentum=0.01)
    stage_conf = BlockConfig
    Tstage_conf = TBlockConfig
    
    if arch == "_PRVNet_32_16_8_ca":
        block_setting = [
            stage_conf(16, 1, 16, 3, 1, 1, 'RE', False),
            stage_conf(16, 6, 24, 3, 2, 1, 'RE', False),
            stage_conf(24, 6, 24, 3, 1, 1, 'RE', False),
            # stage_conf(24, 6, 40, 5, 2, 1, 'RE', True),
            # stage_conf(40, 6, 40, 5, 1, 1, 'RE', True),
        ]
        Tblock_setting = [
            Tstage_conf(24, 24, 40, 6, 40, 3, 1, 'HS', 'L'),
            Tstage_conf(40, 40, 80, 6, 40, 3, 1, 'HS', 'H'),
            Tstage_conf(40, 40, 160, 6, 40, 3, 1, 'HS', 'H'),
        ]
   
    elif arch == 'PRVNet_small':
        block_setting = [
            # stage_conf(16, 1, 16, 3, 1, 1, 'RE', False),
            stage_conf(16, 6, 24, 3, 2, 1, 'RE', False),
            # stage_conf(24, 6, 24, 3, 1, 1, 'RE', False),
            stage_conf(24, 6, 40, 5, 2, 1, 'RE', True),
            # stage_conf(40, 6, 40, 5, 1, 1, 'RE', True),
        ]
        Tblock_setting = [
            Tstage_conf(40, 40, 80, 6, 80, 3, 1, 'HS', 'L'),
            Tstage_conf(80, 112, 160, 6, 80, 5, 1, 'HS', 'H'),
            Tstage_conf(80, 112, 160, 6, 80, 5, 1, 'HS', 'H'),
        ]

    else:
        raise ValueError(f"Unsupported model type {arch}")
    return block_setting, Tblock_setting
    # return block_setting, Tblock_setting, last_channel

def get_pose_net(cfg, is_train):
    block_setting, Tbock_setting = _mobileposnet_conf(cfg.MODEL.NAME)
    pmap_size = (cfg.MODEL.IMAGE_SIZE[0] // cfg.MODEL.PATCH_SIZE[0], cfg.MODEL.IMAGE_SIZE[1] // cfg.MODEL.PATCH_SIZE[1])
    model = MobilePosNet(block_setting, Tbock_setting, cfg.MODEL.NUM_JOINTS, cfg.MODEL.PATCH_SIZE, pmap_size)
    if is_train:
        model.init_weights()

    return model

    

             
        
        

        





        