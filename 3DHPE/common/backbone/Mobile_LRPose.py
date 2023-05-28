import torch 
import torch.nn as nn

from typing import List, Optional
from thop import profile
import math
from einops import rearrange
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def IMG_EMBEDDING(img):
    b, c, h, w = img.shape
    h_embed = torch.arange(0,h,1).reshape(-1,1)
    w_embed = torch.arange(0,w,1)
    h_embed = torch.tile(h_embed/(h-1), (b, 1, 1, w)).cuda()
    w_embed = torch.tile(w_embed/(w-1), (b, 1, h, 1)).cuda()
    return torch.cat((img, h_embed, w_embed), dim = 1)

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

class LSA(nn.Module):
    def __init__(self, inp, stride = 4, reduction = 16):
        super(LSA, self).__init__()
 
        mip =  _make_divisible(inp // reduction, 8)  
 
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace = True)
 
        self.conv2 = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor = stride)
        self.pool = nn.AvgPool2d(kernel_size = 5, stride = stride, padding = 2)
 
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()     
        scale = self.pool(x)

        scale = self.conv1(scale)
        scale = self.bn1(scale)
        scale = self.act(scale)
 
        scale = self.upsample(self.conv2(scale)).sigmoid()
 
        out = identity * scale
 
        return out

        
class CA(nn.Module):
    def __init__(self, inp, reduction = 4):
        super(CA, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # (b,c,h,w)-->(b,c,h,1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # (b,c,h,w)-->(b,c,1,w)
 
        mip =  _make_divisible(inp // reduction, 8)  
 
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

    
class Conv2dBNActivation(nn.Module):
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
        super(Conv2dBNActivation, self).__init__()
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

class InvertedResBlockConfig:
    def __init__(
        self,
        in_channels,
        exp_channels,
        out_channels,
        kernel_size,
        stride,
        dilation,
        activation,
        use_ca = True,
    ):
        self.in_channels = in_channels
        self.exp_channels = exp_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.activation = activation
        self.use_ca = use_ca
        
class InvertedResBlock(nn.Module):
    def __init__(
        self,
        cnf: InvertedResBlockConfig,
        ):
        super(InvertedResBlock, self).__init__()
        self.use_res_connect = cnf.stride == 1 and cnf.in_channels == cnf.out_channels

        layers = []
        # expand
        if cnf.exp_channels != cnf.in_channels:
            layers.append(
                Conv2dBNActivation(
                cnf.in_channels, 
                cnf.exp_channels, 
                kernel_size=1,
                activation=cnf.activation))

        # depthwise
        layers.append(Conv2dBNActivation(cnf.exp_channels, 
                                         cnf.exp_channels, 
                                         kernel_size=cnf.kernel_size,
                                         stride=cnf.stride,
                                         groups=cnf.exp_channels,
                                         dilation=cnf.dilation,
                                         activation=cnf.activation))
        
        if cnf.use_ca == True:
            layers.append(CA(cnf.exp_channels, 16))
        else:
            layers.append(PA(cnf.exp_channels))
        layers.append(
            Conv2dBNActivation(
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

    
class MobilePosNet(nn.Module):
    def __init__(
        self,
        num_joints,
        depth_dim,
        embedding_size,
        output_shape,
    ):
        super(MobilePosNet, self).__init__()
        stage_conf = InvertedResBlockConfig
        block_settings = [
            stage_conf(32, 32, 16, 3, 1, 1, 'RE', True),
            stage_conf(16, 96, 24, 3, 2, 1, 'RE', False),
            stage_conf(24, 144, 24, 3, 1, 1, 'HS', False),
            stage_conf(24, 144, 32, 3, 2, 1, 'HS', False),
            stage_conf(32, 192, 32, 3, 1, 1, 'HS', False),
            stage_conf(32, 192, 32, 3, 1, 1, 'HS', False),
            stage_conf(32, 192, 64, 3, 2, 1, 'HS', False),
            stage_conf(64, 384, 64, 3, 1, 1, 'HS', True),
            stage_conf(64, 384, 64, 3, 1, 1, 'HS', True),
            stage_conf(64, 384, 64, 3, 1, 1, 'HS', True),
            stage_conf(64, 384, 64, 3, 1, 1, 'HS', True),
            stage_conf(64, 768, 64, 3, 1, 1, 'HS', True),
            stage_conf(64, 768, 64, 3, 1, 1, 'HS', True),
            stage_conf(64, 768, 64, 3, 1, 1, 'HS', True),
            
        ]
        
        self.num_joints = num_joints
        self.depth_dim = depth_dim
        self.output_shape = output_shape
        layers = []
        first_channel = block_settings[0].in_channels
        layers.append(
            Conv2dBNActivation(5, first_channel, 3, 2, activation='HS')
        )
        # building first layer
        for cnf in block_settings:
            layers.append(InvertedResBlock(cnf))
        self.Baseline = nn.Sequential(*layers)
        heatmaps_size = output_shape
        output_channel = block_settings[-1].out_channels
        # self.conv1 = Conv2dBNActivation(block_settings[-1].out_channels, output_channel, kernel_size=1)
        self.stage_level1 = nn.Sequential(
            Conv2dBNActivation(output_channel, output_channel//4, kernel_size = 3, stride = 2, activation = 'HS'),
            nn.UpsamplingBilinear2d(scale_factor=2))
        self.stage_level2 = nn.Sequential(
            Conv2dBNActivation(output_channel, output_channel//4, kernel_size = 3, stride = 2, dilation=2, activation='HS'),
            nn.UpsamplingBilinear2d(scale_factor=2))
        self.stage_level3 = nn.Sequential(
            Conv2dBNActivation(output_channel, output_channel//4, kernel_size=3, stride=2, dilation=4, activation='HS'),
            nn.UpsamplingBilinear2d(scale_factor=2))
        self.stage_level4 = nn.Sequential(
            Conv2dBNActivation(output_channel, output_channel//4, kernel_size=3, stride = 2, dilation = 5, activation = 'HS'),
            nn.UpsamplingBilinear2d(scale_factor=2))
        self.conv2 = nn.Sequential(
            Conv2dBNActivation(output_channel*2, 1024, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(1024, num_joints*depth_dim, kernel_size = 1, bias = False))
            
        
       
        
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
            
    def forward(self, img):
        img = IMG_EMBEDDING(img)
        x0 = self.Baseline(img)
        # x0 = self.conv1(x0)
        x1 = self.stage_level1(x0)
        x2 = self.stage_level2(x0)
        x3 = self.stage_level3(x0)
        x4 = self.stage_level4(x0)
        # print(x4.shape)
        x = torch.cat((x0,x1,x2,x3,x4),dim=1)
        x = self.conv2(x)
        
        x = rearrange(x, 'b (c d) h w -> b c d h w', c = self.num_joints)
        
        
        return x


if __name__ == "__main__":
    model = MobilePosNet(18, 32, 768, [16, 16]).cuda()
    test_data = torch.rand(1, 3, 256, 256).cuda()
    test_outputs = model(test_data)
    # print(test_outputs.size())
    flops, params = profile(model, (test_data,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

             
        
        

        





        