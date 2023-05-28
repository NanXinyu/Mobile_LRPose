import torch 
import torch.nn as nn

from typing import List, Optional
from thop import profile
from einops import rearrange
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def IMG_EMBEDDING(img):
    b, c, h, w = img.shape
    h_embed = torch.arange(0,h,1).reshape(1,1,h,1)#.reshape(-1,1)
    w_embed = torch.arange(0,w,1).reshape(1,1,1,w)
    # h_embed = torch.tile(h_embed/(h-1), (b, 1, 1, w)).cuda()
    # w_embed = torch.tile(w_embed/(w-1), (b, 1, h, 1)).cuda()
    # print(w_embed.repeat(b,1,h,1))
    h_embed = (h_embed.repeat(b,1,1,w)/(h-1)).reshape(b,1,h,w).cuda()
    w_embed = (w_embed.repeat(b,1,h,1)/(w-1)).reshape(b,1,h,w).cuda()
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
        super(PA, self).__init__()
        # self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # (b,c,h,w)-->(b,c,h,1)
        # self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # (b,c,h,w)-->(b,c,1,w)
 
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
    def __init__(self, inp, reduction = 16):
        super(CA, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  
 
        mip =  _make_divisible(inp // reduction, 8)  
 
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace = True)
 
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
 
    def forward(self, x):
        identity = x
 
        n, c, h, w = x.size()
        x_h = self.pool_h(x) 
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  
 
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
        am
    ):
        self.in_channels = in_channels
        self.exp_channels = exp_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.activation = activation
        self.am = am
        
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
                activation=cnf.activation)
                # MultiChannelFusion(cnf.in_channels)
                )

        # depthwise
        layers.append(Conv2dBNActivation(cnf.exp_channels, 
                                         cnf.exp_channels, 
                                         kernel_size=cnf.kernel_size,
                                         stride=cnf.stride,
                                         groups=cnf.exp_channels,
                                         dilation=cnf.dilation,
                                         activation=cnf.activation))

        if cnf.am == 'CA':
            layers.append(CA(cnf.exp_channels))
        elif cnf.am == 'LSA':
            layers.append(LSA(cnf.exp_channels))
        
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

class globalconv(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            in_seq,
            out_seq
            ):
        super(globalconv, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_seq, out_seq),
            nn.LayerNorm(out_seq),
            nn.ReLU(inplace = True))
            
        self.out = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size = 1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace = True)
        )
    def forward(self, x):
        x = self.mlp(x)
        x = self.out(x)
        return x

class MobilePosNet(nn.Module):
    def __init__(
        self,
        num_joints,
        img_size,
        representation_size,
        BlockSetting = [
            InvertedResBlockConfig(32, 32, 16, 3, 1, 1, 'RE', 'CA'),
            InvertedResBlockConfig(16, 96, 24, 3, 2, 1, 'RE', 'LSA'),
            InvertedResBlockConfig(24, 144, 24, 3, 1, 1, 'HS', 'LSA'),
            InvertedResBlockConfig(24, 144, 32, 3, 2, 1, 'HS', 'LSA'),
            InvertedResBlockConfig(32, 192, 32, 3, 1, 1, 'HS', 'LSA'),
            InvertedResBlockConfig(32, 192, 32, 3, 1, 1, 'HS', 'LSA'),
            InvertedResBlockConfig(32, 192, 64, 3, 2, 1, 'HS', 'LSA'),
            InvertedResBlockConfig(64, 384, 64, 3, 1, 1, 'HS', 'CA'),
            InvertedResBlockConfig(64, 384, 64, 3, 1, 1, 'HS', 'CA'),
            InvertedResBlockConfig(64, 384, 64, 3, 1, 1, 'HS', 'CA'),
            InvertedResBlockConfig(64, 384, 64, 3, 1, 1, 'HS', 'CA'),
            InvertedResBlockConfig(64, 768, 64, 3, 1, 1, 'HS', 'CA'),
            InvertedResBlockConfig(64, 768, 64, 3, 1, 1, 'HS', 'CA'),
            InvertedResBlockConfig(64, 768, 64, 3, 1, 1, 'HS', 'CA'),
        ]
    
    ):
        super(MobilePosNet, self).__init__()
        
        self.num_joints = num_joints
        layers = []
        # building first layer
        first_channels = BlockSetting[0].in_channels
        layers.append(
            Conv2dBNActivation(5, first_channels, kernel_size=3,
                               stride=2, activation='HS')          
        )

        # building stage other blocks
        for cnf in BlockSetting:
            layers.append(InvertedResBlock(cnf))
        
        output_channel = BlockSetting[-1].out_channels
        self.BaseLine = nn.Sequential(*layers)
        
        seq = representation_size[0] * representation_size[1]
        
        self.stage_level1 = nn.Sequential(
            Conv2dBNActivation(output_channel, output_channel//4, kernel_size = 3, stride = 2, activation = 'HS'),
            nn.UpsamplingBilinear2d(size=representation_size))
        self.stage_level2 = nn.Sequential(
            Conv2dBNActivation(output_channel, output_channel//4, kernel_size = 3, stride = 2, dilation=2, activation='HS'),
            nn.UpsamplingBilinear2d(size=representation_size))
        self.stage_level3 = nn.Sequential(
            Conv2dBNActivation(output_channel, output_channel//4, kernel_size=3, stride=2, dilation=4, activation='HS'),
            nn.UpsamplingBilinear2d(size=representation_size))
        self.stage_level4 = nn.Sequential(
            Conv2dBNActivation(output_channel, output_channel//4, kernel_size=3, stride = 2, dilation = 6, activation = 'HS'),
            nn.UpsamplingBilinear2d(size=representation_size))

        self.conv1 = Conv2dBNActivation(output_channel*2, num_joints*4, kernel_size=1, stride=1, activation='HS')
        self.conv2 = nn.Sequential(
            nn.Conv1d(num_joints, num_joints*16, kernel_size = 1, bias = False),
            nn.BatchNorm1d(num_joints*16),
            nn.Hardswish(inplace = True),
            nn.Conv1d(num_joints*16, num_joints, kernel_size = 1, bias = False),
            nn.BatchNorm1d(num_joints),
            nn.Hardswish(inplace = True),
        )
        
        self.generate_x = nn.Linear(seq*4, img_size[0])
        self.generate_y = nn.Linear(seq*4, img_size[1])
        
        
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
        x0 = self.BaseLine(img)
        x1 = self.stage_level1(x0)
        x2 = self.stage_level2(x0)
        x3 = self.stage_level3(x0)
        x4 = self.stage_level4(x0)
        x = torch.cat((x0,x1,x2,x3,x4),dim=1)
        joints = self.conv1(x)
        joints = rearrange(joints, 'b (n c) h w -> b c (n h w)', c = self.num_joints)
        joints = self.conv2(joints)
        pred_x = self.coord_x(joints)
        pred_y = self.coord_y(joints)
        
        return pred_x, pred_y



def get_pose_net(cfg, is_train):
    print("=" * 60)
    print(f"{cfg.MODEL.NAME} Model Generated!")
    print("=" * 60)
    
    img_size = cfg.MODEL.IMAGE_SIZE
    representation_size = cfg.MODEL.INTERMEDIATE_SIZE
    model = MobilePosNet(cfg.MODEL.NUM_JOINTS, 
                         img_size, 
                         representation_size)
    if is_train:
        model.init_weights()
    return model

def model_summary(cfg):
    img_size = cfg.MODEL.IMAGE_SIZE
    representation_size = cfg.MODEL.INTERMEDIATE_SIZE
    dummy_input = torch.randn(1, 3, 256, 256).cuda()
    model_ = MobilePosNet(cfg.MODEL.NUM_JOINTS, 
                         img_size, 
                         representation_size).cuda()
    flops, params = profile(model_, (dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

             
        
        

        





        