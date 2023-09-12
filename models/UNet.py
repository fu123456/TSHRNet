import torch.nn.functional as F
import torch.nn as nn
import torch

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class Cvi(nn.Module):
    def __init__(self, in_channels, out_channels, before=None, after=False, kernel_size=4, stride=2,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Cvi, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv.apply(weights_init('gaussian'))

        if after=='BN':
            self.after = nn.BatchNorm2d(out_channels)
        elif after=='Tanh':
            self.after = torch.tanh
        elif after=='sigmoid':
            self.after = torch.sigmoid

        if before=='ReLU':
            self.before = nn.ReLU(inplace=True)
        elif before=='LReLU':
            self.before = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        if hasattr(self, 'before'):
            x = self.before(x)

        x = self.conv(x)

        if hasattr(self, 'after'):
            x = self.after(x)

        return x


class CvTi(nn.Module):
    def __init__(self, in_channels, out_channels, before=None, after=False, kernel_size=4, stride=2,
                 padding=1, dilation=1, groups=1, bias=False):
        super(CvTi, self).__init__()
        # with errors: TypeError: conv_transpose2d(): argument 'output_padding' (position 6) must be tuple of ints, not tuple
        # self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.conv.apply(weights_init('gaussian'))

        if after=='BN':
            self.after = nn.BatchNorm2d(out_channels)
        elif after=='Tanh':
            self.after = torch.tanh
        elif after=='sigmoid':
            self.after = torch.sigmoid

        if before=='ReLU':
            self.before = nn.ReLU(inplace=True)
        elif before=='LReLU':
            self.before = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        if hasattr(self, 'before'):
            x = self.before(x)

        x = self.conv(x)

        if hasattr(self, 'after'):
            x = self.after(x)

        return x

class UNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=1):
        super(UNet, self).__init__()

        self.Cv0 = Cvi(input_channels, 64)

        self.Cv1 = Cvi(64, 128, before='LReLU', after='BN', dilation=1)

        self.Cv2 = Cvi(128, 256, before='LReLU', after='BN', dilation=1)

        self.Cv3 = Cvi(256, 512, before='LReLU', after='BN', dilation=1)

        self.Cv4 = Cvi(512, 512, before='LReLU', after='BN', dilation=1)

        self.Cv5 = Cvi(512, 512, before='LReLU', dilation=1)

        self.CvT6 = CvTi(512, 512, before='ReLU', after='BN', dilation=1)

        self.CvT7 = CvTi(1024, 512, before='ReLU', after='BN', dilation=1)

        self.CvT8 = CvTi(1024, 256, before='ReLU', after='BN', dilation=1)

        self.CvT9 = CvTi(512, 128, before='ReLU', after='BN', dilation=1)

        self.CvT10 = CvTi(256, 64, before='ReLU', after='BN', dilation=1)

        self.CvT11 = CvTi(128, output_channels, before='ReLU', after='Tanh', dilation=1)

    def forward(self, input):
        # encoder
        x0 = self.Cv0(input)
        x1 = self.Cv1(x0)
        x2 = self.Cv2(x1)
        x3 = self.Cv3(x2)
        x4_1 = self.Cv4(x3)
        x4_2 = self.Cv4(x4_1)
        x4_3 = self.Cv4(x4_2)
        x5 = self.Cv5(x4_3)

        # decoder
        x6 = self.CvT6(x5)

        cat1_1 = torch.cat([x6, x4_3], dim=1)
        x7_1 = self.CvT7(cat1_1)
        cat1_2 = torch.cat([x7_1, x4_2], dim=1)
        x7_2 = self.CvT7(cat1_2)
        cat1_3 = torch.cat([x7_2, x4_1], dim=1)
        x7_3 = self.CvT7(cat1_3)

        cat2 = torch.cat([x7_3, x3], dim=1)
        x8 = self.CvT8(cat2)

        cat3 = torch.cat([x8, x2], dim=1)
        x9 = self.CvT9(cat3)

        cat4 = torch.cat([x9, x1], dim=1)
        x10 = self.CvT10(cat4)

        cat5 = torch.cat([x10, x0], dim=1)
        out = self.CvT11(cat5)

        return out
