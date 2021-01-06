import torch
import torch.nn as nn
import torch.nn.functional as F
#辅助任务
class liteRGBpred(nn.Module):
    '''更轻量化的RGB预测网络'''
    def __init__(
        self,
        input_channels = 128
    ):
        super(liteRGBpred, self).__init__()
        self.input_channels = input_channels
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=(2,2),mode='bilinear',align_corners=True),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2,2),mode='bilinear',align_corners=True),
            nn.Conv2d(128, 64, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2,2),mode='bilinear',align_corners=True),
            nn.Conv2d(64, 32, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 4),
            nn.ReLU(inplace=True),
            )
    
    def forward(self, x, params = None):

        if params == None:
            out = self.net(x)
        else:
            raise NotImplementedError
        return out


class RGBpred(nn.Module):
#参考自splitNet的RGB还原网络
    def __init__(
        self,
        input_channels = 128
    ):
        super(RGBpred, self).__init__()
        self.input_channels = input_channels
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2,2),mode='bilinear',align_corners=True),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2,2),mode='bilinear',align_corners=True),
            nn.Conv2d(128, 64, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2,2),mode='bilinear',align_corners=True),
            nn.Conv2d(64, 32, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2,2),mode='bilinear',align_corners=True),
            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2,2),mode='bilinear',align_corners=True),
            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, 1, padding=1),
            )
        

    def forward(self, x, params = None):

        if params == None:
            out = self.net(x)
        else:
            raise NotImplementedError
        return out

    #def output_sz(self,h,w):
        #return CNNout_sz(self.net,h,w)