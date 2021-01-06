import torch
import torch.nn as nn
import torch.nn.functional as F

#No activate func, no flatten

def conv2dout_sz(H, K, S, P):
    return (H + 2*P - K)//S + 1

def deconv2dout_sz(H,K,S,P):
    return (H-1)*S-2*P+(K-1)+1

def CNNout_sz(net,h,w):
    H,W,out_c = CNNout_HWC(net, h, w)
    return H*W*out_c

def CNNout_HWC(net,h,w):
    H,W = h,w
    for c in net:
        name = c.__class__.__name__
        if  'Conv2d' in name:
            out_c = c.out_channels
            k,s,p = c.kernel_size,c.stride,c.padding
            H = conv2dout_sz(H,k[0],s[0],p[0])
            W = conv2dout_sz(W,k[1],s[1],p[1])
        elif 'Pool' in name:
            k,s,p = c.kernel_size,c.stride,c.padding
            H = conv2dout_sz(H,k,s,p)
            W = conv2dout_sz(W,k,s,p)
        elif 'sample' in name:
            a,b = c.scale_factor
            H *= a
            W *= b
        elif 'ConvTranspose2d' in name:
            out_c = c.out_channels
            k,s,p = c.kernel_size,c.stride,c.padding
            H = deconv2dout_sz(H,k[0],s[0],p[0])
            W = deconv2dout_sz(W,k[1],s[1],p[1])
    
    return int(H),int(W),int(out_c)

class SplitNetCNN(nn.Module):
#参考自splitNet的前四层卷积层
    def __init__(
        self,
        input_channels = 3
    ):
        super(SplitNetCNN, self).__init__()
        self.input_channels = input_channels
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, 7, 4, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2,2),
            #nn.Flatten()
            )

    def forward(self, x):
        return self.net(x)

    def out_fc_sz(self,h,w):
        return CNNout_sz(self.net,h,w)

    def out_sz(self,h,w):
        return CNNout_HWC(self.net,h,w)

class House3DCNN(nn.Module):
#House3D CNN
#输入数据为120x90的时候输出1024维向量
    def __init__(
        self,
        input_channels = 3
    ):
        super(House3DCNN, self).__init__()
        self.input_channels = input_channels
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 5, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 5, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 5, 2),
            nn.ReLU(inplace=True),
            #nn.Flatten()
            )

    def forward(self, x):
        return self.net(x)

    def out_fc_sz(self,h,w):
        return CNNout_sz(self.net,h,w)

    def out_sz(self,h,w):
        return CNNout_HWC(self.net,h,w)