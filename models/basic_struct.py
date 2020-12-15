import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.net_utils import toFloatTensor
#不会对输入输出做激活函数处理的。 

class SingleLSTM(nn.Module):

    def __init__(
        self,
        input_sz,
        output_sz,
    ):
        super(SingleLSTM, self).__init__()
        self.input_sz = input_sz
        self.output_sz = output_sz
        self.layer1 = nn.LSTMCell(input_sz, output_sz)

    def forward(self, x, hidden, params = None):

        if params == None:
            out = self.layer1(x, hidden)
        else:
            out = torch.lstm_cell(
                x,
                self.hidden,
                params["layer1.weight_ih"],
                params["layer1.weight_hh"],
                params["layer1.bias_ih"],
                params["layer1.bias_hh"],
            )

        return out

class SingleLinear(nn.Module):

    def __init__(
        self,
        input_sz,
        output_sz
    ):
        super(SingleLinear, self).__init__()
        self.input_sz = input_sz
        self.output_sz = output_sz
        self.layer1 = nn.Linear(input_sz, output_sz)

    def forward(self, x, params = None):

        if params == None:
            out = self.layer1(x)
        else:
            out = F.linear(
                    x,
                    weight=params["layer1.weight"],
                    bias=params["layer1.bias"],
                )
        return out

class ConvNet1(nn.Module):
#参考自splitNet的前四层卷积层
    def __init__(
        self,
        input_channels = 3
    ):
        super(ConvNet1, self).__init__()
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
            nn.Flatten()
            )

    def forward(self, x, params = None):

        if params == None:
            out = self.net(x)
        else:
            raise NotImplementedError
        return out

    def output_shape(self,input_sz):
        pass