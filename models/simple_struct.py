import torch
import torch.nn as nn
import torch.nn.functional as F
#不会对输入输出做激活函数处理的。

class SingleLSTM(nn.Module):

    def __init__(
        self,
        input_sz,
        output_sz,
        batch_size = 1
    ):
        super(SingleLSTM, self).__init__()
        self.input_sz = input_sz
        self.output_sz = output_sz
        self.batch_size = batch_size
        self.layer1 = nn.LSTMCell(input_sz, output_sz)
        self.hidden = (
            torch.zeros(batch_size, output_sz),
            torch.zeros(batch_size, output_sz)
        )

    def forward(self, x, params = None):

        if params == None:
            self.hidden = self.layer1(x, self.hidden)
        else:
            self.hidden = torch.lstm_cell(
                x,
                self.hidden,
                params["layer1.weight_ih"],
                params["layer1.weight_hh"],
                params["layer1.bias_ih"],
                params["layer1.bias_hh"],
            )

        return self.hidden[0]

    def reset_hidden(self, thread = 0):

        self.hidden[0][thread] = torch.zeros(1, self.output_sz)
        self.hidden[1][thread] = torch.zeros(1, self.output_sz)


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