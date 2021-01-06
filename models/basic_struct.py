import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.net_utils import toFloatTensor
#不会对输入输出做激活函数处理的。基本结构不封装model input，因为他们不是最终模型，不会被直接用于训练

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

class SimpleMP1(torch.nn.Module):
    """简单的后端网络，包括LSTM和决策输出层，输入是obs的embedding向量"""
    def __init__(
        self,
        action_sz,
        vobs_sz,
        tobs_sz = 300,
    ):
        super(SimpleMP1, self).__init__()
        tobs_embed_sz = 512
        vobs_embed_sz = 512
        infer_sz = 512

        self.vobs_embed = SingleLinear(vobs_sz, vobs_embed_sz)
        self.tobs_embed = SingleLinear(tobs_sz, tobs_embed_sz)
        #mem&infer
        self.hidden_sz = infer_sz
        self.infer = SingleLSTM(tobs_embed_sz+vobs_embed_sz, infer_sz)
        #plan
        self.actor_linear = nn.Linear(infer_sz, action_sz)
        self.critic_linear = nn.Linear(infer_sz, 1)

    def forward(self, vobs, tobs, hidden):

        vobs_embed = F.relu(self.vobs_embed(vobs), True)
        tobs_embed = F.relu(self.tobs_embed(tobs), True)
        x = torch.cat((vobs_embed, tobs_embed), dim=1)
        (hx, cx) = self.infer(x, hidden)
        x = hx
        
        return dict(
            policy=self.actor_linear(x),
            value=self.critic_linear(x),
            hidden = (hx, cx)
            )

class SimpleMP2(torch.nn.Module):
    """简单的后端网络，纯线性，输入是obs的embedding向量"""
    def __init__(
        self,
        action_sz,
        vobs_sz,
        tobs_sz = 300,
    ):
        super(SimpleMP2, self).__init__()
        tobs_embed_sz = 512
        vobs_embed_sz = 512
        infer_sz = 512

        self.vobs_embed = SingleLinear(vobs_sz, vobs_embed_sz)
        self.tobs_embed = SingleLinear(tobs_sz, tobs_embed_sz)
        #mem&infer
        self.hidden_sz = infer_sz
        self.infer = SingleLinear(tobs_embed_sz+vobs_embed_sz, infer_sz)
        #plan
        self.actor_linear = nn.Linear(infer_sz, action_sz)
        self.critic_linear = nn.Linear(infer_sz, 1)

    def forward(self, vobs, tobs):

        vobs_embed = F.relu(self.vobs_embed(vobs), True)
        tobs_embed = F.relu(self.tobs_embed(tobs), True)
        x = torch.cat((vobs_embed, tobs_embed), dim=1)
        x = self.infer(x)
        
        return dict(
            policy=self.actor_linear(x),
            value=self.critic_linear(x),
            )