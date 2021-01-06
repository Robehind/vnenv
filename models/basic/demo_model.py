import torch
import torch.nn as nn
import torch.nn.functional as F
from .simple_model import SimpleMP
#model里不再分别设置state、target之类的概念，而是要对input中的每一个成分进行处理
#target可能有不同成分，以后修复
class LiteModel(torch.nn.Module):
    """一个简单的模型,model里不再分别设置state、target之类的概念，
    而是要对input中的每一个成分进行处理"""
    def __init__(
        self,
        action_sz,
        vobs_sz,
        tobs_sz,
    ):
        super(LiteModel, self).__init__()
        self.net = SimpleMP(action_sz, vobs_sz, tobs_sz)

    def forward(self, model_input):
        '''保证输入的数据都是torch的tensor'''
        return self.net(model_input['fc'], model_input['glove'])

if __name__ == "__main__":
    model = LiteModel(4,2048,300)
    input1 = torch.randn(1,2048)
    input2 = torch.randn(1,300)
    out = model.forward(dict(fc=input1, glove=input2))
    print(out['policy'])
    print(out['value'])
    out = model.forward(dict(fc=input1, glove=input2))
    print(out['policy'])
    print(out['value'])

