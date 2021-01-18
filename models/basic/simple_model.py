import torch
import torch.nn as nn
import torch.nn.functional as F
from ..perception.simple_cnn import CNNout_sz, House3DCNN, SplitNetCNN
from ..plan.rl_linear import AClinear
from torchvision import transforms as T
from utils.net_utils import weights_init, norm_col_init

class SimpleMP(torch.nn.Module):
    """vobs和tobs都是已经被处理好的特征向量,没有封装input，不能直接用于训练"""
    def __init__(
        self,
        action_sz,
        vobs_sz,
        tobs_sz = 300,
        tobs_embed_sz = 512,
        vobs_embed_sz = 512,
        infer_sz = 512,
        mode = 0 #0 for linear 1 for lstm
    ):
        super(SimpleMP, self).__init__()

        self.vobs_embed = nn.Linear(vobs_sz, vobs_embed_sz)
        self.tobs_embed = nn.Linear(tobs_sz, tobs_embed_sz)
        #mem&infer
        self.mode = mode
        if mode == 0:
            self.infer = nn.Linear(tobs_embed_sz+vobs_embed_sz, infer_sz)
        else:
            self.hidden_sz = infer_sz
            self.infer = nn.LSTMCell(tobs_embed_sz+vobs_embed_sz, infer_sz)
        #plan
        self.ac_out = AClinear(action_sz, infer_sz)
        self.apply(weights_init)
        self.ac_out.actor_linear.weight.data = norm_col_init(
            self.ac_out.actor_linear.weight.data, 0.01
        )
        self.ac_out.critic_linear.weight.data = norm_col_init(
            self.ac_out.critic_linear.weight.data, 1.0
        )

    def forward(self, vobs, tobs, hidden = None):

        vobs_embed = F.relu(self.vobs_embed(vobs), True)
        tobs_embed = F.relu(self.tobs_embed(tobs), True)
        x = torch.cat((vobs_embed, tobs_embed), dim=1)
        if self.mode == 0:
            x = self.infer(x)
            return self.ac_out(x)
        (x, cx) = self.infer(x, hidden)
        out = self.ac_out(x)
        out.update(dict(hidden=(x,cx)))
        return out

class SplitLinear(torch.nn.Module):
    """简单模型1, splitnet + linear, 延时堆叠"""
    def __init__(
        self,
        action_sz,
        vobs_sz = (128, 128, 3),
        tobs_sz = 300,
        obs_stack = 1
        #state_sz,
        #target_sz,
    ):
        super(SplitLinear, self).__init__()
        self.obs_stack = obs_stack
        self.vobs_sz = vobs_sz
        #perception
        CNN = SplitNetCNN()
        self.conv_out_sz = CNN.out_fc_sz(vobs_sz[0], vobs_sz[1])
        self.vobs_conv = nn.Sequential(
            CNN,
            nn.Flatten(),
        )
        self.MP = SimpleMP(action_sz, self.conv_out_sz*obs_stack, tobs_sz)

    def forward(self, model_input):

        vobs = model_input['image|4'].view(-1,*self.vobs_sz).permute(0,3,1,2)
        vobs = vobs / 255.

        vobs_embed = self.vobs_conv(vobs)
        vobs_embed = torch.flatten(vobs_embed).view(-1,self.obs_stack*self.conv_out_sz)

        return self.MP(vobs_embed, model_input['glove'])

class SplitLstm(torch.nn.Module):
    """简单模型2,是简单模型1带LSTM的版本"""
    def __init__(
        self,
        action_sz,
        vobs_sz = (128, 128, 3),
        tobs_sz = 300,
    ):
        super(SplitLstm, self).__init__()
        #perception
        self.vobs_sz = vobs_sz
        CNN = SplitNetCNN()
        self.conv_out_sz = CNN.out_fc_sz(vobs_sz[0], vobs_sz[1])
        self.vobs_conv = nn.Sequential(
            CNN,
            nn.Flatten(),
        )
        self.MP = SimpleMP(action_sz, self.conv_out_sz, tobs_sz, mode = 1)
        self.hidden_sz = self.MP.hidden_sz

        mean = torch.tensor([0.5269, 0.4565, 0.3687]).view(1,3,1,1)
        self.mean = torch.nn.Parameter(mean)
        self.mean.requires_grad = False
        std = torch.tensor([0.0540, 0.0554, 0.0567]).view(1,3,1,1)
        self.std = torch.nn.Parameter(std)
        self.std.requires_grad = False

    def forward(self, model_input):

        vobs = model_input['image'].permute(0,3,1,2)/ 255.
        vobs = (vobs-self.mean)/self.std
        vobs_embed = self.vobs_conv(vobs)

        return self.MP(vobs_embed, model_input['glove'], model_input['hidden'])

class FcLstmModel(torch.nn.Module):
    """观察都是预处理好的特征向量的lstm模型，类似于LiteModel"""
    def __init__(
        self,
        action_sz,
        vobs_sz = 2048,
        tobs_sz = 300,
    ):
        super(FcLstmModel, self).__init__()
        self.net = SimpleMP(action_sz, vobs_sz, tobs_sz, mode = 1)
        self.hidden_sz = self.net.hidden_sz

    def forward(self, model_input):
        return self.net(model_input['fc'], model_input['glove'], model_input['hidden'])

class FcLinearModel(torch.nn.Module):
    """观察都是预处理好的特征向量的linear模型，类似于LiteModel"""
    def __init__(
        self,
        action_sz,
        vobs_sz = 2048,
        tobs_sz = 300,
    ):
        super(FcLinearModel, self).__init__()
        self.net = SimpleMP(action_sz, vobs_sz, tobs_sz)
        #self.hidden_sz = self.net.hidden_sz

    def forward(self, model_input):
        return self.net(model_input['fc'], model_input['glove'])
