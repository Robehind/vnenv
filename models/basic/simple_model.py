import torch
import torch.nn as nn
import torch.nn.functional as F
from ..basic_struct import SingleLinear, SingleLSTM
from ..perception.simple_cnn import CNNout_sz, House3DCNN, SplitNetCNN

class SplitLinear(torch.nn.Module):
    """简单模型1, splitnet + linear, 延时堆叠"""
    def __init__(
        self,
        action_sz,
        obs_stack = 1
        #state_sz,
        #target_sz,
    ):
        super(SplitLinear, self).__init__()
        self.obs_stack = obs_stack
        #perception
        tobs_sz = 300
        tobs_embed_sz = 512

        conv_out_sz = CNNout_sz(SplitNetCNN().net,128,128)
        self.vobs_conv = F.relu(SplitNetCNN(), True)
        vobs_embed_sz = 512
        self.vobs_embed = SingleLinear(conv_out_sz*obs_stack, vobs_embed_sz)

        self.tobs_embed = SingleLinear(tobs_sz,tobs_embed_sz)
        #mem&infer
        infer_sz = 512
        self.infer = SingleLinear(tobs_embed_sz+vobs_embed_sz, infer_sz)
        #plan
        self.actor_linear = nn.Linear(infer_sz, action_sz)
        self.critic_linear = nn.Linear(infer_sz, 1)

    def forward(self, model_input):

        vobs = model_input['image|4'].view(-1,128,128,3).permute(0,3,1,2)
        #vobs = F.interpolate(vobs,(128,128))

        vobs_embed = self.vobs_conv(vobs)
        vobs_embed = torch.flatten(vobs_embed).view(-1,self.obs_stack*2048)
        vobs_embed = F.relu(self.vobs_embed(vobs_embed), True)

        tobs_embed = F.relu(self.tobs_embed(model_input['glove']),True)

        x = torch.cat((vobs_embed, tobs_embed), dim=1)

        x = F.relu(self.infer(x), True)
        
        return dict(
            policy=self.actor_linear(x),
            value=self.critic_linear(x)
            )

class SplitLstm(torch.nn.Module):
    """简单模型2,是简单模型1带LSTM的版本"""
    def __init__(
        self,
        action_sz,
        #state_sz,
        #target_sz,
    ):
        super(SplitLstm, self).__init__()
        #perception
        tobs_sz = 300
        tobs_embed_sz = 512

        conv_out_sz = CNNout_sz(SplitNetCNN().net,128,128)
        self.vobs_conv = SplitNetCNN()
        vobs_embed_sz = 512
        self.vobs_embed = SingleLinear(conv_out_sz, vobs_embed_sz)

        self.tobs_embed = SingleLinear(tobs_sz,tobs_embed_sz)
        #mem&infer
        infer_sz = 512
        self.hidden_sz = infer_sz
        self.infer = SingleLSTM(tobs_embed_sz+vobs_embed_sz, infer_sz)
        #plan
        self.actor_linear = nn.Linear(infer_sz, action_sz)
        self.critic_linear = nn.Linear(infer_sz, 1)

    def forward(self, model_input):

        vobs = model_input['image'].view(-1,128,128,3).permute(0,3,1,2)
        #vobs = F.interpolate(vobs,(128,128))
        hidden = model_input['hidden']

        vobs_embed = self.vobs_conv(vobs)
        #vobs_embed = torch.flatten(vobs_embed).view(-1,2048)
        vobs_embed = F.relu(self.vobs_embed(vobs_embed),True)

        tobs_embed = F.relu(self.tobs_embed(model_input['glove']),True)

        x = torch.cat((vobs_embed, tobs_embed), dim=1)

        (hx, cx) = self.infer(x, hidden)
        x = hx
        
        return dict(
            policy=self.actor_linear(x),
            value=self.critic_linear(x),
            hidden = (hx, cx)
            )

class Simple3(torch.nn.Module):
    """简单模型3,House3D CNN带LSTM的版本"""
    def __init__(
        self,
        action_sz,
        #state_sz,
        #target_sz,
    ):
        super(Simple3, self).__init__()
        #perception
        tobs_sz = 300
        tobs_embed_sz = 512

        conv_out_sz = CNNout_sz(House3DCNN().net,120,90)
        self.vobs_conv = nn.Sequential(
            House3DCNN(),
            SingleLinear(conv_out_sz, 1024),
            nn.ReLU(True),
        )
        vobs_embed_sz = 512
        self.vobs_embed = SingleLinear(1024, vobs_embed_sz)

        self.tobs_embed = SingleLinear(tobs_sz,tobs_embed_sz)
        #mem&infer
        infer_sz = 512
        self.hidden_sz = infer_sz
        self.infer = SingleLSTM(tobs_embed_sz+vobs_embed_sz, infer_sz)
        #plan
        self.actor_linear = nn.Linear(infer_sz, action_sz)
        self.critic_linear = nn.Linear(infer_sz, 1)

    def forward(self, model_input):

        vobs = model_input['image'].view(-1,300,300,3).permute(0,3,1,2)
        vobs = F.interpolate(vobs,(120,90))
        hidden = model_input['hidden']

        vobs_embed = self.vobs_conv(vobs)
        vobs_embed = torch.flatten(vobs_embed).view(-1,1024)
        vobs_embed = F.relu(self.vobs_embed(vobs_embed),True)

        tobs_embed = F.relu(self.tobs_embed(model_input['glove']),True)

        x = torch.cat((vobs_embed, tobs_embed), dim=1)

        (hx, cx) = self.infer(x, hidden)
        x = hx
        
        return dict(
            policy=self.actor_linear(x),
            value=self.critic_linear(x),
            hidden = (hx, cx)
            )


