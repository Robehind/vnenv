import torch
import torch.nn as nn
import torch.nn.functional as F
from ..simple_struct import ConvNet1, SingleLinear, SingleLSTM

class Simple1(torch.nn.Module):
    """简单模型1"""
    def __init__(
        self,
        action_sz,
        obs_stack = 1
        #state_sz,
        #target_sz,
    ):
        super(Simple1, self).__init__()
        self.obs_stack = obs_stack
        #perception
        tobs_sz = 300
        tobs_embed_sz = 512

        conv_out_sz = 8192
        self.vobs_conv = nn.Sequential(
            ConvNet1(),
            SingleLinear(conv_out_sz, 2048),
            nn.ReLU(True),
        )
        vobs_embed_sz = 512
        self.vobs_embed = SingleLinear(2048*obs_stack, vobs_embed_sz)

        self.tobs_embed = SingleLinear(tobs_sz,tobs_embed_sz)
        #mem&infer
        infer_sz = 256
        self.infer = SingleLinear(tobs_embed_sz+vobs_embed_sz, infer_sz)
        #plan
        self.actor_linear = nn.Linear(infer_sz, action_sz)
        self.critic_linear = nn.Linear(infer_sz, 1)

    def forward(self, model_input):

        vobs = model_input['image|4'].view(-1,300,300,3).permute(0,3,1,2)
        vobs = F.interpolate(vobs,(256,256))

        vobs_embed = self.vobs_conv(vobs)
        vobs_embed = torch.flatten(vobs_embed).view(-1,self.obs_stack*2048)
        vobs_embed = F.relu(self.vobs_embed(vobs_embed),True)

        tobs_embed = F.relu(self.tobs_embed(model_input['glove']),True)

        x = torch.cat((vobs_embed, tobs_embed), dim=1)

        x = F.relu(self.infer(x), True)
        
        return dict(
            policy=self.actor_linear(x),
            value=self.critic_linear(x)
            )

class Simple2(torch.nn.Module):
    """简单模型2"""
    def __init__(
        self,
        action_sz,
        #state_sz,
        #target_sz,
    ):
        super(Simple2, self).__init__()
        #perception
        tobs_sz = 300
        tobs_embed_sz = 512

        conv_out_sz = 8192
        self.vobs_conv = nn.Sequential(
            ConvNet1(),
            SingleLinear(conv_out_sz, 2048),
            nn.ReLU(True),
        )
        vobs_embed_sz = 512
        self.vobs_embed = SingleLinear(2048, vobs_embed_sz)

        self.tobs_embed = SingleLinear(tobs_sz,tobs_embed_sz)
        #mem&infer
        infer_sz = 256
        self.infer = SingleLSTM(tobs_embed_sz+vobs_embed_sz, infer_sz)
        #plan
        self.actor_linear = nn.Linear(infer_sz, action_sz)
        self.critic_linear = nn.Linear(infer_sz, 1)

    def forward(self, model_input):

        vobs = model_input['image'].view(-1,300,300,3).permute(0,3,1,2)
        vobs = F.interpolate(vobs,(256,256))

        vobs_embed = self.vobs_conv(vobs)
        vobs_embed = torch.flatten(vobs_embed).view(-1,2048)
        vobs_embed = F.relu(self.vobs_embed(vobs_embed),True)

        tobs_embed = F.relu(self.tobs_embed(model_input['glove']),True)

        x = torch.cat((vobs_embed, tobs_embed), dim=1)

        x = F.relu(self.infer(x), True)
        
        return dict(
            policy=self.actor_linear(x),
            value=self.critic_linear(x)
            )

    def reset_hidden(self, thread):
        self.infer.reset_hidden(thread)


