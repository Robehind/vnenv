import torch
import torch.nn as nn
import torch.nn.functional as F
from ..basic_struct import SingleLinear, SingleLSTM
from ..perception.auxiliary_net import RGBpred, liteRGBpred
from ..perception.simple_cnn import SplitNetCNN, CNNout_HWC

class SplitLstmRGBpred(torch.nn.Module):
    """RGB预测作为在线辅助任务的SplitLstm"""
    def __init__(
        self,
        action_sz,
        nsteps = 10
        #state_sz,
        #target_sz,
    ):
        super(SplitLstmRGBpred, self).__init__()
        #perception
        tobs_sz = 300
        tobs_embed_sz = 512
        self.nsteps = nsteps

        H,W,C = CNNout_HWC(SplitNetCNN().net,128,128)
        conv_out_sz =  H*W*C
        self.vobs_conv = SplitNetCNN()
        self.RGBpred = liteRGBpred(C)
            
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

        vobs = model_input['image']#.view(-1,128,128,3).permute(0,3,1,2)
        #vobs = vobs.to(torch.float32)/255 
        #vobs = F.interpolate(vobs,(128,128))
        hidden = model_input['hidden']

        vobs_embed = self.vobs_conv(vobs)
        vobs_embed_f = torch.flatten(vobs_embed, 1)

        RGBloss = 0
        if vobs_embed.shape[0] > self.nsteps:
            RGBloss = F.smooth_l1_loss(self.RGBpred(vobs_embed), vobs)
        #vobs_embed = torch.flatten(vobs_embed).view(-1,2048)
        vobs_embed = F.relu(self.vobs_embed(vobs_embed_f),True)

        tobs_embed = F.relu(self.tobs_embed(model_input['glove']),True)

        x = torch.cat((vobs_embed, tobs_embed), dim=1)

        (hx, cx) = self.infer(x, hidden)
        x = hx
        
        return dict(
            policy=self.actor_linear(x),
            value=self.critic_linear(x),
            hidden = (hx, cx),
            rgb_loss = RGBloss
            )