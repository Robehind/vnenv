import torch.nn as nn
import torch.nn.functional as F
from .lstm_model import LstmModel

class LstmLinearLint(LstmModel):
    def __init__(
        self,
        action_sz,
        nsteps,
        #target_sz = 300,
        #dropout_rate = 0.25,
        ):
        super(LstmLinearLint, self).__init__(
            action_sz, 
            #target_sz,
            #dropout_rate,
            )
        self.feature_size = 512 + action_sz
        self.ll = nn.Linear(self.feature_size, 1)

    def learned_loss(self, H_input, params=None):
        #H_input = H.unsqueeze(0)
        #print(H.shape)
        if params == None:
            x = F.relu(self.ll(H_input),True).squeeze()
        else:
            x = F.relu(F.linear(
                    H_input, 
                    weight=params["ll.weight"],
                    bias=params["ll.bias"],
                    ),True).squeeze()
        #print(x.shape)
        return x.pow(2).sum(0).pow(0.5)