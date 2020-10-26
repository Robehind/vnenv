import torch.nn as nn
from .lstm_model import LstmModel
from .tcn import TemporalConvNet


class LiteSAVN(LstmModel):
    def __init__(
        self,
        action_sz,
        nsteps,
        #target_sz = 300,
        #dropout_rate = 0.25,
        ):
        super(LiteSAVN, self).__init__(
            action_sz, 
            #target_sz,
            #dropout_rate,
            )
        self.num_steps = nsteps
        self.ll_tc = TemporalConvNet(
            self.num_steps, [10, 1], kernel_size=2, dropout=0.0
        )

    def learned_loss(self, H, params=None):
        H_input = H.unsqueeze(0)
        x = self.ll_tc(H_input, params).squeeze(0)
        return x.pow(2).sum(1).pow(0.5)
