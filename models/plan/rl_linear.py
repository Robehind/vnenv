import torch
import torch.nn as nn
import torch.nn.functional as F
#输出是用dict封装的
class AClinear(torch.nn.Module):
    """Actor-critic的输出层, 挂一个激活函数的"""
    def __init__(
        self,
        action_sz,
        infer_sz,
    ):
        super(AClinear, self).__init__()
        self.actor_linear = nn.Linear(infer_sz, action_sz)
        self.critic_linear = nn.Linear(infer_sz, 1)

    def forward(self, x):

        x = F.relu(x, True)
        return dict(
            policy=self.actor_linear(x),
            value=self.critic_linear(x),
            )