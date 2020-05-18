import torch
from utils.net_utils import gpuify
import torch.nn.functional as F
import numpy as np
def a3c_loss(
    done,
    v_batch,
    pi_batch,
    last_v,
    exps,
    gpu_id = -1,
    gamma = 0.99,#discount factor for exps['rewards']
    #tau = 1.00,#parameter for GAE
    #beta = 1e-2,#entropy regularization term
    ):
    
    R = 0.0 if done else last_v
    policy_loss = 0
    value_loss = 0
    td_target_lst = []
    for reward in exps['rewards'][::-1]:
        R = gamma * R + reward
        td_target_lst.append([R])
    td_target_lst.reverse()

    a_batch = torch.tensor(exps['action_idxs'])
    a_batch = gpuify(a_batch, gpu_id)
    
    td_target = torch.FloatTensor(td_target_lst)
    td_target = gpuify(td_target, gpu_id)
    
    advantage = td_target - v_batch.detach()
    advantage = gpuify(advantage, gpu_id)
    

    pi_a = F.softmax(pi_batch, dim = 1).gather(1, a_batch)
    policy_loss = -torch.log(pi_a) * advantage.detach()
    value_loss = 0.5*F.smooth_l1_loss(v_batch, td_target.detach())
    total_loss = policy_loss + value_loss

    return dict(
        total_loss=total_loss, 
        policy_loss=policy_loss, 
        value_loss=value_loss
        )

def a2c_loss(
    v_batch,
    pi_batch,
    last_v,
    exps,
    gpu_id = -1,
    gamma = 0.99,#discount factor for exps['rewards']
    #tau = 1.00,#parameter for GAE
    #beta = 1e-2,#entropy regularization term
    ):
    
    policy_loss = 0
    value_loss = 0

    R = last_v
    td_target = list()

    for r, mask in zip(exps['rewards'][::-1], exps['masks'][::-1]):
        R = r + gamma * R * mask
        td_target.append(R)

    td_target = torch.FloatTensor(td_target[::-1]).reshape(-1,1)
    td_target = gpuify(td_target, gpu_id)

    a_batch = torch.tensor(exps['action_idxs']).reshape(-1, 1)
    a_batch = gpuify(a_batch, gpu_id)
    
    advantage = td_target - v_batch.detach()
    advantage = gpuify(advantage, gpu_id)

    pi_a = F.softmax(pi_batch, dim = 1).gather(1, a_batch)
    policy_loss = (-torch.log(pi_a) * advantage.detach()).mean()
    value_loss = (0.5*F.smooth_l1_loss(v_batch, td_target.detach())).mean()
    total_loss = policy_loss + value_loss

    return dict(
        total_loss=total_loss, 
        policy_loss=policy_loss, 
        value_loss=value_loss
        )