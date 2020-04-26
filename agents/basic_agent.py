import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy

from utils.net_utils import toFloatTensor
#用来计算loss的前几步的value等等信息由agent自己把握
#定义成数据生产者，不执行损失函数计算
class BasicAgent:
    """实现一个基本导航智能体的相关功能，并记录仅智能体可以得到的数据到exps变量中。"""
    def __init__(
        self,
        action_str,
        state_types,
        model,
        gpu_id = -1
    ):
        self.actions = action_str
        self.state_types = state_types
        self.gpu_id = gpu_id
        self.model = model
        self.done = False#智能体是否提出done？当动作中不含'Done'时，一定一直为False
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.model = self.model.cuda()
        self.target_reper  = None#静态的目标的表示，在reset智能体的时候就确定了
        #agent来记state好了，储存预处理后的state。反正后面batch_state都是往model喂的
        self.exps = {
            'action_idxs':[],
            'states':{k:[] for k in state_types},
            'rewards':[]
        }

    def reset(self, target_reper):
        target_reper = {k:v.squeeze() for k,v in target_reper.items()}
        self.target_reper  = target_reper
        self.done = False
        #清空队列
        self.clear_exps()

    def get_pi_v(self, env_state):

        tmp = env_state[list(env_state)[0]]
        target_reper = copy.deepcopy(self.target_reper)
        num_s = 1
        if isinstance(tmp, list):
            num_s = len(tmp)
            target_reper = {
                k:np.expand_dims(v,0).repeat(num_s, 0) for k,v in self.target_reper.items()
                }
        model_input = {}
        model_input.update(env_state)
        model_input.update(target_reper)
        for k in model_input:
            model_input[k] = toFloatTensor(model_input[k], self.gpu_id)
            model_input[k].squeeze_()
            if num_s == 1:
                model_input[k].unsqueeze_(0)
        out = self.model.forward(model_input)
        
        return out['policy'], out['value']

    def get_reward(self, reward):
        self.exps['rewards'].append(reward)

    def action(self, env_state):
        for k in env_state.keys():
            if k not in self.state_types:
                raise Exception('%s can\'t process state type %s'%(self.__class__.__name__, k))
        
        pi, _ = self.get_pi_v(env_state)
        #softmax,形成在动作空间上的分布
        prob = F.softmax(pi, dim=1)
        #采样
        action_idx = prob.multinomial(1).item()
        #记录
        self.exps['action_idxs'].append([action_idx])
        for k in self.state_types:
            self.exps['states'][k].append(env_state[k].copy())
        
        if self.actions[action_idx] == 'Done':
            self.done = True
        return self.actions[action_idx]

    def clear_exps(self):
        self.exps = {
            'action_idxs':[],
            'states':{k:[] for k in self.state_types},
            'rewards':[]
        }

    def agent_exps(self):
        return self.exps

    def sync_with_shared(self, shared_model):
        """ Sync with the shared model. """
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.model.load_state_dict(shared_model.state_dict())
        else:
            self.model.load_state_dict(shared_model.state_dict())
        pass
