import torch
import torch.nn.functional as F
import copy
import os
from utils.net_utils import toFloatTensor
from utils.net_utils import gpuify
#让agent可以知道动作的字符串，也许在未来有作用
class A2CLstmAgent:
    """lstm智能体"""
    def __init__(
        self,
        action_str,
        model,
        threads,
        gpu_id = -1,
        hidden_state_sz = 512
    ):
        self.actions = action_str
        self.gpu_id = gpu_id
        self.model = model
        self.done = False
        self.threads = threads
        self.hidden_state_sz = hidden_state_sz
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.model = self.model.cuda()
        ###############################################
        self.hidden_batch = [
            [torch.zeros(threads, self.hidden_state_sz)],
            [torch.zeros(threads, self.hidden_state_sz)]
            ]
        self.probs_batch = [torch.zeros((threads, len(self.actions)))]

    def model_forward(self, obs, batch_opt = False):
        """obs is dict. values of obs must in numpy, and first dim is batch dim"""
        #就算只有一个环境，返回的状态也会是1x2048,不需要unsqueeze
        model_input = obs.copy()#防止obs被改变，因为obs在外部还被保存了一次
        if batch_opt:
            model_input['hidden'] = (
                torch.cat(self.hidden_batch[0][:-1]),
                torch.cat(self.hidden_batch[1][:-1])
                )
            model_input['action_probs'] = torch.cat(self.probs_batch[:-1])
        else:
            model_input['hidden'] = (
                self.hidden_batch[0][-1],
                self.hidden_batch[1][-1]
                )
            model_input['action_probs'] = self.probs_batch[-1]

        for k in model_input:
            model_input[k] = toFloatTensor(model_input[k], self.gpu_id)

        
            #obs[k].squeeze_()
        out = self.model.forward(model_input)
        
        return out

    def action(self, env_state):
        out = self.model_forward(env_state)
        pi, hidden = out['policy'], out['hidden']

        self.hidden_batch[0].append(hidden[0].cpu().detach())
        self.hidden_batch[1].append(hidden[1].cpu().detach())
        #softmax,形成在动作空间上的分布
        prob = F.softmax(pi, dim = 1).cpu()
        self.probs_batch.append(prob.detach())
        #采样
        action_idx = prob.multinomial(1).numpy().squeeze(1)

        #print(action_idx.shape)
        return [self.actions[i] for i in action_idx], action_idx

    def sync_with_shared(self, shared_model):
        """ Sync with the shared model. """
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.model.load_state_dict(shared_model.state_dict())
        else:
            self.model.load_state_dict(shared_model.state_dict())
        pass

    def save_model(self, path_to_save, title):
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        state_to_save = self.model.state_dict()
        import time
        start_time = time.time()
        time_str = time.strftime(
            "%H%M%S", time.localtime(start_time)
        )
        save_path = os.path.join(
            path_to_save,
            "{0}_{1}.dat".format(
                title, time_str
            ),
        )
        torch.save(state_to_save, save_path)

    def reset_hidden(self, thread):
        self.hidden_batch[0][-1][thread] = torch.zeros(1, self.hidden_state_sz)
        self.hidden_batch[1][-1][thread] = torch.zeros(1, self.hidden_state_sz)
        self.probs_batch[-1][thread] = torch.zeros((1, len(self.actions)))
        #self.learned_input = None
    
    def clear_mems(self):
        self.hidden_batch = [
                self.hidden_batch[0][-1:],
                self.hidden_batch[1][-1:],
            ]
        self.probs_batch = self.probs_batch[-1:]
