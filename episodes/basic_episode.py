"""To generate an Episode in a single thread"""
from environment.discrete_env import DiscreteEnvironment
import random
import torch
import copy
import time
class BasicEpisode:
    """用于管理agent和env的交互和重置，只记录计算相关数据，不进行loss计算和反向传播"""
    def __init__(
        self,
        agent,
        env,
        verbose = False,
        visualize = False,
    ):
        if list(env.actions) != list(agent.actions):
            raise Exception("Actions not matched")
        self.env = env
        self.verbose = verbose
        self.visualize = visualize
        self.agent = agent
        self.done = False
        self.state = None
        self.length = 0
        self.move_length = 0
        self.total_reward = 0
        #包含一系列的total参数用于记录
        self.infos = []
        self.success = False

    def new_episode(
        self,
        scene_type_idx = None,
        scene_name = None,#if None, choose randomly in all chosen scenes
        target = None, #if None, choose randomly in all_objects and chosen objects
        init_pose = None,
    ):
        
        _, target_reper = self.env.reset(scene_name, target, init_pose)
        self.agent.reset(target_reper)
        if self.visualize: 
            self.env.render() 
            time.sleep(0.3)

    def step(self):
        if self.done:
            raise Exception('episode step while done')
        action = self.agent.action(self.env.get_obs())
        if self.verbose:
            print(action)
        
        _, reward, self.done, info = self.env.step(action)
        if self.visualize: 
            self.env.render()
            time.sleep(0.3)
            print(action)
        self.agent.get_reward(reward)
        self.total_reward += reward
        self.infos.append(info)
        if self.done:
            self.success = info['success']
        self.length += 1
        #if info['moved']:
        if action is not 'Done':
            self.move_length += 1
        
    
    def get_nstep_exps(self, num_steps):
        '''产生n步交互数据。'''
        for _ in range(num_steps):
            self.step()
            if self.done:
                break
        #涉及传参的问题
        _ , R = self.agent.get_pi_v(self.env.get_obs())
        #这里的输出是保留计算图的变量
        pi_batch, v_batch = self.agent.get_pi_v(self.agent.exps['states'])
        return self.agent.exps, pi_batch, v_batch, R.cpu().item()

    def compute_spl(self):
        _, best_len = self.env.best_path_len()
        if self.success:
            #TODO 这里是没办法了，以后修正了数据集再来改吧
            if self.move_length < best_len:
                return 1.
            if best_len == 0:
                print("Warning: The best path len goes to 0")
                if self.move_length == 0:
                    return 1
                else:
                    #一种暂时的处理
                    return 1/float(self.move_length)
            return float(best_len)/float(self.move_length)
        return 0

    def clear_exps(self):
        self.agent.clear_exps()
        
    def end_episode(self):
        self.total_reward = 0
        self.done = False
        self.success = False
        self.length = 0
        self.move_length = 0
        self.agent.clear_exps()
        self.infos = []
