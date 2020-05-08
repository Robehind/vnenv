"""To generate an Episode in a single thread"""
from environment.discrete_env import DiscreteEnvironment
import random
import torch
import copy
import time
#最大步数在这里设置.完成一个单个的episode能完成的所有事情
# 定义好了之后就可以不断通过外部参数重置开始
class BasicEpisode:
    """用于管理agent和env的交互和重置，只记录计算相关数据，不进行loss计算和反向传播"""
    def __init__(
        self,
        agent,
        env,
        chosen_scene_names,
        chosen_objects = None,
        max_epi_lengh = 100,
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
        self.chosen_objects = chosen_objects
        self.chosen_scene_names = chosen_scene_names
        self.max_length = max_epi_lengh
        self.length = 0
        self.last_state = None
        self.move_length = 0
        self.total_reward = 0
        self.scene_idxs = range(0, len(self.chosen_scene_names))
        random.shuffle(self.scene_idxs)
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
        if scene_name == None:
            if scene_type_idx == None:
                scene_name = random.choice(random.choice(self.chosen_scene_names))
            else:
                idx = self.scene_idxs[scene_type_idx]
                scene_name = random.choice(self.chosen_scene_names[idx])
        pose = None if init_pose == None else init_pose
        #必须先重置环境才知道这个环境到底支持去找哪些物品
        self.env.reset(scene_name, None, pose)
        if target == None:
            objects = list(set(self.env.all_objects).intersection(set(self.chosen_objects)))
            target = random.choice(objects)
        if self.verbose:
            print("In scene %s heading towards %s"%(scene_name, target))
        reper = self.env.set_target(target)
        self.agent.reset(reper)
        if self.visualize: 
            self.env.render() 
            time.sleep(0.3)

    def step(self):
        if self.done:
            raise Exception('episode step while done')
        action = self.agent.action(self.env.state)
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
        
        if self.length >= self.max_length:
            self.done = True
            if self.verbose:
                print('Done by reaching max epi length')
    
    def get_nstep_exps(self, num_steps):
        '''产生n步交互数据。'''
        for _ in range(num_steps):
            self.step()
            if self.done:
                break
        #涉及传参的问题
        _ , R = self.agent.get_pi_v(self.env.state)
        #这里的输出是保留计算图的变量
        pi_batch, v_batch = self.agent.get_pi_v(self.agent.exps['states'])
        return self.agent.exps, pi_batch, v_batch, R.cpu().item()

    def compute_spl(self):
        if self.success:
            assert self.move_length >= self.env.best_path_len(), (
                self.move_length, self.env.best_path_len()
                )
            if self.env.best_path_len() == 0:
                print("Warning: The best path len goes to 0")
                if self.move_length == 0:
                    return 1
                else:
                    #一种暂时的处理
                    return 1/float(self.move_length)
            return float(self.env.best_path_len())/float(self.move_length)
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
