import cv2
import copy
import numpy as np
import random
import h5py
import os
import json
import importlib
from .agent_pose_state import get_state_from_str
#改成私有变量
class DiscreteEnvironment:
    """读取数据集，模拟交互，按照dict的组织和标识来返回数据和信息。
       所有数据都是用np封装的
       特点在于是通过动作的字符串来交互的。"""
    grid_file_name = "grid.json"
    graph_file_name = "graph.json"
    visible_file_name = "visible_object_map.json"
    def __init__(
        self,
        offline_data_dir = '../thordata/mixed_offline_data',#包含所有房间文件夹的路径
        action_dict = {
            'MoveAhead':['m0'],
            'TurnLeft':['r-45'],
            'TurnRight':['r45'],
            'LookUp':['p-30'],
            'LookDown':['p30'],
            'Done':None,#Done动作一定必须绑定为None
            #包含Done字符串时，需要智能体自主提出结束episode，不包含时环境会自动判定是否结束
        },
        target_dict = {
            'image':'images.hdf5',
            'fc':'resnet50_fc.hdf5',
            'glove':'../thordata/word_embedding/word_embedding.hdf5',
        },
        obs_dict = {
            'image':'images.hdf5',
        },
        reward_dict = {
            'collision':-0.1,
            'step':-0.01,
            'SuccessDone':10,
            'FalseDone':0,
        },
        max_steps = 100,
        grid_size = 0.25,
        rotate_angle = 45,
        move_angle = 45,
        horizon_angle = 30,
        chosen_scenes = ['FloorPlan1_physics'],#scene names random from
        chosen_targets = None,#默认值为None时则无限制,这个表是针对env目前能加载的所有scene而言的
        debug = False,
    ):
        self.actions = list(action_dict.keys())
        self.action_dict = action_dict
        self.reward_dict = reward_dict
        self.offline_data_dir = offline_data_dir
        self.max_steps = max_steps

        #file loader
        self.nx = importlib.import_module("networkx")
        self.json_graph_loader = importlib.import_module("networkx.readwrite")
        
        self.debug = debug
        self.chosen_targets = chosen_targets
        self.chosen_scenes = chosen_scenes
        
        self.grid_size = grid_size
        self.rotate_angle = rotate_angle
        self.move_angle = move_angle
        self.horizon_angle = horizon_angle

        #根据不同的绝对移动角度，x和z坐标的变化符合以下表格规律
        self.move_list = [0, 1, 1, 1, 0, -1, -1, -1]
        self.move_list = [x*self.grid_size for x in self.move_list]
        # Allowed rotate angles
        self.rotations = [x*self.rotate_angle for x in range(0,360//self.rotate_angle)]
        # Allowed move angles
        self.move_angles = [x*self.move_angle for x in range(0,360//self.move_angle)]
        # Allowed horizons.
        self.horizons = [0, 30]

        #判断环境是否需要自动为智能体停止模拟
        self.auto_done = True
        if 'Done' in self.actions:
            self.auto_done = False
        
        #可变变量
        self.scene_name = None
        self.done = False
        self.reward = 0
        self.steps = 0

        #self.agent_state的数据格式沿用AgentPoseState
        self.agent_state = None
        self.start_state = None
        self.last_agent_state = None
        self.last_action = None
        self.last_opt = None
        self.last_opt_success = True

        self.obs_dict = obs_dict
        self.obs_loader = {k:None for k in obs_dict}
        
        self.target_str = None
        self.target_type = list(target_dict.keys())
        self.target_dict = target_dict
        self.info = {}

        #Reading and gernerating meta data
        self.all_objects = None #房间支持可以找的所有物体str
        self.all_objects_id = None #房间支持可以找的所有物体以及其坐标，in str
        self.all_agent_states = None #智能体所有的可能的位姿状态，str
        self.all_visible_states = None #智能体在哪些位置可以看到当前目标， in str
        
        #不同的目标表示可能会导致在每次重置环境时读取新的状态表示文件,未来再改善，应该写到reset里
        self.target_reper_info = {}
        self.tLoader = None
        self.loader_support = {} #scene里有的object未必在reper里就有
        for str_ in self.target_type:
            if str_ in ['glove', 'fasttext', 'onehot']:
                self.tLoader = h5py.File(self.target_dict[str_], "r",)
                self.loader_support[str_] = self.tLoader[str_].keys()
                tmp = self.tLoader[str_][list(self.tLoader[str_].keys())[0]][:]
                self.target_reper_info.update({str_:(tmp.shape, tmp.dtype)})
        
        #随机读一个房间的数据，生成状态的信息，在并行化环境的时候用得上
        self.his_states = [] #in str
        self.his_len = 0
        scene_name = random.choice(self.chosen_scenes)
        self.obs_info = {}
        for type_, name_ in self.obs_dict.items():
            loader = h5py.File(
                os.path.join(offline_data_dir, scene_name, name_),"r",
            )
            tmp = loader[list(loader.keys())[0]][:]
            if '|' in type_:
                shape = (int(type_.split('|')[1]), *tmp.shape)
                self.his_len = max(self.his_len, int(type_.split('|')[1]))
            else:
                shape = tmp.shape
            self.obs_info.update({type_:(shape, tmp.dtype)})
            loader.close()

        self.data_info = {}
        self.data_info.update(self.target_reper_info)
        self.data_info.update(self.obs_info)

        #action check
        need_pitch = False
        min_rotate = 999
        for act_str in self.action_dict.values():
            if act_str == None:
                continue
            for str_ in act_str:
                angle = (int(str_[1:]) + 360) % 360
                if str_[0] == 'm':
                    assert angle in self.move_angles
                elif str_[0] == 'r':
                    if angle<min_rotate:
                        min_rotate = angle
                    assert angle % self.rotate_angle == 0 
                elif str_[0] == 'p':
                    need_pitch = True
                    assert angle % self.horizon_angle == 0 
                else:
                    raise Exception('Unsupported action %s'%str_)
        if not need_pitch:
            self.horizons = [0]
        if min_rotate > self.rotate_angle:
            print("Warning: min rotate angle is bigger than rotate_angle")
    
    def close(self):
        pass

    def reset(
        self, 
        scene_name = None, 
        target_str = None, 
        agent_state = None, 
        allow_no_target = False
        ):
        if scene_name == None:
            scene_name = random.choice(self.chosen_scenes)
        assert scene_name in self.chosen_scenes
        #reading metadata and obs data
        if scene_name != self.scene_name:
            self.scene_name = scene_name
            s_path = os.path.join(self.offline_data_dir, self.scene_name)
            with open(
                os.path.join(s_path, self.graph_file_name),"r",
            ) as f:
                graph_json = json.load(f)
            self.graph = self.json_graph_loader.node_link_graph(graph_json).to_directed()
            self.all_agent_states = [
                x for x in list(self.graph.nodes()) 
                if int(x.split('|')[2]) in self.rotations\
                and int(x.split('|')[3]) in self.horizons\
                and float(x.split('|')[0])%self.grid_size==0.0\
                and float(x.split('|')[1])%self.grid_size==0.0\
                ]
            with open(
                os.path.join(s_path, self.visible_file_name),"r",
            ) as f:
                self.visible_data = json.load(f)
            self.all_objects_id = self.visible_data.keys()
            self.all_objects = [x.split('|')[0] for x in self.all_objects_id]
            if self.chosen_targets == None:
                self.chosen_targets = self.all_objects
            #读h5py数据.没有读到的报错还没写
            for type_, image_ in self.obs_loader.items():
                if image_ is not None:
                    self.obs_loader[type_].close()
                self.obs_loader[type_] = h5py.File(
                    os.path.join(s_path, self.obs_dict[type_]),"r",
                )
        #set target
        if target_str == None and not allow_no_target:
            legal_list = list(set(self.all_objects).intersection(set(self.chosen_targets)))
            target_str = random.choice(legal_list)
        self.set_target(target_str)
        self.all_visible_states = self.states_where_visible(self.target_str)

        #Initialize position
        self.set_agent_state(agent_state, self.all_visible_states)
        self.start_state = copy.deepcopy(self.agent_state)
        self.his_states = [self.start_state for _ in range(self.his_len)]

        self.last_action = None
        self.reward = 0
        self.done = False
        self.steps = 0
        self.info = dict(success = False)
        
        return self.get_obs(True),\
               self.get_target_reper(self.target_str)

    def set_agent_state(self, agent_state = None, ban_list = []):
        """设置智能体的位姿。如果agent_state为None，则会随机选择一个不在banlist中的位姿
        """
        if agent_state == None:
            legal_list = list(set(self.all_agent_states).difference(set(ban_list)))
            agent_state = random.choice(legal_list)
        else:
            assert agent_state in self.all_agent_states
            assert agent_state not in ban_list
        if isinstance(agent_state, str):
            agent_state = get_state_from_str(agent_state)
        self.agent_state = agent_state
    
    def set_target(self, target_str):
        """可以在外部调用的设置目标"""
        if target_str == None:
            self.target_str = target_str
            return
        #判断所设置目标的合法性
        if not target_str in self.all_objects:
            raise Exception("No \'%s\' in \'%s\'"%(target_str, self.scene_name))
        for k, v in self.loader_support.items():
            if not target_str in v:
                raise Exception("Object \'%s\' unsupported by \'%s\'"%(target_str,k))
        if not self.chosen_targets == None:
            if not target_str in self.chosen_targets:
                raise Exception("Object \'%s\' is not chosen"%target_str)
        self.target_str = target_str

    def states_where_visible(self, target_reper):
        if target_reper == None:
            return []
        tmp = []
        for k in self.all_objects_id:
            if k.split("|")[0] == target_reper:
                tmp.extend(self.visible_data[k])
        return tmp

    def get_obs(self, init = False):
        """返回obs。可能有多个，以初始化controller时的关键字为索引"""
        tmp = {}
        for k,v in self.obs_loader.items():
            if '|' in k:
                tmp.update({
                    k:np.array(
                        [v[str(s)][:] for s in self.his_states[:int(k.split('|')[1])]]
                        )
                })
            else:
                tmp.update({k:v[str(self.agent_state)][:]})
        return tmp
        
    # def prepro_obs(self, obs, init = False):
    #     """一个特殊的二次封装obs的函数，如果需要对obs尽心进一步预处理，
    #     例如累计4帧图像数据，就写在这里"""
    #     #暂时只支持处理n帧特征向量
    #     for k in obs:
    #         if '|' in k:
    #             if init:
    #                 self.last_obs[k] = np.tile(obs[k].squeeze(), (int(k.split("|")[1])))
    #                 #obs[k] = np.tile(obs[k].squeeze(), (int(k.split("|")[1]))) 
    #             else:
    #                 obs[k] = np.append(self.last_obs[k][np.prod(obs[k].shape):], obs[k])
    #                 self.last_obs[k] = 
    #     return obs

    def rotate(self, angle:int):
        """Rotate a angle. Positive angle for turn right. Negative angle for turn left.
           Will be complished in one step."""
        self.last_opt = 'rotate'
        angle = (angle + self.agent_state.rotation + 360) % 360
        if angle in self.rotations:
            self.agent_state.rotation = angle
            self.last_opt_success = True
        else:
            self.last_opt_success = False

    def look_up_down(self, angle):
        """Look up or down by an angle. Positive angle for look down. Negative angle for look up.
           Will be complished in one step."""
        self.last_opt = 'look_up_down'
        angle = (angle + self.agent_state.horizon + 360) % 360
        if angle in self.horizons:
            self.agent_state.horizon = angle
            self.last_opt_success = True
        else:
            self.last_opt_success = False

    def move(self, angle):
        """Move towards any supported directions"""
        self.last_opt = 'move'
        abs_angle = (angle + self.agent_state.rotation + 360) % 360
        temp_rotation = self.agent_state.rotation
        self.agent_state.rotation = abs_angle
        next_state = copy.deepcopy(self.agent_state)
        #因为定死了movelist，所以这里的moveangle必须是45才行
        index = next_state.rotation//45#self.move_angle
        next_state.x += self.move_list[index%8]
        next_state.z += self.move_list[(index+2)%8]
        #当需要全方向移动时，graph里可能并没有保存这些转换关系
        #但是如果查询一下面向这个方向时是否可以移动过去，那就说明可以转换
        neighbors = self.graph.neighbors(str(self.agent_state))
        self.agent_state.rotation = temp_rotation
        if str(next_state) in neighbors:
            self.agent_state.x, self.agent_state.z = next_state.x, next_state.z
            self.last_opt_success = True
        else:
            self.last_opt_success = False

    def action_interpret(self, act_str):
        """翻译一个动作序列。"""
        #空操作视为一个正常的操作
        temp_state = copy.deepcopy(self.agent_state)
        self.last_opt_success = True
        if not act_str == None:
            for str_ in act_str:
                if str_[0] == 'm':
                    self.move(int(str_[1:]))
                elif str_[0] == 'r':
                    self.rotate(int(str_[1:]))
                elif str_[0] == 'p':
                    self.look_up_down(int(str_[1:]))
                if self.last_opt_success == False:
                    self.agent_state = temp_state
                    break

    def step(self, action):
        if self.target_str == None:
            print("Warning: Didn\'t set target. Check target visibility will always be false")

        if self.done and not self.debug:
            raise Exception('Should not interact with env when env is done')
        if not action in self.actions:
            raise Exception("Unsupport action")

        self.action_interpret(self.action_dict[action])
        self.his_states = [str(self.agent_state)] + self.his_states[1:]
        self.last_agent_state = copy.deepcopy(self.agent_state)
        self.steps += 1
        self.last_action = action
        #分析events，给reward
        event, self.done = self.judge(action)
        self.reward = self.reward_dict[event]
        #可以配置更多的额外环境信息在info里

        return self.get_obs(), self.reward, self.done, self.info

    def judge(self, action):
        #详细的奖惩设置都在这里
        done = False
        event = 'step'
        if not self.last_opt_success:
            if self.last_opt in ['move', 'look_up_down']:
                event = 'collision'
        if self.auto_done:
            if self.target_visiable():
                event = 'SuccessDone'
                done = True
                self.info['success'] = True
        elif action == 'Done':
            event = 'SuccessDone' if self.target_visiable() else 'FalseDone'
            done = True
            self.info['success'] = (event == 'SuccessDone')
        
        if self.steps == self.max_steps:
            done = True
            if event not in ['SuccessDone', 'FalseDone']:
                self.info['success'] = False

        return event, done

    def target_visiable(self):
        if str(self.agent_state) in self.all_visible_states:
            return True
        return False
    
    def get_target_reper(self, target_str):
        
        if target_str == None:
            return None
        reper = {}
        for str_ in self.target_type:
            if str_ in ['glove', 'fasttext', 'onehot']:
                reper[str_] = self.tLoader[str_][target_str][:]
            else:
                reper[str_] = self.get_data_of_obj(target_str, self.target_dict[str_])
        return reper

    def get_data_of_obj(self, target_str, file_name, target_json='target.json'):
        '''获取一个目标的相关数据。通过检索能‘看见’这个目标的位置，来透过这个位置读取
        hdf5文件，获取这个位置的‘状态’来作为目标的表示。用file-name来指定是哪种‘状态’'''
        all_objID = [k for k in self.all_objects_id if k.split("|")[0] == target_str]
        objID = random.choice(all_objID)
        tmp_loader = h5py.File(
            os.path.join(self.offline_data_dir, self.scene_name, file_name),"r",
            )
        with open(
            os.path.join(self.offline_data_dir, self.scene_name, target_json),"r",
            ) as f:
            state_str = json.load(f)
        data = tmp_loader[state_str[objID]][:]
        tmp_loader.close()
        return data

    def best_path_len(self):
        """算最短路，用于计算spl."""
        #如果房间里有复数个目标，还要计算找出离当前位置最近的那一个。。。
        start_state = self.start_state
        best_path_len = 9999
        best_path = None

        for k in self.all_visible_states:
            path = self.nx.shortest_path(self.graph, str(start_state), k)
            path_len = len(path) - 1
            if path_len < best_path_len:
                best_path = path
                best_path_len = path_len
        
        return best_path, best_path_len

    def render(self):
        pic = self.get_obs()['image'][:]
        #RGB to BGR
        pic = pic[:,:,::-1]
        cv2.imshow("Env", pic)
        cv2.waitKey(1)