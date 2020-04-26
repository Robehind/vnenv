from .offline_controller import OfflineController
import cv2
import copy
#只提供最基础的target和图像预处理后的state的数据，如果需要什么以往动作、速度就得在agent里自己写
#但凡agent可以通过自己的历史记录得到的state，都让agent自己搞定，env不管
class DiscreteEnvironment:
    """读取数据集，模拟交互，按照dict的组织和标识来返回数据和信息。
       特点在于是通过动作的字符串来交互的。"""
    def __init__(
        self,
        offline_data_dir = '../thordata/mixed_offline_data',#包含所有房间文件夹的路径
        action_dict={
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
            'glove':'../thordata/thor_glove/glove_map300d.hdf5',
        },
        state_dict = {
            'image':'images.hdf5',
        },
        reward_dict = {
            'collision':-0.1,
            'step':-0.01,
            'SuccessDone':10,
            'FalseDone':0,
            #'angle_restrict': -0.05
        },
        grid_size = 0.25,
        rotate_angle = 45,
        move_angle = 45,
        chosen_objects = None,#默认值为None时则无限制,这个表是针对env目前能加载的所有scene而言的
        debug = False,
    ):
        self.actions = action_dict.keys()
        self.action_dict = action_dict
        self.reward_dict = reward_dict
        self.scene_name = None
        self.debug = debug
        self.chosen_objects = chosen_objects
        self.all_objects = None
        
        self.controller = OfflineController(offline_data_dir, grid_size = grid_size,
        rotate_angle = rotate_angle,
        move_angle = move_angle,
        **state_dict)
        #判断环境是否需要自动为智能体停止模拟
        self.auto_done = True
        if 'Done' in self.actions:
            self.auto_done = False
        self.done = False

        self.reward = 0
        self.state = None
        #self.agent_state的数据格式？要沿用AgentPoseState吗
        self.agent_state = None
        self.last_agent_state = None
        self.last_action = None
        
        self.target_str = None
        self.target_reper = None
        self.target_type = list(target_dict.keys())
        self.target_dict = target_dict

        self.info = {}

        #不同的目标表示可能会导致在每次重置环境时读取新的状态表示文件,未来再改善，应该写到reset里
        self.tLoader = {}
        self.loader_support = {} #scene里有的object未必在reper里就有
        for str_ in self.target_type:
            if str_ == 'glove':
                import h5py
                self.tLoader[str_] = h5py.File(self.target_dict[str_],"r",)
                self.loader_support[str_] = self.tLoader[str_].keys()
            elif str_ == 'fasttext' or str_ == 'onehot':
                import pickle
                with open(self.target_dict[str_], 'rb') as f:
                    self.tLoader[str_] = pickle.load(f)
                self.loader_support[str_] = self.tLoader[str_].keys()

    def reset(self, scene_name, target_str = None, agent_state = None):
        
        self.scene_name = scene_name
        #如果输入了错误的状态类型？
        if isinstance(agent_state, str):
            agent_state = self.controller.get_state_from_str(agent_state)
        controller_event = self.controller.reset(scene_name, agent_state)
        self.agent_state = controller_event.agent_state
        self.start_state = copy.deepcopy(self.agent_state)
        self.all_objects = self.controller.all_objects()
        self.state = controller_event.data
        #will check if target is legal in get_target_reper()
        self.set_target(target_str)

        self.last_action = None
        self.reward = 0
        self.done = False
        self.info = {}
        return self.state, self.target_reper, self.info

    def step(self, action):
        if self.target_str == None:
            print("Warning: Didn\'t set target. Check target visibility will always be false")

        if self.done and not self.debug:
            raise Exception('Should not interact with env when env is done')
        if not action in self.actions:
            raise Exception("Unsupport action")
        controller_event = self.controller.action_interpret(self.action_dict[action])
        self.last_agent_state = self.agent_state
        self.agent_state = controller_event.agent_state
        self.state = controller_event.data
        self.last_action = action
        #分析events，给reward
        event, self.done = self.judge(action, controller_event.metadata)
        self.reward = self.reward_dict[event]
        #可以配置更多的额外环境信息在info里

        self.info['moved'] = True
        if self.last_agent_state.x == self.agent_state.x:
            if self.last_agent_state.z == self.agent_state.z:
                self.info['moved'] = False
        
        self.info['lastActionSuccess'] = controller_event.metadata['lastOptSuccess']
        self.info['lastEvent'] = event
        return self.state, self.reward, self.done, self.info

    def judge(self, action, controller_event):
        #详细的奖惩设置都在这里
        done = False
        event = 'step'
        if not controller_event['lastOptSuccess']:
            if controller_event['lastOpt'] in ['move', 'action', 'set_horizon']:
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

        return event, done
    
    def target_visiable(self):
        if self.target_str == None:
            return False
        return self.object_is_visiable(self.target_str)

    def object_is_visiable(self, target):
        return self.controller.object_is_visible(target)

    def set_target(self, target_str):
        #it's possible to set no target. Target_str can be None
        self.target_reper = self.get_target_reper(target_str)
        self.target_str = target_str
        return self.target_reper

    def get_target_reper(self, target_str):
        #make sure you've called reset() to update self.all_objects
        if target_str == None:
            return None
        if not target_str in self.all_objects:
            raise Exception("No \'%s\' in \'%s\'"%(target_str, self.scene_name))
        for k, v in self.loader_support.items():
            if not target_str in v:
                raise Exception("Object \'%s\' unsupported by \'%s\'"%(target_str,k))
        if not self.chosen_objects == None:
            if not target_str in self.chosen_objects:
                raise Exception("Object \'%s\' is not chosen"%target_str)
        reper = {}
        for str_ in self.target_type:
            if str_ == 'glove':
                reper[str_] = self.tLoader[str_][target_str][:]
            elif str_ == 'fasttext' or str_ == 'onehot':
                reper[str_] = self.tLoader[str_][target_str][:]
            else:
                reper[str_] = self.controller.get_data_of_obj(target_str, self.target_dict[str_])

        return reper

    def current_target_reper(self):
        return self.target_reper
    
    def possible_objects(self):
        return self.all_objects

    def best_path_len(self):
        """算最短路，用于计算spl."""
        #如果房间里有复数个目标，还要计算找出离当前位置最近的那一个。。。
        _,out = self.controller.closest_obj_with_id(self.start_state, self.target_str)
        return out

    def visualize_plan(self, source, plan):
        """ Visualize the best path from source to plan. """
        pass

    def render(self):
        pic = self.state['image'][:]
        #RGB to BGR
        pic = pic[:,:,::-1]
        cv2.imshow("Env", pic)
        cv2.waitKey(1)