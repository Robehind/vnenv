import importlib
import os
import json
import random
import time
import copy

from .agent_pose_state import AgentPoseState
from .controller_event import OfflineControllerEvent

class OfflineController:
    """与底层数据文件交互的类.提供数据的最大性能，只考虑数据本身的限制。
        Note:AI2THOR离散环境中，y坐标为‘高’，是不变的。当智能体自身的角度为0度时，
        面向z轴的正方向；90度时，面向x轴正方向。（顺时针旋转角度增加）"""

    def __init__(
        self,
        offline_data_dir="offline_data",
        grid_file_name="grid.json",
        graph_file_name="graph.json",
        metadata_file_name="visible_object_map.json",
        verbose = False,
        grid_size = 0.25,
        rotate_angle = 45,
        move_angle = 45,
        **file_names,
    ):
        #加载不同数据集时，可能发生变化的常量，考虑从数据集里读取出来
        self.scene_name = None 
        self.grid_size = grid_size
        self.fov = 100
        self.rotate_angle = rotate_angle
        self.move_angle = move_angle
        self.horizon_angle = 30
        # Allowed rotations.
        self.rotations = [x*self.rotate_angle for x in range(0,360//self.rotate_angle)]
        # Allowed move angles
        self.move_angles = [x*self.move_angle for x in range(0,360//self.move_angle)]
        # Allowed horizons.
        self.horizons = [0, 30]

        self.all_states = None

        # Allowed y
        self.y = 0.91
        #根据不同的移动可能会导致不同的move_list，z永远取x的循环后两位或者后一位
        self.move_list = [0, 1, 1, 1, 0, -1, -1, -1]
        self.move_list = [x*self.grid_size for x in self.move_list]

        #file loader
        self.offline_data_dir = offline_data_dir

        self.grid_file_name = grid_file_name
        self.graph_file_name = graph_file_name
        self.metadata_file_name = metadata_file_name

        self.h5py = importlib.import_module("h5py")
        self.nx = importlib.import_module("networkx")
        self.json_graph_loader = importlib.import_module("networkx.readwrite")

        self.grid = None
        self.graph = None
        self.metadata = None
        #images 指向上传递的数据的文件数据, 为一个dict，其中可视化的图像的keys必须叫image
        self.images = {k:None for k in file_names.keys()}
        self.file_names = file_names

        self.last_event = None
        self.state = None
        self.last_opt_success = True
        self.last_opt = None
        #讲道理应该用self.各种angle来初始化的

        self.verbose = verbose

    def get_state_from_str(self, pose_str):
        temp = [float(x) for x in pose_str.split("|")]
        temp.insert(1, self.y)
        return AgentPoseState(*temp)

    def reset(self, scene_name=None, agent_state=None):
        #TODO 如果都是none？
        if scene_name is None:
            scene_name = self.scene_name
        if scene_name != self.scene_name:
            self.scene_name = scene_name
            with open(
                os.path.join(
                    self.offline_data_dir, self.scene_name, self.grid_file_name
                ),
                "r",
            ) as f:
                self.grid = json.load(f)
                #把grid的y全保留两位小数
                self.grid = [{'x':x['x'], 'y':round(x['y'], 2), 'z':x['z'] } for x in self.grid]
            with open(
                os.path.join(
                    self.offline_data_dir, self.scene_name, self.graph_file_name
                ),
                "r",
            ) as f:
                graph_json = json.load(f)
            self.graph = self.json_graph_loader.node_link_graph(graph_json).to_directed()
            self.all_states = list(self.graph.nodes())
            with open(
                os.path.join(
                    self.offline_data_dir, self.scene_name, self.metadata_file_name
                ),
                "r",
            ) as f:
                self.metadata = json.load(f)

            #读h5py数据.没有读到的报错还没写
            for type_, image_ in self.images.items():
                if image_ is not None:
                    self.images[type_].close()
                self.images[type_] = self.h5py.File(
                    os.path.join(
                        self.offline_data_dir, self.scene_name, self.file_names[type_]
                    ),
                    "r",
                )
        #Initialize
        if agent_state == None:
            #random start pose
            self.set_random_state()
        else:
            self.set_state(agent_state)
            if not self.last_opt_success:
                raise Exception('Unreachable pose in reset.')

        self.last_opt_success = True
        self.last_opt = 'reset'
        self.last_event = self.get_event()
        return self.last_event
        
    def set_random_state(self, ban_state = None):
        lenth = len(self.grid)
        while 1:
            self.state = AgentPoseState(
                **self.grid[random.choice(range(lenth))], 
                #rotation=random.choice(self.rotations),
                rotation = 0,#这里主要考虑到可能初始化到一个“斜的角度”给只能转90度的智能体，以后可以考虑解决
                horizon = 0
            )
            if ban_state == None or str(self.state) not in ban_state: break

        self.y = self.state.y

        self.last_opt_success = True
        self.last_opt = 'set_state'
        self.last_event = self.get_event()
        return self.last_event

    def set_rotation(self, angle):
        angle = (angle + 360) % 360
        self.last_opt = 'set_rotation'
        if angle % self.rotate_angle != 0:
            raise Exception("Unsupported rotate angle %s"%angle)
        if angle in self.rotations:
            self.state.rotation = angle
            self.last_opt_success = True
        else:
            if self.verbose: print("Out of range. False Opt")
            self.last_opt_success = False
        self.last_event = self.get_event()
        return self.last_event

    def set_horizon(self, angle):
        angle = (angle + 360) % 360
        self.last_opt = 'set_horizon'
        if angle % self.horizon_angle != 0:
            raise Exception("Unsupported horizon angle %s"%angle)
        if angle in self.horizons:
            self.state.horizon = angle
            self.last_opt_success = True
        else:
            if self.verbose: print("Out of range. False action")
            self.last_opt_success = False
        self.last_event = self.get_event()
        return self.last_event

    def set_position(self, x, y, z):
        self.last_opt = 'set_position'
        if {'x':x, 'y':y, 'z':z} in self.grid:
            self.state.x = x
            self.state.y = y
            self.state.z = z
            self.y = y
            self.last_opt_success = True
        else:
            if self.verbose: print('Unreachable position')
            self.last_opt_success = False
        self.last_event = self.get_event()
        return self.last_event

    def set_state(self, state):
        self.last_opt = 'set_state'
        if str(state) in self.all_states:
            self.last_opt_success = True
            self.state = state
            self.last_event = self.get_event()
            return self.last_event
        if self.verbose: print('set state false')
        self.last_opt_success = False
        self.last_event = self.get_event()
        return self.last_event
        
    def rotate(self, angle):
        """Rotate a angle. Positive angle for turn right. Negative angle for turn left.
           Will be complished in one step."""
        return self.set_rotation(angle + self.state.rotation)
    
    def look_up_down(self, angle):
        """Look up or down by an angle. Positive angle for look down. Negative angle for look up.
           Will be complished in one step."""
        return self.set_horizon(angle + self.state.horizon)

    def move(self, angle, step_len = None):
        """Move towards any supported directions"""
        angle = (angle + 360) % 360
        self.last_opt = 'move'
        if not angle in self.move_angles:
            raise Exception("Unsupported move angle %s"%angle)
        abs_move_angle = (angle+self.state.rotation)%360
        next_state = copy.deepcopy(self.state)
        #因为定死了movelist，所以这里的moveangle必须是45才行
        index = abs_move_angle//45#self.move_angle
        next_state.x += self.move_list[index%8]
        next_state.z += self.move_list[(index+2)%8]
        #当需要全方向移动时，graph里并没有保存这些转换关系
        #但是如果查询一下面向这个方向时是否可以移动过去，那就说明可以转换
        temp_rotation = self.state.rotation
        next_state.rotation = abs_move_angle
        self.state.rotation = abs_move_angle
        next_state_key = str(next_state)
        neighbors = self.graph.neighbors(str(self.state))
        self.state.rotation = temp_rotation
        if next_state_key in neighbors:
            self.state = next_state
            self.state.rotation = temp_rotation
            self.last_opt_success = True
            self.last_event = self.get_event()
            return self.last_event
        if self.verbose: print("move false")
        self.last_opt_success = False
        self.last_event =self.get_event()
        return self.last_event

    def action_interpret(self, act_str = None):
        """翻译一个动作序列。"""
        #空操作视为一个正常的操作
        temp_state = copy.deepcopy(self.state)
        self.last_opt_success = True
        if not act_str == None:
            for str_ in act_str:
                if str_[0] == 'm':
                    self.move(int(str_[1:]))
                elif str_[0] == 'r':
                    self.rotate(int(str_[1:]))
                elif str_[0] == 'p':
                    self.look_up_down(int(str_[1:]))
                else:
                    raise Exception('Unsupported action %s'%str_)
                if self.last_opt_success == False:
                    self.state = temp_state
                    self.last_event = self.get_event()
                    return self.last_event

        self.last_opt = 'action'
        self.last_event = self.get_event()
        return self.last_event

    def made_a_move(self, state_str1, state_str2):
        """根据两个字符串形式的智能体位置状态来判断是否移动了"""
        pos1 = [x for x in state_str1.split("|")]
        pos2 = [x for x in state_str2.split("|")]
        if pos1[0] != pos2[0] or pos1[1] != pos2[1]:
            return True
        return False

    def shortest_path(self, source_state, target_state):
        '''这边返回的是一系列str(state)'''
        return self.nx.shortest_path(self.graph, str(source_state), str(target_state))
    
    def closest_obj_with_id(self, start_state, obj):

        path_len = None
        cloest_objID = None
        state = start_state

        for k in self.all_objects_with_id():
            if k.split("|")[0] == obj:
                _ , xx = self.shortest_path_to_target(str(state), k)
                if path_len == None or xx < path_len:
                    cloest_objID = k
                    path_len = xx
        
        return cloest_objID, path_len

    def states_where_visible(self, obj):
        if obj == None:
            return []
        tmp = []
        for k in self.all_objects_with_id():
            if k.split("|")[0] == obj:
                tmp.extend(self.metadata[k])
        return tmp
    
    def shortest_path_to_target(self, source_state, objId):
        """ 最短路定义为使用当前动作集能够做到的最短动作数 
        遗留问题在于无法斜向运动的智能体永远无法走出该函数定义的最短路径，后续修复"""

        states_where_visible = []
        states_where_visible = self.metadata[objId]

        # transform from strings into states
        states_where_visible = [
            self.get_state_from_str(str_)
            for str_ in states_where_visible
        ]
        best_path = None
        best_path_len = 0
        for t in states_where_visible:
            path = self.shortest_path(source_state, t)
            path_len = len(path)-1

            if path_len < best_path_len or best_path is None:
                best_path = path
                best_path_len = path_len

        return best_path, best_path_len

    def object_is_visible_with_ID(self, objId):
        return str(self.state) in self.metadata[objId]

    def object_is_visible(self, obj):
        objs = self.all_objects_with_id()
        for obj_ in objs:
            if obj == obj_.split("|")[0]:
                if str(self.state) in self.metadata[obj_]:
                    return True
        return False

    def get_image(self):
        """返回数据。可能有多个，以初始化controller时的关键字为索引"""
        return {k:v[str(self.state)][:] for k,v in self.images.items()}

    def get_data_of_obj(self, obj, file_name, target_json='target.json'):
        '''获取一个目标的相关数据。通过检索能‘看见’这个目标的位置，来透过这个位置读取
        hdf5文件，获取这个位置的‘状态’来作为目标的表示。用file-name来指定是哪种‘状态’'''
        all_objID = [k for k in self.all_objects_with_id() if k.split("|")[0] == obj]
        objID = random.choice(all_objID)
        tmp_loader = self.h5py.File(
                os.path.join(
                    self.offline_data_dir, self.scene_name, file_name
                ),
                "r",
            )
        with open(
            os.path.join(
                self.offline_data_dir, self.scene_name, target_json
                ),"r",
            ) as f:
            state_str = json.load(f)
        data = tmp_loader[state_str[objID]][:]
        tmp_loader.close()
        return data

    def all_objects(self):
        return [obj.split("|")[0] for obj in self.metadata.keys()]

    def all_objects_with_id(self):
        return self.metadata.keys()

    def get_event(self):
        return OfflineControllerEvent(
            self.last_opt, self.last_opt_success, self.scene_name, self.state, self.get_image()
        )
