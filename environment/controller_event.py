import copy
class OfflineControllerEvent:
    '''controller的每一个动作都会返回一个event'''
    def __init__(self, last_opt, last_opt_success, scene_name, state=None, data=None):
        self.metadata = {
            "lastOptSuccess": last_opt_success,
            "sceneName": scene_name,
            'lastOpt':last_opt,
        }
        self.agent_state = copy.deepcopy(state)
        self.data = copy.deepcopy(data)