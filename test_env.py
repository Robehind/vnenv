from environment.discrete_env import DiscreteEnvironment
import cv2
params = {
    "offline_data_dir" : '..\\thordata\\mixed_offline_data',
    "action_dict":{
        "MoveAhead":["m0"],
        #"BackOff":["m180"],
        "TurnLeft":["r-90"],
        "TurnRight":["r90"],
        #"Done":None
    },
    "target_dict":{
        "glove":"..\\thordata\\word_embedding\\thor_glove\\glove_map300d.hdf5",
        #"image":"xxxx.hdf5"
    },
    "state_dict":{
        "RGB":"images.hdf5",
        #"id":"ids.hdf5"
        #"depth":"depth.hdf5",
        #"featmap":"resnet18_featuremap.hdf5",
        "fc":"resnet50_fc.hdf5"
    },
    "reward_dict":{
        "collision":-0.1,
        "step":-0.01,
        "SuccessDone":10.0,
        "FalseDone":0,
    },
    'grid_size' : 0.5,
    'rotate_angle' : 90,
    'move_angle' : 90,
    "chosen_objects" : None,
    "debug" : False,
}
env = DiscreteEnvironment(**params)
scene_name = 'bathroom_02'
state, reper, info = env.reset(scene_name)
print(env.possible_objects())
t = input('Choose a target:')
t = None if t == '' else t
env.set_target(t)
press_key = None

#数字都是键码
action_dict = {
    119:'MoveAhead', 97:'TurnLeft', 
    100:'TurnRight', 115:'Done',
    105:'LookUp', 107:'LookDown', 120:'BackOff'
    }
reward_sum = 0

while True:
    pic = state['RGB'][:]
    fc = state['fc'][:]
    #RGB to BGR
    pic = pic[:,:,::-1]
    print(fc)
    cv2.imshow("Env", pic)
    press_key = cv2.waitKey(0)
    if press_key in action_dict.keys():
        state, reward, done, info = env.step(action_dict[press_key])
        print('Instant Reward:',reward)
        reward_sum += reward
        print('Total Reward:',reward_sum)
        if done:
            pic = state['RGB'][:,:,::-1]
            reward_sum = 0
            print('Env is done. Press anykey to Reset.')
            cv2.imshow("Env", pic)
            cv2.waitKey(0)
            print(env.possible_objects())
            t = input('Choose a target:')
            t = None if t == '' else t
            state, reper, info = env.reset(scene_name, t)
    elif press_key == 27:
        break
    else:
        print('Unsupported action')