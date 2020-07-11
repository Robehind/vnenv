import cv2
import time
import trainers
import models
import agents
import environment as ENV
import optimizers
import torch
from tqdm import tqdm
from utils.thordata_utils import get_scene_names, random_divide
import os
from trainers.loss_functions import a2c_loss
import numpy as np
from utils.mean_calc import ScalarMeanTracker
from utils.model_search import search_newest_model
from utils.env_wrapper import SingleEnv
#TODO 输出loss
def main():
    #读取参数
    from exp_args.a2c_demo_args import args
    #args.exp_name = '123'
    args.obs_dict.update(dict(image='images.hdf5'))
    #print(args.obs_dict)
    args.agent = 'A3CAgent'
    args.threads = 1
    args.gpu_ids = -1
    gpu_id = -1
    #动态载入构造函数
    creator = {
        'model':getattr(models, args.model),
        'agent':getattr(agents, args.agent),
        'env':getattr(ENV, args.env),
    }
    #生成全局模型并初始化优化算法
    model = creator['model'](**args.model_args)
    if model is not None:
        print(model)
    # 读取存档点，读取最新存档模型的参数到shared_model。其余线程会自动使用sync函数来同步
    if args.load_model_dir is not '':
        print("load %s"%args.load_model_dir)
        model.load_state_dict(torch.load(args.load_model_dir))
    else:
        find_path = search_newest_model(args.exps_dir, args.exp_name)
        if find_path is not None:
            print("Searched the neweset model: %s"%find_path)
            model.load_state_dict(torch.load(find_path))
        else:
            print("Can't find a neweset model. Load Nothing.")

    #这里用于分配各个线程的环境可以加载的场景以及目标
    chosen_scene_names = get_scene_names(args.test_scenes)
    scene_names_div, _ = random_divide(args.total_eval_epi, chosen_scene_names, args.threads)
    chosen_objects = args.test_targets
    #生成多线程环境，每个线程可以安排不同的房间或者目标
    agent = creator['agent'](
        list(args.action_dict.keys()),
        model,
        gpu_id
    )
    if args.verbose:
        print('agent created')

    env = creator['env'](
            offline_data_dir = args.offline_data_dir,
            action_dict = args.action_dict,
            target_dict = args.target_dict,
            obs_dict = args.obs_dict,
            reward_dict = args.reward_dict,
            max_steps = args.max_epi_length,
            grid_size = args.grid_size,
            rotate_angle = args.rotate_angle,
            chosen_scenes = scene_names_div[0],
            chosen_targets = chosen_objects
        )
    env = SingleEnv(env, True)
    eplen = 0
    ep_r = 0

    obs = env.reset()
    print(f"heading {env.env.target_str}")
    while 1:
        
        pic = obs['image'][:]
        #RGB to BGR
        pic = pic[:,:,::-1]
        cv2.imshow("Env", pic)
        p_key = cv2.waitKey(0)
        if p_key == 27:
            break
        action, _ = agent.action(obs)
        print(action)
        obs_new, r, done, info = env.step(action)
        obs = obs_new
        ep_r += r
        if not info['agent_done']: eplen += 1
        if done:
            print({
                'eplen':eplen,
                'success':info['success'],
                'spl':info['success']*info['best_len']/eplen,
                'reward':ep_r
            })
            eplen = 0
            ep_r = 0
            pic = obs['image'][:]
            #RGB to BGR
            pic = pic[:,:,::-1]
            cv2.imshow("Env", pic)
            p_key = cv2.waitKey(0)
            obs = env.reset()
            print(f"Heading {env.env.target_str}")

if __name__ == "__main__":
    main()

