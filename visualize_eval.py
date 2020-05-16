import time
import json
import testers
import episodes
import models
import agents
import environment #as env
import torch
from tqdm import tqdm
from utils.thordata_utils import get_scene_names, random_divide

def main():
    #读取参数
    from exp_args.demo_args import args

    args.gpu_id = -1
    args.threads = 1
    args.visualize = True,
    args.obs_dict.update(dict(image = 'images.hdf5'))

    #查看载入模型是否存在
    if args.load_model_dir == '':
        print('Warining: load_model_dir didn\'t exist. Testing model with init params')
    
    #动态载入构造函数
    creator = {
        'model':getattr(models, args.model),
        'episode':getattr(episodes, args.episode),
        'agent':getattr(agents, args.agent),
        'env':getattr(environment, args.env),
    }
    chosen_scene_names = []
    tmp = get_scene_names(args.train_scenes)
    for k in tmp:
        chosen_scene_names += k
    chosen_objects = []
    for k in args.test_targets.keys():
        chosen_objects = chosen_objects + args.test_targets[k]

    epi_num = args.total_eval_epi
    gpu_id = -1

    #initialize env and agent

    model = creator['model'](**args.model_args)
    
    #加载模型参数
    if args.load_model_dir != "":
        saved_state = torch.load(
            args.load_model_dir, map_location=lambda storage, loc: storage
        )
        print("load %s"%args.load_model_dir)
        model.load_state_dict(saved_state)

    agent = creator['agent'](
        list(args.action_dict.keys()),
        list(args.obs_dict.keys()),
        model,
        gpu_id
    )
    env = creator['env'](
        args.offline_data_dir,
        args.action_dict,
        args.target_dict,
        args.obs_dict,
        args.reward_dict,
        max_steps = args.max_epi_length,
        grid_size = args.grid_size,
        rotate_angle = args.rotate_angle,
        chosen_scenes = chosen_scene_names,
        chosen_targets = chosen_objects
    )
    #initialize a episode
    epi = creator['episode'](
        agent,
        env,
        verbose = args.verbose,
        visualize = True,
    )
    if args.verbose:
        print('Created all componets')

    count = 0

    while count < epi_num:
        #每一次循环为一次episode
        #episode重置
        target = input("Input a target:")
        epi.new_episode(target = target)#场景会重新随机选，目标也会随机选
        if args.verbose:
            print('new epi created')
            print('best path length: %s'%(env.best_path_len()))
        
        # Train on the new episode.
        while not epi.done:
            
            # Run episode for num_steps or until player is done.
            epi.get_nstep_exps(args.max_epi_length)
            epi.clear_exps()
                #agent.clear_exps()

        count+=1

        results = {
            "agent_done": agent.done,#记录智能体是否提出了done这个动作
            "ep_length": epi.length,
            "success": epi.success,
            "total_reward":epi.total_reward,
            "spl":epi.compute_spl()
        }
        print(results)
        epi.end_episode()


if __name__ == "__main__":
    main()

