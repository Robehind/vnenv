import time
import trainers
import models
import agents
import environment as env
import optimizers
import torch
from tqdm import tqdm
from utils.thordata_utils import get_scene_names, random_divide
import os
from utils.env_wrapper import make_envs, VecEnv
from trainers.loss_functions import a2c_loss
import numpy as np
from utils.mean_calc import ScalarMeanTracker
from utils.model_search import search_newest_model
#TODO 输出loss
def main():
    #读取参数
    from exp_args.a3c_savn_base import args
    args.agent = 'A2CLstmAgent'
    #确认gpu可用情况
    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        #torch.cuda.manual_seed(args.seed)
        assert torch.cuda.is_available()
    
    gpu_id = args.gpu_ids[0]

    #动态载入构造函数
    creator = {
        'model':getattr(models, args.model),
        'agent':getattr(agents, args.agent),
        'env':getattr(env, args.env),
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
    scene_names_div, nums_div = random_divide(args.total_eval_epi, chosen_scene_names, args.threads)
    chosen_objects = []
    for k in args.test_targets.keys():
        chosen_objects = chosen_objects + args.test_targets[k]
    #生成多线程环境，每个线程可以安排不同的房间或者目标
    agent = creator['agent'](
        list(args.action_dict.keys()),
        model,
        args.threads,
        gpu_id
    )
    if args.verbose:
        print('agent created')

    #生成多线程环境，每个线程可以安排不同的房间或者目标
    env_fns = []
    for i in range(args.threads):
        env_args = dict(
            offline_data_dir = args.offline_data_dir,
            action_dict = args.action_dict,
            target_dict = args.target_dict,
            obs_dict = args.obs_dict,
            reward_dict = args.reward_dict,
            max_steps = args.max_epi_length,
            grid_size = args.grid_size,
            rotate_angle = args.rotate_angle,
            chosen_scenes = scene_names_div[i],
            chosen_targets = chosen_objects
        )
        env_fns.append(make_envs(env_args, creator['env']))
    envs = VecEnv(env_fns, eval_mode = True)


    n_epis_thread = [0 for _ in range(args.threads)]
    chosen_scene_names = [k for i in chosen_scene_names for k in i]
    n_epis = {k:0 for k in chosen_scene_names}
    eplen = [0 for _ in range(args.threads)]
    n_eplen = {k:0 for k in chosen_scene_names}
    total_reward = {k:0 for k in chosen_scene_names}
    success_num = {k:0 for k in chosen_scene_names}
    spl_sum = {k:0 for k in chosen_scene_names}

    pbar = tqdm(total=args.total_eval_epi)
    obs = envs.reset()
    while 1:
        agent.clear_mems()
        action, _ = agent.action(obs)
        obs_new, r, done, info = envs.step(action)
        obs = obs_new
        stop = True
        for i in range(args.threads):
            if n_epis_thread[i] < nums_div[i]:
                n_epis_thread[i] += done[i]
                pbar.update(done[i])
                n_epis[info[i]['scene_name']] += done[i]
                n_eplen[info[i]['scene_name']] += 1
                if not info[i]['agent_done']: eplen[i] += 1
                if done[i]:
                    if info[i]['success']:
                        assert info[i]['best_len'] <= eplen[i]
                        spl_sum[info[i]['scene_name']] += info[i]['best_len']/eplen[i]
                    eplen[i] = 0
                    agent.reset_hidden(i)
                total_reward[info[i]['scene_name']] += r[i]
                success_num[info[i]['scene_name']] += info[i]['success']
                stop = False
        
        if stop: break
    envs.close()
    pbar.close()
    logger = {}
    for name in chosen_scene_names:
        data = {
            'epis:':float(n_epis[name]),
            'ave ep length:':n_eplen[name]/n_epis[name],
            'SR:':success_num[name]/n_epis[name],
            'SPL:':spl_sum[name]/n_epis[name],
            'ave total reward:':total_reward[name]/n_epis[name]
        }
        logger.update({name:data})
    logger.update(
        {
            'Total':{
                'epis:':float(sum(n_epis.values())),
                'ave ep length:':sum(n_eplen.values())/args.total_eval_epi,
                'SR:':sum(success_num.values())/args.total_eval_epi,
                'SPL:':sum(spl_sum.values())/args.total_eval_epi,
                'ave total reward:':sum(total_reward.values())/args.total_eval_epi
            }
        }
    )

    import json
    with open(args.results_json, "w") as fp:
        json.dump(logger, fp, sort_keys=True, indent=4)
    print(f'Results write into {args.results_json}')

    

if __name__ == "__main__":
    main()

