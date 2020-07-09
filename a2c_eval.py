import time
import trainers
import models
import agents
import environment as env
import optimizers
import torch
from tqdm import tqdm
from utils.thordata_utils import get_scene_names, random_divide, get_type
import os
from utils.env_wrapper import make_envs, VecEnv
from trainers.loss_functions import a2c_loss
import numpy as np
from utils.mean_calc import ScalarMeanTracker
from utils.model_search import search_newest_model
#TODO 输出loss
def main():
    #读取参数
    from exp_args.a2c_lite_args import args
    args.agent = 'A2CAgent'#TODO

    #生成测试文件夹
    start_time = time.time()
    time_str = time.strftime(
        "%y%m%d_%H%M%S", time.localtime(start_time)
    )
    args.exp_dir = os.path.join(args.exps_dir, 'TEST'+args.exp_name + '_' + time_str)
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    # 通过指定路径寻找参数或者自动寻找最新模型
    if args.load_model_dir is not '':
        print("load %s"%args.load_model_dir)
    else:
        find_path = search_newest_model(args.exps_dir, args.exp_name)
        if find_path is not None:
            print("Searched the neweset model: %s"%find_path)
            args.load_model_dir = find_path
        else:
            print("Can't find a neweset model. Load Nothing.")

    #保存本次测试的参数
    args.save_args(os.path.join(args.exp_dir, 'args.json'))

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
    #生成全局模型并加载参数
    model = creator['model'](**args.model_args)
    if model is not None:
        print(model)
    if args.load_model_dir is not '':
        model.load_state_dict(torch.load(args.load_model_dir))

    #这里用于分配各个线程的环境可以加载的场景以及目标
    chosen_scene_names = get_scene_names(args.test_scenes)
    scene_names_div, nums_div = random_divide(args.total_eval_epi, chosen_scene_names, args.threads)
    chosen_objects = args.test_targets
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
    thread_steps = [0 for _ in range(args.threads)]
    thread_reward = [0 for _ in range(args.threads)]

    all_scenes = [x for i in chosen_scene_names.values() for x in i]
    all_targets = [
        y+'/'+x 
        for i in chosen_objects
        for y in chosen_scene_names[i] 
        for x in chosen_objects[i] 
        ]
    test_scalars = {k:ScalarMeanTracker() for k in all_scenes + all_targets}

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
                stop = False
                t_info = info[i]
                thread_reward[i] += r[i]
                thread_steps[i] += not t_info['agent_done']
                if done[i]:
                    n_epis_thread[i] += 1
                    pbar.update(1)
                    spl = 0
                    if t_info['success']:
                        assert t_info['best_len'] <= thread_steps[i]
                        spl = t_info['best_len']/thread_steps[i]
                    data = {
                        'ep_length:':thread_steps[i],
                        'SR:':t_info['success'],
                        'SPL:':spl,
                        'total_reward:':thread_reward[i]
                    }
                    target_str = t_info['scene_name']+'/'+t_info['target']
                    res = {
                        t_info['scene_name']:data,
                        target_str:data
                    }
                    for k in res:
                        test_scalars[k].add_scalars(res[k])
                        test_scalars[k].add_scalars(dict(epis=1), False)
                    thread_steps[i] = 0
                    thread_reward[i] = 0
                    agent.reset_hidden(i)
        
        if stop: break
    envs.close()
    pbar.close()
    
    total_scalars = {k:ScalarMeanTracker() for k in chosen_scene_names}
    scene_split = {k:{} for k in chosen_scene_names}
    target_split = {
        k:{i:ScalarMeanTracker() for i in chosen_objects[k]} 
        for k in chosen_scene_names
        }
    result = {k:v.pop_and_reset() for k,v in test_scalars.items()}

    for k in result:
        k_sp = k.split('/')
        s_type = get_type(k_sp[0])
        if result[k] == {}:
            continue
        if len(k_sp) == 1:
            scene_split[s_type][k] = result[k].copy()
            epis = result[k].pop('epis')
            total_scalars[s_type].add_scalars(result[k])
            total_scalars[s_type].add_scalars(dict(epis=epis), False)
        else:
            epis = result[k].pop('epis')
            target_split[s_type][k_sp[-1]].add_scalars(result[k])
            target_split[s_type][k_sp[-1]].add_scalars(dict(epis=epis), False)
    
    for k in target_split:
        for t in chosen_objects[k]:
            target_split[k][t] = target_split[k][t].pop_and_reset()
        total_scalars[k] = total_scalars[k].pop_and_reset()
    
    import json
    for k in scene_split:
        tmp = dict(Total = total_scalars[k].copy())
        scene_split[k].update(target_split[k])
        tmp.update(scene_split[k])
        result_path = os.path.join(args.exp_dir, k+'_'+args.results_json)
        with open(result_path, "w") as fp:
            json.dump(tmp, fp, indent=4)

    all_objs = list(set([x for i in chosen_objects.values() for x in i]))
    all_objs.sort()
    total_t = {x:ScalarMeanTracker() for x in all_objs} 
    total_s = ScalarMeanTracker()

    for k in target_split:
        for t in target_split[k]:
            if target_split[k][t] == {}:
                continue
            epis = target_split[k][t].pop('epis')
            total_t[t].add_scalars(target_split[k][t])
            total_t[t].add_scalars(dict(epis=epis), False)
        epis = total_scalars[k].pop('epis')    
        total_s.add_scalars(total_scalars[k])
        total_s.add_scalars(dict(epis=epis), False)
    total_s = dict(Total = total_s.pop_and_reset())
    for k in total_t:
        total_t[k] = total_t[k].pop_and_reset()
    total_s.update(total_t)
    result_path = os.path.join(args.exp_dir, 'Total_'+args.results_json)
    with open(result_path, "w") as fp:
        json.dump(total_s, fp, indent=4)

if __name__ == "__main__":
    main()

