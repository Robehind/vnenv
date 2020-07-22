import os
os.environ["OMP_NUM_THREADS"] = '1'
from tensorboardX import SummaryWriter
import time
import trainers
import models
import agents
import runners
import environment as env
import optimizers
import torch
from tqdm import tqdm
from utils.thordata_utils import get_scene_names, random_divide
from utils.env_wrapper import make_envs, VecEnv
import numpy as np
def main():
    #读取参数
    from exp_args.a2c_demo_args import args
    #生成实验文件夹
    start_time = time.time()
    time_str = time.strftime(
        "%y%m%d_%H%M%S", time.localtime(start_time)
    )
    args.exp_dir = os.path.join(args.exps_dir, args.exp_name + '_' + time_str)
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    #保存本次实验的参数
    args.save_args(os.path.join(args.exp_dir, 'args.json'))
    #确认gpu可用情况
    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        assert torch.cuda.is_available()
    
    #TODO 在a2c中暂时只调用一块gpu用于训练，多线程训练可能需要调用pytorch本身的api
    gpu_id = args.gpu_ids[0]

    #动态载入构造函数
    creator = {
        'model':getattr(models, args.model),
        'agent':getattr(agents, args.agent),
        'env':getattr(env, args.env),
        'optimizer':getattr(optimizers, args.optimizer),
        'runner':getattr(runners, args.runner)
    }
    trainer = getattr(trainers, args.trainer)
    loss_func = getattr(trainers, args.loss_func)

    #生成全局模型并初始化优化算法
    model = creator['model'](**args.model_args)
    if model is not None:
        #optimizer.share_memory()
        print(model)
    # 读取存档点，读取最新存档模型的参数到shared_model。其余线程会自动使用sync函数来同步
    if args.load_model_dir is not '':
        print("load %s"%args.load_model_dir)
        model.load_state_dict(torch.load(args.load_model_dir))

    #这里用于分配各个线程的环境可以加载的场景以及目标
    chosen_scene_names = get_scene_names(args.train_scenes)
    scene_names_div, _ = random_divide(1000, chosen_scene_names, args.threads)
    chosen_objects = args.train_targets

    #初始化各个对象
    optimizer = creator['optimizer'](
        model.parameters(),
        **args.optim_args
    )

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
    envs = VecEnv(env_fns)

    runner = creator['runner'](
        args.nsteps,
        args.threads,
        envs,
        agent
    )

    #初始化TX
    tx_writer = SummaryWriter(log_dir = args.exp_dir)
    #training
    trainer(
        args,
        agent,
        envs,
        runner,
        optimizer,
        loss_func,
        tx_writer
    )
    
if __name__ == "__main__":
    main()

