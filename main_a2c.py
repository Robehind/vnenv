from tensorboardX import SummaryWriter
import time
import trainers
import models
import agents
import environment as env
import optimizers
import torch
from tqdm import tqdm
from utils.thordata_utils import get_scene_names
import os
from utils.parallel_env import make_envs, VecEnv
from trainers.loss_functions import a2c_loss
import numpy as np
from utils.mean_calc import ScalarMeanTracker
#TODO 输出loss
def main():
    #读取参数
    from exp_args.a2c_gcn import args
    #生成日志文件
    start_time = time.time()
    local_start_time_str = time.strftime(
        "%Y-%m-%d_%H-%M-%S", time.localtime(start_time)
    )

    if args.log_dir is not None:
        tb_log_dir = args.log_dir + "/" + args.log_title + "-" + local_start_time_str
        log_writer = SummaryWriter(log_dir=tb_log_dir)
    else:
        log_writer = SummaryWriter(comment=args.log_title)

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
        'optimizer':getattr(optimizers, args.optimizer)
    }
    #trainer = getattr(trainers, args.trainer)

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
    #暂时就每个线程都加载一样的场景集合和物体
    chosen_scene_names = []
    tmp = get_scene_names(args.train_scenes)
    for k in tmp:
        chosen_scene_names += k
    chosen_objects = []
    for k in args.train_targets.keys():
        chosen_objects = chosen_objects + args.train_targets[k]
    optimizer = creator['optimizer'](
        model.parameters(),
        **args.optim_args
    )
    #生成多线程环境，每个线程可以安排不同的房间或者目标
    agent = creator['agent'](
        list(args.action_dict.keys()),
        model,
        gpu_id
    )
    if args.verbose:
        print('agent created')

    #生成多线程环境，每个线程可以安排不同的房间或者目标
    env_args = dict(
        offline_data_dir = args.offline_data_dir,
        action_dict = args.action_dict,
        target_dict = args.target_dict,
        obs_dict = args.obs_dict,
        reward_dict = args.reward_dict,
        max_steps = args.max_epi_length,
        grid_size = args.grid_size,
        rotate_angle = args.rotate_angle,
        chosen_scenes = chosen_scene_names,
        chosen_targets = chosen_objects
    )
    env_fns = [make_envs(env_args, creator['env']) for _ in range(args.threads)]
    envs = VecEnv(env_fns)

    n_frames = 0
    total_epis = 0

    n_epis = 0
    total_reward = 0
    success_num = 0

    # exps={
    #     'rewards':[],
    #     'masks':[],
    #     'action_idxs':[]
    # }
    obses = {k:[] for k in envs.keys}
    loss_traker = ScalarMeanTracker()
    pbar = tqdm(total=args.total_train_frames)
    obs = envs.reset()
    while n_frames < args.total_train_frames:
        exps = {
            'rewards':[],
            'masks':[],
            'action_idxs':[]
        }
        for k in obses:
            obses[k] = []
        for _ in range(args.nsteps):
            action, a_idx = agent.action(obs)
            obs_new, r, done, success = envs.step(action)
            for k in obses:
                obses[k].append(obs[k])
            exps['action_idxs'].append(a_idx)
            exps['rewards'].append(r)
            exps['masks'].append(1 - done)
            obs = obs_new
            n_epis += done.sum()
            total_reward += r.sum()
            success_num += success.sum()

        _, v_final = agent.get_pi_v(obs)
        v_final = v_final.detach().cpu().numpy().reshape(-1)
        for k in obses:
            obses[k] = np.array(obses[k]).reshape(-1, *obses[k][0][0].shape)
        pi_batch, v_batch = agent.get_pi_v(obses)
        loss = a2c_loss(v_batch, pi_batch, v_final, exps, gpu_id)
        optimizer.zero_grad()
        loss['total_loss'].backward()
        for k in loss:
            loss_traker.add_scalars({k:loss[k].item()})
        optimizer.step()

        #记录、保存、输出
        pbar.update(args.nsteps * args.threads)
        n_frames += args.nsteps * args.threads
        
        if n_frames % args.print_freq == 0:
            total_epis += n_epis
            log_writer.add_scalar("n_frames", n_frames, total_epis)
            log_writer.add_scalar("epi length", args.print_freq/n_epis, n_frames)
            log_writer.add_scalar("total reward", total_reward/n_epis, n_frames)
            log_writer.add_scalar("success", success_num/n_epis, n_frames)
            for k,v in loss_traker.pop_and_reset().items():
                log_writer.add_scalar(k, v, n_frames)
            n_epis, total_reward, success_num = 0,0,0

        if n_frames % args.model_save_freq == 0:
            if not os.path.exists(args.save_model_dir):
                os.makedirs(args.save_model_dir)
            state_to_save = model.state_dict()
            save_path = os.path.join(
                args.save_model_dir,
                "{0}_{1}_{2}.dat".format(
                    args.log_title, n_frames, local_start_time_str
                ),
            )
            torch.save(state_to_save, save_path)
    envs.close()
    log_writer.close()
    pbar.close()

if __name__ == "__main__":
    main()

