import torch
from .train_util import copy_gradient
from torch.nn.utils import clip_grad_norm_
#import setproctitle

from .loss_functions import a3c_loss

def a3c_train(
    args,
    thread_id,
    result_queue,
    end_flag,#多线程停止位
    shared_model,
    creator,
    chosen_scene_names = None,
    chosen_objects = None,
):
    #setproctitle.setproctitle("Training Agent: {}".format(thread_id))
    #判断是否有gpu,分配gpu
    if args.verbose:
        print('agent %s created'%thread_id)
    
    gpu_id = args.gpu_ids[thread_id % len(args.gpu_ids)]
    torch.cuda.set_device(gpu_id)
    #设置随机数种子
    torch.manual_seed(args.seed + thread_id)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + thread_id)
    #initialize env and agent

    model = creator['model'](**args.model_args)
    if args.verbose:
        print('model created')
    agent = creator['agent'](
        list(args.action_dict.keys()),
        list(args.obs_dict.keys()),
        model,
        gpu_id
    )
    if args.verbose:
        print('agent created')
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
        verbose = args.verbose
    )
    optim = creator['optimizer'](
        shared_model.parameters(),
        **args.optim_args
    )
    if args.verbose:
        print('env and episode created')
    if args.verbose:
        print('Created all componets')

    while not end_flag.value:
        #每一次循环为一次episode
        #episode重置
        loss_tracker = {
        'total_loss':0,
        'policy_loss':0,
        'value_loss':0,
        }
        epi.new_episode()#场景会重新随机选，目标也会随机选
        # Train on the new episode.
        while not epi.done:
            #同步全局模型参数
            agent.sync_with_shared(shared_model)
            # Run episode for num_steps or until player is done.
            exps, pi_batch, v_batch, last_v = epi.get_nstep_exps(args.nsteps)
            if args.verbose:
                print('Got exps')
            # Compute the loss.
            loss = a3c_loss(epi.done, v_batch, pi_batch, last_v, exps, gpu_id=gpu_id)
            loss["total_loss"].mean().backward()
            for k in loss_tracker:
                loss_tracker[k]+=loss[k].mean().cpu().item()
            
            if args.verbose:
                print('Loss computed')

            clip_grad_norm_(model.parameters(), 50.0)

            # Transfer gradient to shared model and step optimizer.
            copy_gradient(shared_model, model)
            optim.step()
            model.zero_grad()
            
            if args.verbose:
                print('optimized')
            # Clear actions
            epi.clear_exps()
                
        results = {
            "agent_done": agent.done,#记录智能体是否提出了done这个动作
            "ep_length": epi.length,
            "success": epi.success,
            "total_reward":epi.total_reward
        }
        results.update(loss_tracker)
        epi.end_episode()
        result_queue.put(results)
