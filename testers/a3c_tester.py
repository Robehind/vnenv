import torch
#import setproctitle

def a3c_eval(
    args,
    thread_id,
    epi_num,
    result_queue,
    creator,
    chosen_scene_names = None,
    chosen_objects = None,
):
    #setproctitle.setproctitle("Testing Agent: {}".format(thread_id))
    #判断是否有gpu,分配gpu
    
    gpu_id = args.gpu_ids[thread_id % len(args.gpu_ids)]
    torch.cuda.set_device(gpu_id)
    #设置随机数种子
    torch.manual_seed(args.seed + thread_id)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + thread_id)
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
        list(args.state_dict.keys()),
        model,
        gpu_id
    )
    env = creator['env'](
        args.offline_data_dir,
        args.action_dict,
        args.target_dict,
        args.state_dict,
        args.reward_dict,
        grid_size = args.grid_size,
        chosen_objects = chosen_objects
    )
    #initialize a episode
    epi = creator['episode'](
        agent,
        env,
        chosen_scene_names,
        chosen_objects,
        args.max_epi_length,
        verbose = args.verbose
    )
    if args.verbose:
        print('Created all componets')

    count = 0

    while count < epi_num:
        #每一次循环为一次episode
        #episode重置
        epi.new_episode()#场景会重新随机选，目标也会随机选
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
        
        if args.verbose:
            print('epi ended')
            print('move length: %s'%(epi.move_length))
            print("ep_length: %s"%epi.length)

        results = {
            "agent_done": agent.done,#记录智能体是否提出了done这个动作
            "ep_length": epi.length,
            "success": epi.success,
            "total_reward":epi.total_reward,
            "spl":epi.compute_spl()
        }
        epi.end_episode()
        result_queue.put(results)

