import time
import random
import torch
from .train_util import get_params, transfer_gradient_to_shared, SGD_step
from utils.mean_calc import ScalarMeanTracker
from utils.env_wrapper import SingleEnv


def savn_train(
    args,
    thread_id,
    result_queue,
    end_flag,#多线程停止位
    shared_model,
    optim,
    creator,
    loss_func,
    chosen_scene_names = None,
    chosen_objects = None,
    gradient_limit = 4
):

    gpu_id = args.gpu_ids[thread_id % len(args.gpu_ids)]
    torch.cuda.set_device(gpu_id)
    #initialize env and agent

    model = creator['model'](**args.model_args)
    agent = creator['agent'](
        list(args.action_dict.keys()),
        model,
        gpu_id
    )
    env = creator['env'](
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
    env = SingleEnv(env, False)
    #initialize a runner
    runner = creator['runner'](
        args.nsteps, 1, env, agent
    )
    if optim == None:
        optim = creator['optimizer'](
            shared_model.parameters(),
            **args.optim_args
        )
    #n_frames = 0
    #update_frames = args.nsteps
    loss_tracker = ScalarMeanTracker()
    while not end_flag.value:
        

        # theta <- shared_initialization
        params_list = [get_params(shared_model, gpu_id)]
        params = params_list[-1]
        #loss_dict = {}
        episode_num = 0
        num_gradients = 0
        exps = {
            'rewards':[],
            #'masks':[],
            'action_idxs':[]
        }
        agent.reset_hidden()
        agent.clear_mems()
        
        # Accumulate loss over all meta_train episodes.
        while True:
            agent.learned_input = None
            # Run episode for k steps or until it is done or has made a mistake (if dynamic adapt is true).

            if args.verbose:
                print("New inner step")
            exps_ = runner.run(params)
            for k in exps:
                exps[k] += exps_[k]

            if runner.done:
                break

            if gradient_limit < 0 or episode_num < gradient_limit:

                num_gradients += 1

                # Compute the loss.
                #loss_hx = torch.cat((agent.hidden[0], agent.last_action_probs), dim=1)
                learned_loss = agent.model.learned_loss(
                        agent.learned_input, params
                    )
                agent.learned_input = None

                if args.verbose:
                    print("inner gradient")
                inner_gradient = torch.autograd.grad(
                    learned_loss,
                    [v for _, v in params_list[episode_num].items()],
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )

                params_list.append(
                    SGD_step(params_list[episode_num], inner_gradient, args.inner_lr)
                )
                params = params_list[-1]

                # reset_player(player)
                episode_num += 1

                #for k, v in learned_loss.items():
                    #loss_dict["{}/{:d}".format(k, episode_num)] = v.item()

        #loss = compute_loss(args, player, gpu_id, model_options)
        pi_batch = torch.cat(agent.pi_batch, dim = 0)
        v_batch = torch.cat(agent.v_batch, dim = 0)
        if runner.done:
            v_final = 0.0
        else:
            model_out = agent.model_forward(runner.last_obs, params = params)
            v_final = model_out['value'].detach().cpu().item()
        loss = loss_func(v_batch, pi_batch, v_final, exps, gpu_id=gpu_id)
        for k in loss:
            loss_tracker.add_scalars({k:loss[k].item()})

        if args.verbose:
            print("meta gradient")

        # Compute the meta_gradient, i.e. differentiate w.r.t. theta.
        meta_gradient = torch.autograd.grad(
            loss["total_loss"],
            [v for _, v in params_list[0].items()],
            allow_unused=True,
        )

        transfer_gradient_to_shared(meta_gradient, shared_model, gpu_id)
        optim.step()
        #model.zero_grad()
        
        results = runner.pop_mems()

        results.update(loss_tracker.pop_and_reset())
        result_queue.put(results)
        


