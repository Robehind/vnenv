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
    creator,
    loss_func,
    chosen_scene_names = None,
    chosen_objects = None,
    gradient_limit = 4
):

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
    optim = creator['optimizer'](
        shared_model.parameters(),
        **args.optim_args
    )
    n_frames = 0
    update_frames = args.nsteps
    loss_tracker = ScalarMeanTracker()
    while not end_flag.value:
        

        # theta <- shared_initialization
        params_list = [get_params(shared_model, gpu_id)]
        params = params_list[-1]
        loss_dict = {}
        episode_num = 0
        num_gradients = 0

        # Accumulate loss over all meta_train episodes.
        while True:
            # Run episode for k steps or until it is done or has made a mistake (if dynamic adapt is true).
            agent.sync_with_shared(shared_model)
            if args.verbose:
                print("New inner step")
            pi_batch, v_batch, v_final, exps = runner.run()

            if epi.done:
                break

            if gradient_limit < 0 or episode_num < gradient_limit:

                num_gradients += 1

                # Compute the loss.
                loss_hx = torch.cat((agent.hidden[0], agent.last_action_probs), dim=1)
                learned_loss = {
                    "learned_loss": agent.model.learned_loss(
                        loss_hx, agent.learned_input, params
                    )
                }
                agent.learned_input = None

                if args.verbose:
                    print("inner gradient")
                inner_gradient = torch.autograd.grad(
                    learned_loss["learned_loss"],
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

                for k, v in learned_loss.items():
                    loss_dict["{}/{:d}".format(k, episode_num)] = v.item()

        #loss = compute_loss(args, player, gpu_id, model_options)
        policy_loss, value_loss = loss_func(args, agent, gpu_id, params)
        total_loss = policy_loss + 0.5 * value_loss
        loss = dict(
            total_loss=total_loss, 
            policy_loss=policy_loss, 
            value_loss=value_loss
            )

        for k, v in loss.items():
            loss_dict[k] = v.item()

        if args.verbose:
            print("meta gradient")

        # Compute the meta_gradient, i.e. differentiate w.r.t. theta.
        meta_gradient = torch.autograd.grad(
            loss["total_loss"],
            [v for _, v in params_list[0].items()],
            allow_unused=True,
        )
        
        results = {
            "done_count": agent.done,
            "ep_length": epi.length,
            "success": epi.success,
            "total_reward":epi.total_reward
        }

        result_queue.put(results)

        # Copy the meta_gradient to shared_model and step.
        transfer_gradient_to_shared(meta_gradient, shared_model, gpu_id)
        optim.step()
        epi.end_episode()
        


