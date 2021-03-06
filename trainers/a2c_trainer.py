from tqdm import tqdm
from utils.mean_calc import ScalarMeanTracker
from utils.net_utils import save_model
import os

def a2c_train(
    args,
    agent,
    envs,
    runner,
    optimizer,
    loss_func,
    tx_writer
    ):
    #TODO 在a2c中暂时只调用一块gpu用于训练，多线程训练可能需要调用pytorch本身的api
    gpu_id = args.gpu_ids[0]
    
    n_frames = 0
    update_frames = args.nsteps * args.threads
    total_epis = 0
    print_freq = args.print_freq
    save_freq = args.model_save_freq
    print_gate_frames = print_freq
    save_gate_frames = save_freq
    loss_traker = ScalarMeanTracker()
    pbar = tqdm(total=args.total_train_frames)
    while n_frames < args.total_train_frames:
        
        batch_out, v_final, exps = runner.run()
        #loss = loss_func(v_batch, pi_batch, v_final, exps, gpu_id)
        loss = loss_func(batch_out, v_final, exps, gpu_id)
        optimizer.zero_grad()
        loss['total_loss'].backward()
        for k in loss:
            loss_traker.add_scalars({k:loss[k].item()})
        optimizer.step()

        #记录、保存、输出
        pbar.update(update_frames)
        n_frames += update_frames
        
        if n_frames >= print_gate_frames:
            print_gate_frames += print_freq
            record = runner.pop_mems()
            total_epis += record.pop('epis')
            tx_writer.add_scalar("n_frames", n_frames, total_epis)
            for k,v in record.items():
                tx_writer.add_scalar(k, v, n_frames)
            for k,v in loss_traker.pop_and_reset().items():
                tx_writer.add_scalar(k, v, n_frames)

        if n_frames >= save_gate_frames:
            save_gate_frames += save_freq
            agent.save_model(args.exp_dir, f'{args.model}_{n_frames}')
            optim_path = os.path.join(args.exp_dir, 'optim')
            save_model(optimizer, optim_path, f'{args.optimizer}_{n_frames}')

    envs.close()
    tx_writer.close()
    pbar.close()