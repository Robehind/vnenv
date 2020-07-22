import os
#os.environ["MKL_NUM_THREADS"] = '4'
#os.environ["NUMEXPR_NUM_THREADS"] = '4'
os.environ["OMP_NUM_THREADS"] = '1'
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
import ctypes
import time
import trainers
import models
import agents
import runners
import environment as env
import optimizers
import torch
from tqdm import tqdm
from utils.mean_calc import ScalarMeanTracker
from utils.thordata_utils import get_scene_names, random_divide

def main():
    #读取参数
    from exp_args.a3c_gcn_savn import args
    #生成日志文件
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
        #torch.cuda.manual_seed(args.seed)
        assert torch.cuda.is_available()
        mp.set_start_method("spawn")

    #动态载入构造函数
    creator = {
        'model':getattr(models, args.model),
        'agent':getattr(agents, args.agent),
        'env':getattr(env, args.env),
        'runner':getattr(runners, args.runner),
        'optimizer':getattr(optimizers, args.optimizer)
    }
    trainer = getattr(trainers, args.trainer)
    loss_func = getattr(trainers, args.loss_func)
    #生成全局模型并初始化优化算法
    shared_model = creator['model'](**args.model_args)
    if shared_model is not None:
        shared_model.share_memory()
        #optimizer.share_memory()
        print(shared_model)
    # 读取存档点，读取最新存档模型的参数到shared_model。其余线程会自动使用sync函数来同步
    if args.load_model_dir is not '':
        print("load %s"%args.load_model_dir)
        shared_model.load_state_dict(torch.load(args.load_model_dir))
   

    #这里用于分配各个线程的环境可以加载的场景以及目标
    #暂时就每个线程都加载的物体
    chosen_scene_names = get_scene_names(args.train_scenes)
    scene_names_div, _ = random_divide(1000, chosen_scene_names, args.threads)
    chosen_objects = args.train_targets

    #初始化TX
    log_writer = SummaryWriter(log_dir = args.exp_dir)

     #生成各个线程
    processes = []
    end_flag = mp.Value(ctypes.c_bool, False)
    result_queue = mp.Queue()

    for thread_id in range(0, args.threads):
        if args.verbose:
            print('creating threads')
        p = mp.Process(
            target=trainer,
            args=(
                args,
                thread_id,
                result_queue,
                end_flag,
                shared_model,
                creator,
                loss_func,
                scene_names_div[thread_id],
                chosen_objects,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)
    print("Train agents created.")

    #取结果并记录
    print_freq = args.print_freq
    save_freq = args.model_save_freq
    train_scalars = ScalarMeanTracker()

    print_gate_frames = print_freq
    save_gate_frames = save_freq

    n_epis = 0
    n_frames = 0


    pbar = tqdm(total=args.total_train_frames)

    try:
        while n_frames < args.total_train_frames:

            train_result = result_queue.get()
            n_epis += train_result.pop('epis')
            update_frames = train_result.pop('n_frames')
            train_scalars.add_scalars(train_result)
            
            pbar.update(update_frames)
            n_frames += update_frames
            if n_frames >= print_gate_frames:
                print_gate_frames += print_freq
                log_writer.add_scalar("n_epis", n_epis, n_frames)
                tracked_means = train_scalars.pop_and_reset()
                for k, v in tracked_means.items():
                    log_writer.add_scalar(k, v, n_frames)

            if n_frames >= save_gate_frames:
                save_gate_frames += save_freq
                if not os.path.exists(args.exp_dir):
                    os.makedirs(args.exp_dir)
                state_to_save = shared_model.state_dict()
                start_time = time.time()
                time_str = time.strftime(
                    "%H%M%S", time.localtime(start_time)
                )
                save_path = os.path.join(
                    args.exp_dir,
                    f'{args.model}_{n_frames}_{time_str}.dat'
                )
                torch.save(state_to_save, save_path)

    finally:
        log_writer.close()
        end_flag.value = True
        for p in processes:
            time.sleep(0.1)
            p.join()
    pbar.close()

if __name__ == "__main__":
    main()

