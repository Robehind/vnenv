import torch.multiprocessing as mp
import time
import json
import testers
import episodes
import models
import agents
import environment as env
import torch
from tqdm import tqdm
from utils.thordata_utils import get_scene_names, random_divide
from utils.mean_calc import ScalarMeanTracker

def main():
    #读取参数
    from exp_args.demo_args import args
    #查看载入模型是否存在
    if args.load_model_dir == '':
        print('Warining: load_model_dir didn\'t exist. Testing model with init params')
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
        'episode':getattr(episodes, args.episode),
        'agent':getattr(agents, args.agent),
        'env':getattr(env, args.env),
    }
    tester = getattr(testers, args.tester)
    #生成各个线程
    processes = []

    result_queue = mp.Queue()

    chosen_scene_names = get_scene_names(args.test_scenes)
    scene_names_div, thread_epi_num = random_divide(args.total_eval_epi, chosen_scene_names, args.threads)
    chosen_objects = []
    for k in args.test_targets.keys():
        chosen_objects = chosen_objects + args.test_targets[k]

    for thread_id in range(0, args.threads):
        p = mp.Process(
            target=tester,
            args=(
                args,
                thread_id,
                thread_epi_num[thread_id],
                result_queue,
                creator,
                #这里要注意
                [scene_names_div[thread_id]],
                chosen_objects,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)
    print("Test agents created.")

    #取结果并记录
    train_scalars = ScalarMeanTracker()

    epi_count = 0
    pbar = tqdm(total=args.total_eval_epi)

    try:
        while epi_count < args.total_eval_epi:

            result = result_queue.get()
            epi_count+=1
            pbar.update(1)
            train_scalars.add_scalars(result)

        tracked_means = train_scalars.pop_and_reset()

    finally:
        
        for p in processes:
            time.sleep(0.1)
            p.join()
    pbar.close()

    with open(args.results_json, "w") as fp:
        json.dump(tracked_means, fp, sort_keys=True, indent=4)

if __name__ == "__main__":
    main()

