import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
import time
import testers
import models
import agents
import runners
import environment as env
import torch
from tqdm import tqdm
from utils.mean_calc import ScalarMeanTracker
from utils.thordata_utils import get_scene_names
from utils.model_search import search_newest_model
import os

def main():
    #读取参数
    from exp_args.a3c_gcn_savn import args

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
    }
    tester = getattr(testers, args.tester)

    # 读取存档点，读取最新存档模型的参数到shared_model。其余线程会自动使用sync函数来同步
    load_model_dir = None
    if args.load_model_dir is not '':
        print("load %s"%args.load_model_dir)
        load_model_dir = args.load_model_dir
    else:
        find_path = search_newest_model(args.exps_dir, args.exp_name)
        if find_path is not None:
            print("Searched the neweset model: %s"%find_path)
            load_model_dir = find_path
        else:
            print("Can't find a neweset model. Load Nothing.")
   
    #这里用于分配各个线程的环境可以加载的场景以及目标
    chosen_scene_names = get_scene_names(args.test_scenes)
    chosen_objects = args.test_targets
    t_epis = args.total_eval_epi // len(chosen_scene_names.keys())
    total_epi = len(chosen_scene_names.keys())*t_epis

     #生成各个线程
    processes = []
    result_queue = mp.Queue()
    thread_id = 0
    for s_type in chosen_scene_names:
        if args.verbose:
            print('creating threads')
        p = mp.Process(
            target=tester,
            args=(
                args,
                thread_id,
                result_queue,
                load_model_dir,
                creator,
                chosen_scene_names[s_type],
                chosen_objects[s_type],
                t_epis,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)
        thread_id += 1
    print("Test agents created.")

    #取结果并记录
    all_scenes = [x for i in chosen_scene_names.values() for x in i]
    all_targets = [x for i in chosen_objects.values() for x in i]
    test_scalars = {k:ScalarMeanTracker() for k in all_scenes+all_targets}

    n_epis = 0

    pbar = tqdm(total = total_epi)

    try:
        while n_epis < total_epi:

            test_result = result_queue.get()
            n_epis += 1
            
            for k in test_result:
                test_scalars[k].add_scalars(test_result[k])
                test_scalars[k].add_scalars(dict(epis=1), False)
            pbar.update(1)

    finally:
        for p in processes:
            time.sleep(0.1)
            p.join()
    pbar.close()
    total_scalars = ScalarMeanTracker()
    result = {}
    for k in test_scalars:
        res = test_scalars[k].pop_and_reset()
        total_scalars.add_scalars(res)
        result[k] = res
    result.update(dict(Total = total_scalars.pop_and_reset()))
    import json
    with open(args.results_json, "w") as fp:
        json.dump(result, fp, sort_keys=False, indent=4)
    #json.dump()
    print(f'Results write into {args.results_json}')

if __name__ == "__main__":
    main()

