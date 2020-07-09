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
from utils.thordata_utils import get_scene_names, get_type
from utils.model_search import search_newest_model
import os

def main():
    #读取参数
    from exp_args.a3c_gcn_savn import args

    #生成测试文件夹
    start_time = time.time()
    time_str = time.strftime(
        "%y%m%d_%H%M%S", time.localtime(start_time)
    )
    args.exp_dir = os.path.join(args.exps_dir, 'TEST'+args.exp_name + '_' + time_str)
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    # 通过指定路径寻找参数或者自动寻找最新模型
    if args.load_model_dir is not '':
        print("load %s"%args.load_model_dir)
    else:
        find_path = search_newest_model(args.exps_dir, args.exp_name)
        if find_path is not None:
            print("Searched the neweset model: %s"%find_path)
            args.load_model_dir = find_path
        else:
            print("Can't find a neweset model. Load Nothing.")

    #保存本次测试的参数
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
    }
    tester = getattr(testers, args.tester)
   
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
                args.load_model_dir,
                creator,
                chosen_scene_names[s_type],
                chosen_objects,
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
    all_targets = [
        y+'/'+x 
        for i in chosen_objects
        for y in chosen_scene_names[i] 
        for x in chosen_objects[i] 
        ]
    test_scalars = {k:ScalarMeanTracker() for k in all_scenes + all_targets}

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
    total_scalars = {k:ScalarMeanTracker() for k in chosen_scene_names}
    scene_split = {k:{} for k in chosen_scene_names}
    target_split = {
        k:{i:ScalarMeanTracker() for i in chosen_objects[k]} 
        for k in chosen_scene_names
        }
    result = {k:v.pop_and_reset() for k,v in test_scalars.items()}

    for k in result:
        k_sp = k.split('/')
        s_type = get_type(k_sp[0])
        if result[k] == {}:
            continue
        if len(k_sp) == 1:
            scene_split[s_type][k] = result[k].copy()
            epis = result[k].pop('epis')
            total_scalars[s_type].add_scalars(result[k])
            total_scalars[s_type].add_scalars(dict(epis=epis), False)
        else:
            epis = result[k].pop('epis')
            target_split[s_type][k_sp[-1]].add_scalars(result[k])
            target_split[s_type][k_sp[-1]].add_scalars(dict(epis=epis), False)
    
    for k in target_split:
        for t in chosen_objects[k]:
            target_split[k][t] = target_split[k][t].pop_and_reset()
        total_scalars[k] = total_scalars[k].pop_and_reset()
    
    import json
    for k in scene_split:
        tmp = dict(Total = total_scalars[k].copy())
        scene_split[k].update(target_split[k])
        tmp.update(scene_split[k])
        result_path = os.path.join(args.exp_dir, k+'_'+args.results_json)
        with open(result_path, "w") as fp:
            json.dump(tmp, fp, indent=4)

    all_objs = list(set([x for i in chosen_objects.values() for x in i]))
    all_objs.sort()
    total_t = {x:ScalarMeanTracker() for x in all_objs} 
    total_s = ScalarMeanTracker()

    for k in target_split:
        for t in target_split[k]:
            if target_split[k][t] == {}:
                continue
            epis = target_split[k][t].pop('epis')
            total_t[t].add_scalars(target_split[k][t])
            total_t[t].add_scalars(dict(epis=epis), False)
        epis = total_scalars[k].pop('epis')    
        total_s.add_scalars(total_scalars[k])
        total_s.add_scalars(dict(epis=epis), False)
    total_s = dict(Total = total_s.pop_and_reset())
    for k in total_t:
        total_t[k] = total_t[k].pop_and_reset()
    total_s.update(total_t)
    result_path = os.path.join(args.exp_dir, 'Total_'+args.results_json)
    with open(result_path, "w") as fp:
        json.dump(total_s, fp, indent=4)
    print(f'Results write into {args.results_json}')

if __name__ == "__main__":
    main()

