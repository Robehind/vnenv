from .default_args import args

args.update(
    train_scenes = {'bathroom':[31],},
    train_targets = {'bathroom':["Sink","ToiletPaper","Towel","Cloth"],},
    test_scenes = {'bathroom':[31],},
    test_targets = {'bathroom':["Sink","ToiletPaper","Towel","Cloth"],},
    action_dict = {
        'MoveAhead':['m0'],
        'TurnLeft':['r-90'],
        'TurnRight':['r90'],
        #'BackOff':['m180']
        'Done':None,
    },
    grid_size = 0.5,
    total_train_epi = 20000,
    total_eval_epi = 1000,
    threads = 4,
    log_title = 'A3C',
    optimizer = 'Adam',
    model = 'LiteModel',
    optim_args = dict(lr = args.lr,),
    print_freq = 100,
    max_epi_length = 100,
    model_save_freq = 10000,
    nsteps = 50,
    verbose = False,
    gpu_ids = -1,
    load_model_dir = "../check_points/A3C_20000_2020-05-07_16-40-59.dat",
    results_json = "result_demo.json"
)
model_args_dict = dict(
        action_sz = len(args.action_dict),
        state_sz = 2048,
        target_sz = 300,
    )
args.update(
    model_args = model_args_dict,
)
