from default_args import args

args.update(
    train_scenes = {'bathroom':[31],},
    train_targets = {'bathroom':["SoapBottle"],},
    test_scenes = {'bathroom':[31],},
    test_targets = {'bathroom':["SoapBottle"],},
    action_dict = {
        'MoveAhead':['m0'],
        'TurnLeft':['r-90'],
        'TurnRight':['r90'],
        'BackOff':['m180']
    },
    grid_size = 0.5,
    total_train_epi = 2000,
    total_eval_epi = 100,
    threads = 4,
    log_title = 'DemoModel',
    optimizer = 'Adam',
    model = 'DemoModel',
    optim_args = dict(lr = args.lr,),
    print_freq = 100,
    max_epi_length = 100,
    model_save_freq = 2000,
    nsteps = 50,
    verbose = False,
    gpu_ids = -1,
    #load_model_dir = '',
    results_json = "result_demo.json"
)
model_args_dict = {}
args.update(
    model_args = model_args_dict,
)
