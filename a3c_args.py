from default_args import args

args.update(
    train_scenes = {'kitchen':[25],},
    train_targets = {'kitchen':["Microwave"],},
    test_scenes = {'kitchen':[25],},
    test_targets = {'kitchen':["Microwave"],},
    action_dict = {
        'MoveAhead':['m0'],
        'TurnLeft':['r-90'],
        'TurnRight':['r90'],
        'BackOff':['m180']
    },
    grid_size = 0.25,
    total_train_epi = 2000,
    total_eval_epi = 100,
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
    #load_model_dir = "../check_points/A3CTrain_89905_2000_2020-04-25_16-23-30.dat",
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
