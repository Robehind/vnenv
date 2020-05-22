from .default_args import args

args.update(
    train_scenes = {'kitchen':range(1,9),},
    train_targets = {'kitchen':["Toaster", "Microwave", "Fridge", "CoffeeMachine", "GarbageCan", "Bowl"],},
    test_scenes = {'kitchen':range(9,16),},
    test_targets = {'bedroom':["Mug","Pot","Cup"]},
    action_dict = {
        'MoveAhead':['m0'],
        'TurnLeft':['r-90'],
        'TurnRight':['r90'],
        #'BackOff':['m180']
        'Done':None,
    },
    obs_dict = {
        'fc|4':'resnet50_fc_new.hdf5',
        'score':'resnet50_score.hdf5'
        },
    grid_size = 0.25,
    rotate_angle = 90,
    total_train_frames = 2000000,
    total_eval_epi = 1000,
    threads = 4,
    log_title = 'A2CGCN',
    optimizer = 'RMSprop',
    model = 'ScenePriorsModel',
    agent = 'A2CAgent',
    optim_args = dict(lr = 0.0007, alpha = 0.99, eps = 0.1),
    print_freq = 10000,
    max_epi_length = 300,
    model_save_freq = 1000000,
    nsteps = 10,
    verbose = False,
    gpu_ids = [0],
    load_model_dir = "../check_points/A2CGCN_4000000_2020-05-21_17-44-59.dat",
    results_json = "done_gcn_unseenk_uo.json"
)
model_args_dict = dict(
        action_sz = len(args.action_dict),
        state_sz = 8192,
        target_sz = 300,
    )
args.update(
    model_args = model_args_dict,
)
