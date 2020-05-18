from .default_args import args

args.update(
    train_scenes = {'kitchen':[25],},
    train_targets = {'bathroom':["Microwave"],},
    test_scenes = {'kitchen':[1],},#{'bathroom':[31],},
    test_targets = {'bathroom':["Microwave"],},
    action_dict = {
        'MoveAhead':['m0'],
        'TurnLeft':['r-90'],
        'TurnRight':['r90'],
        #'Done':None
        #'BackOff':['m180']
    },
    obs_dict = {
        'fc':'resnet50_fc_new.hdf5',
        #'score':'resnet50_score.hdf5'
        },
    grid_size = 0.25,
    rotate_angle = 90,
    total_train_frames = 40000,
    total_eval_epi = 1000,
    threads = 4,
    log_title = 'A2CDemoModel',
    optimizer = 'Adam',
    model = 'DemoModel',
    agent = 'A2CAgent',
    optim_args = dict(lr = args.lr,),
    print_freq = 1000,
    max_epi_length = 100,
    model_save_freq = 200000,
    nsteps = 10,
    verbose = False,
    gpu_ids = [0],
    #load_model_dir = '../check_points/DemoModel_100006_2020-05-16_21-35-19.dat',
    results_json = "result_demo.json"
)
model_args_dict = {'action_size' : len(args.action_dict)}
args.update(
    model_args = model_args_dict,
)
