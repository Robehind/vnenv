from .default_args import args

args.update(
    train_scenes = {'kitchen':'25',},#{'bathroom':[31],},
    train_targets = {'kitchen':["Microwave"],},
    test_scenes = {'kitchen':'25',},#{'bathroom':[31],},
    test_targets = {'kitchen':["Microwave"],},
    action_dict = {
        'MoveAhead':['m0'],
        'TurnLeft':['r-90'],
        'TurnRight':['r90'],
        #'BackOff':['m180']
    },
    obs_dict = {
        'fc':'resnet50_fc_new.hdf5',
        #'score':'resnet50_score.hdf5'
        },
    target_dict = {},
    grid_size = 0.25,
    rotate_angle = 90,
    total_train_frames = 40000,
    total_eval_epi = 1000,
    threads = 4,
    exp_name = 'A3CNoMask',
    optimizer = 'Adam',
    model = 'DemoModel',
    agent = 'A3CRecordAgent',
    runner = 'A3CNoMaskRunner',
    loss_func = 'basic_loss_no_mask',
    trainer = 'a3c_train',
    optim_args = dict(lr = args.lr,),
    print_freq = 1000,
    max_epi_length = 100,
    model_save_freq = 40000,
    nsteps = 20,
    verbose = False,
    gpu_ids = -1,
    #load_model_dir = '../check_points/A2CDemoModel_40000_2020-05-20_10-49-28.dat',
    results_json = "result_demo40000.json"
)
model_args_dict = {'action_size' : len(args.action_dict)}
args.update(
    model_args = model_args_dict,
)

