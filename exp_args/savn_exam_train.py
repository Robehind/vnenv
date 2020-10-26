from .default_args import args

args.update(
    # train_scenes = {
    #     'kitchen':'1-20',

    #     },#{'bathroom':[31],},
    # train_targets = {'kitchen':["Microwave"],},
    seed = 1,
    train_scenes = {
        'kitchen':'1-20',
    },
    train_targets = {
        'kitchen':[
            "Toaster", "Microwave", "Fridge",
            "CoffeeMaker", "GarbageCan", "Box", "Bowl",
            ],
    },
    # test_scenes = {
    #     'kitchen':'22,24,26,28,30',
    # },
    # test_targets = {
    #     'kitchen':[
    #         "Toaster", "Microwave", "Fridge",
    #         "CoffeeMaker", "GarbageCan", "Box", "Bowl",
    #         ],
    # },
    test_sche_dir = '../thordata/test_schedule',
    shuffle = False,
    reward_dict = {
        'collision':0,
        'step':-0.01,
        'SuccessDone':5,
        'FalseDone':0,
    },
    action_dict = {
        'MoveAhead':['m0'],
        'TurnLeft':['r-45'],
        'TurnRight':['r45'],
        'LookUp':['p-30'],
        'LookDown':['p30'],
        'Done':None,
    },
    obs_dict = {
        'fc':'resnet50_fc_new.hdf5',
        },
    target_dict = {
        'glove':'../thordata/word_embedding/word_embedding.hdf5',
    },
    grid_size = 0.25,
    rotate_angle = 45,
    total_train_frames = 20000000,
    total_eval_epi = 1000,
    threads = 4,
    exp_name = 'LiteSavn',
    optimizer = 'SharedAdam',
    model = 'LiteSAVN',
    agent = 'OriSavnAgent',
    runner = 'SavnRunner',
    loss_func = 'savn_loss',
    trainer = 'ori_savn_train',
    tester = 'savn_test',
    optim_args = dict(lr = 0.0001),
    inner_lr = 0.0001,
    print_freq = 10000,
    max_epi_length = 200,
    model_save_freq = 500000,
    nsteps = 6,
    verbose = False,
    gpu_ids = [0],
    #load_model_dir = '../EXPS/LiteSavn_200918_192454/LiteSAVN_3000002_224122.dat',
    results_json = "result.json"
)
model_args_dict = {'action_sz' : len(args.action_dict),'nsteps':args.nsteps}
args.update(
    model_args = model_args_dict,
)

