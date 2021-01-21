from ..default_args import args

args.update(
    seed = 1,
    min_len_file = 'min_len.json',
    exps_dir = '../brain_exp/resnet50lstm',
    train_scenes = {
        'kitchen':'1-15',
        'living_room':'1-15',
        'bedroom':'1-15',
        'bathroom':'1-15',
    },
    train_targets = {
        'kitchen':[
            "Toaster", "Microwave", "Fridge","CoffeeMaker",
            ],
        'living_room':[
            "Pillow", "Laptop", "Television","GarbageCan",
            ],
        'bedroom':["HousePlant", "Lamp", "Book", "AlarmClock"],
        'bathroom':["Sink", "ToiletPaper", "SoapBottle", "LightSwitch"],
    },
    test_scenes = {
        'kitchen':'16-20',
        'living_room':'16-20',
        'bedroom':'16-20',
        'bathroom':'16-20',
    },
    test_targets = {
        'kitchen':[
            "Toaster", "Microwave", "Fridge","CoffeeMaker",
            ],
        'living_room':[
            "Pillow", "Laptop", "Television","GarbageCan",
            ],
        'bedroom':["HousePlant", "Lamp", "Book", "AlarmClock"],
        'bathroom':["Sink", "ToiletPaper", "SoapBottle", "LightSwitch"],
    },
    action_dict = {
        'MoveAhead':['m0'],
        'TurnLeft':['r-45'],
        'TurnRight':['r45'],
        'Done':None,
    },
    obs_dict = {
        'fc':'resnet50fc.hdf5',
    },
    target_dict = {
        'glove':'../thordata/word_embedding/word_embedding.hdf5',
    },
    grid_size = 0.25,
    rotate_angle = 45,
    total_train_frames = 1e8,
    total_eval_epi = 1000,
    threads = 16,
    exp_name = 'exp7',
    optimizer = 'RMSprop',
    model = 'FcLstmModel',
    agent = 'A2CLstmAgent',
    runner = 'A2CRunner',
    loss_func = 'basic_loss',
    trainer = 'a2c_train',
    optim_args = dict(lr = 0.0007,alpha = 0.99, eps = 0.1),
    print_freq = 10000,
    max_epi_length = 200,
    model_save_freq = 1e7,
    nsteps = 40,
    gpu_ids = [0],
    reward_dict = {
        'collision':-0.1,
        'step':-0.01,
        'SuccessDone':10,
        'FalseDone':0,
    },
    #load_model_dir = '../brain_exp/resnet50linear/exp1_210118_144617/FcLinearModel_100000000_201124.dat',
    #load_optim_dir = '../EXPS/CNNpretrainLstm_Stable_5R_210109_202939/optim/RMSprop_100000000_021719.dat'
)
model_args_dict = dict(
        action_sz = len(args.action_dict)
    )
args.update(
    model_args = model_args_dict,
)
