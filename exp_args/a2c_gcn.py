from .default_args import args

args.update(
    seed = 1,
    test_scenes = {
        'kitchen':'21-30',
        'living_room':'21-30',
        'bedroom':'21-30',
        'bathroom':'21-30',
    },
    test_targets = {
        'kitchen':[
            "Toaster", "Microwave", "Fridge",
            "CoffeeMaker", "GarbageCan", "Box", "Bowl",
            ],
        'living_room':[
            "Pillow", "Laptop", "Television",
            "GarbageCan", "Box", "Bowl",
            ],
        'bedroom':["HousePlant", "Lamp", "Book", "AlarmClock"],
        'bathroom':["Sink", "ToiletPaper", "SoapBottle", "LightSwitch"],
    },
    
    test_sche_dir = '../thordata/test_schedule',
    shuffle = False,
    action_dict = {
        'MoveAhead':['m0'],
        'TurnLeft':['r-45'],
        'TurnRight':['r45'],
        'LookUp':['p-30'],
        'LookDown':['p30'],
        'Done':None,
    },
    obs_dict = {
        'fc|4':'resnet50_fc_new.hdf5',
        'score':'resnet50_score.hdf5'
        },
    grid_size = 0.25,
    rotate_angle = 45,
    total_train_frames = 20000,
    total_eval_epi = 1000,
    threads = 4,
    exp_name = 'A2CGCN',
    optimizer = 'RMSprop',
    model = 'ScenePriorsModel',
    agent = 'A2CAgent',
    runner = 'A2CRunner',
    loss_func = 'basic_loss',
    trainer = 'a2c_train',
    optim_args = dict(lr = 0.0007, alpha = 0.99, eps = 0.1),
    print_freq = 1000,
    max_epi_length = 300,
    model_save_freq = 1000000,
    nsteps = 10,
    verbose = False,
    gpu_ids = [0],
    #load_model_dir = "../check_points/A2CGCN_4000000_2020-05-21_17-44-59.dat",
    results_json = "result.json"
)
model_args_dict = dict(
        action_sz = len(args.action_dict),
        state_sz = 8192,
        target_sz = 300,
    )
args.update(
    model_args = model_args_dict,
)
