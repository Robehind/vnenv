from .default_args import args

args.update(
    train_scenes = {'bedroom':'27'},
    train_targets = {'bedroom':["AlarmClock"],},
    #test_scenes = {'kitchen':'25',},
    #test_targets = {'kitchen':["Microwave", 'Sink'],},
    test_scenes = {
        'kitchen':'21-30',
        'living_room':'21-22,24-30',
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
    action_dict = {
        'MoveAhead':['m0'],
        'TurnLeft':['r-90'],
        'TurnRight':['r90'],
        #'BackOff':['m180']
        #'Done':None,
    },
    obs_dict = {
        'fc':'resnet50_fc_new.hdf5',
    },
    target_dict = {
        'glove':'../thordata/word_embedding/word_embedding.hdf5',
    },
    grid_size = 0.25,
    rotate_angle = 90,
    total_train_frames = 40000,
    total_eval_epi = 1000,
    threads = 4,
    exp_name = '327A2CLiteDemo',
    optimizer = 'Adam',
    model = 'LiteModel',
    agent = 'A2CAgent',
    runner = 'A2CRunner',
    loss_func = 'a2c_loss',
    trainer = 'a2c_train',
    optim_args = dict(lr = args.lr,),
    print_freq = 1000,
    max_epi_length = 100,
    model_save_freq = 40000,
    nsteps = 10,
    verbose = False,
    gpu_ids = [0],
    results_json = "result_a2c_demo.json"
)
model_args_dict = dict(
        action_sz = len(args.action_dict),
        state_sz = 2048,
        target_sz = 300,
    )
args.update(
    model_args = model_args_dict,
)
