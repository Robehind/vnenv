from ..default_args import args

args.update(
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
    #train_scenes = {'kitchen':'25'},
    #train_targets = {'kitchen':["Microwave", 'Sink'],},
    test_scenes = {'kitchen':'25',},
    test_targets = {'kitchen':["Microwave", 'Sink'],},
    action_dict = {
        'MoveAhead':['m0'],
        'TurnLeft':['r-45'],
        'TurnRight':['r45'],
        'Done':None,
    },
    obs_dict = {
        'image':'images128.hdf5',
    },
    target_dict = {
        'glove':'../thordata/word_embedding/word_embedding.hdf5',
    },
    grid_size = 0.25,
    rotate_angle = 45,
    total_train_frames = 18e7,
    total_eval_epi = 1000,
    threads = 16,
    exp_name = 'Perception1',
    optimizer = 'RMSprop',
    model = 'SplitLstm',
    agent = 'A2CLstmAgent',
    runner = 'A2CRunner',
    loss_func = 'loss_with_entro',
    trainer = 'a2c_train',
    optim_args = dict(lr = 0.0007,alpha = 0.99, eps = 0.1),
    print_freq = 10000,
    max_epi_length = 200,
    model_save_freq = 1e7,
    nsteps = 10,
    gpu_ids = [0],
    #load_model_dir = '/home/zhiyu/EXPS/Perception1_201228_105017/SplitLstm_120000000_072718.dat',
    #load_optim_dir = '/home/zhiyu/EXPS/Perception1_201228_105017/optim/RMSprop_120000000_072718.dat',
)
model_args_dict = dict(
        action_sz = len(args.action_dict)
    )
args.update(
    model_args = model_args_dict,
)
