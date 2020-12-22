from ..default_args import args

args.update(
    train_scenes = {
        'kitchen':'1-16',
    },
    train_targets = {
        'kitchen':[
            "Toaster", "Microwave", "Fridge",
            "CoffeeMaker", "GarbageCan", "Box", "Bowl",
            ],
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
        'image':'images.hdf5',
    },
    target_dict = {
        'glove':'../thordata/word_embedding/word_embedding.hdf5',
    },
    grid_size = 0.25,
    rotate_angle = 45,
    total_train_frames = 20000000,
    total_eval_epi = 1000,
    threads = 16,
    exp_name = 'Simple3train',
    optimizer = 'RMSprop',
    model = 'Simple3',
    agent = 'A2CLstmAgent',
    runner = 'A2CRunner',
    loss_func = 'loss_with_entro',
    trainer = 'a2c_train',
    optim_args = dict(lr = 0.0007,alpha = 0.99, eps = 0.1),
    print_freq = 10000,
    max_epi_length = 200,
    model_save_freq = 4000000,
    nsteps = 10,
    gpu_ids = [0],
)
model_args_dict = dict(
        action_sz = len(args.action_dict)
    )
args.update(
    model_args = model_args_dict,
)
