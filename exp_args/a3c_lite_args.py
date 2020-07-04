from .default_args import args

args.update(
    train_scenes = {'kitchen':'25',},#{'bathroom':[31],},
    train_targets = {'kitchen':["Microwave", 'Sink'],},
    test_scenes = {'kitchen':'25',},#{'bathroom':[31],},
    test_targets = {'kitchen':["Microwave", 'Sink'],},
    action_dict = {
        'MoveAhead':['m0'],
        'TurnLeft':['r-90'],
        'TurnRight':['r90'],
        #'BackOff':['m180']
        #'Done':None,
    },
    obs_dict = {
        'fc':'resnet50_fc_new.hdf5',
        #'score':'resnet50_score.hdf5'
    },
    target_dict = {
        'glove':'../thordata/word_embedding/word_embedding.hdf5',
    },
    grid_size = 0.25,
    rotate_angle = 90,
    total_train_frames = 80000,
    total_eval_epi = 1000,
    threads = 4,
    exp_name = 'A3C_kitchen_2targets',
    optimizer = 'Adam',
    model = 'LiteModel',
    agent = 'A3CAgent',
    runner = 'A3CRunner',
    loss_func = 'a2c_loss',
    trainer = 'a3c_train',
    optim_args = dict(lr = args.lr,),
    print_freq = 1000,
    max_epi_length = 100,
    model_save_freq = 80000,
    nsteps = 40,
    verbose = False,
    gpu_ids = -1,
    results_json = "result_a3c_k25t2.json"
)
model_args_dict = dict(
        action_sz = len(args.action_dict),
        state_sz = 2048,
        target_sz = 300,
    )
args.update(
    model_args = model_args_dict,
)