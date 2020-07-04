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
        'res18fm':'resnet18_featuremap.hdf5',
        #'score':'resnet50_score.hdf5'
        },
    target_dict = {
        'glove':'../thordata/word_embedding/word_embedding.hdf5',
    },
    grid_size = 0.25,
    rotate_angle = 90,
    total_train_frames = 100000,
    total_eval_epi = 1000,
    threads = 4,
    exp_name = 'savnBaseModel',
    optimizer = 'RMSprop',
    model = 'BaseModel',
    agent = 'A3CLstmAgent',
    runner = 'A3CRunner',
    loss_func = 'a2c_loss',
    trainer = 'a3c_train',
    optim_args = dict(lr = 0.0001, alpha = 0.99, eps = 0.1),
    print_freq = 1000,
    max_epi_length = 100,
    model_save_freq = 25000,
    nsteps = 40,
    verbose = False,
    gpu_ids = [0, 1],
    #load_model_dir = '../check_points/A2CDemoModel_40000_2020-05-20_10-49-28.dat',
    results_json = "result_savnbase.json"
)
model_args_dict = {'action_sz' : len(args.action_dict)}
args.update(
    model_args = model_args_dict,
)
