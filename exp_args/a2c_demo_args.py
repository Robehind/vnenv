from .default_args import args

args.update(
    # train_scenes = {'bedroom':'27',},
    # train_targets = {'bedroom':["Mirror"],},
    # test_scenes = {'bedroom':'27',},
    # test_targets = {'bedroom':["Mirror"],},
    # #test_scenes = {'kitchen':'25',},#{'bathroom':[31],},
    # #test_targets = {'kitchen':["Microwave"],},
    # action_dict = {
    #     'MoveAhead':['m0'],
    #     'TurnLeft':['r-45'],
    #     'TurnRight':['r45'],
    #     #'Done':None
    #     #'BackOff':['m180']
    # },
    obs_dict = {
        'fc':'resnet50_fc_new.hdf5',
        #'score':'resnet50_score.hdf5'
        },
    grid_size = 0.25,
    rotate_angle = 45,
    total_train_frames = 40000,
    total_eval_epi = 1000,
    threads = 4,
    exp_name = 'A2CDemoModel',
    optimizer = 'Adam',
    model = 'DemoModel',
    agent = 'A2CAgent',
    runner = 'A2CRunner',
    loss_func = 'a2c_loss',
    trainer = 'a2c_train',
    optim_args = dict(lr = args.lr,),
    print_freq = 1000,
    max_epi_length = 250,
    model_save_freq = 40000,
    nsteps = 10,
    verbose = False,
    gpu_ids = [0],
    results_json = "result_demo.json"
)
model_args_dict = {'action_size' : len(args.action_dict)}
args.update(
    model_args = model_args_dict,
)
