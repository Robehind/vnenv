import torch
import torch.nn.functional as F
from environment import DiscreteEnvironment as Denv
import models
from agents import OriSavnAgent
import os,h5py
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils.net_utils import toFloatTensor, gpuify

model_str = 'LiteSAVN'
scene_name = 'FloorPlan1_physics'
model_param = [6,6]
exp_dir = '/home/zhiyu/EXPS/LiteSavn_200918_192454'#LstmLinerLoss_200922_202547'#

poskey = '1.75|-2.00|315|0'
data_dir = '../thordata/mixed_offline_data/'
fc = h5py.File(data_dir+scene_name+'/resnet50_fc_new.hdf5', "r",)
glove = h5py.File('../thordata/word_embedding/word_embedding.hdf5', "r",)
hidden = (torch.zeros(1,512), torch.zeros(1,512))

model = getattr(models,model_str)(*model_param)
with torch.cuda.device(0):
    model = model.cuda()
#agent = OriSavnAgent(['1','2','3','4','5','6'],model,gpu_id=0)

out_dir = '/home/zhiyu/EXPS/grad_mean_var_curve/'+model_str
_, _, files = next(os.walk(exp_dir))
model_names = [f for f in files if 'dat' in f]
model_names.sort(key = lambda p:int(p.split("_")[1]))
logger = SummaryWriter(out_dir)

input_ = {
        'fc':toFloatTensor(fc[poskey][:],0).unsqueeze_(0),
        #'res18fm':toFloatTensor(fc[poskey][:],0).unsqueeze_(0),
        'hidden':toFloatTensor(hidden,0),
        'glove':toFloatTensor(glove['glove']['Microwave'][:],0).unsqueeze_(0),
        #'action_probs':torch.zeros(1,6).cuda()
    }
bar = tqdm(total=len(model_names))
for m in model_names:
    bar.update(1)
    load_dir = os.path.join(exp_dir, m)
    frame = int(m.split('_')[1])
    model.load_state_dict(torch.load(load_dir))
    learned_input = None
    input_['hidden'] = toFloatTensor(hidden,0)
    for _ in range(6):
        out = model.forward(input_)
        input_['hidden'] = out['hidden']
        out['prob'] = F.softmax(out['policy'], dim = 1)
        res = torch.cat((input_['hidden'][0], out['prob']), dim=1)
        if learned_input is None:
            learned_input = res
        else:
            learned_input = torch.cat((learned_input, res), dim=0)
    l_loss = model.learned_loss(learned_input)
    i_grad = torch.autograd.grad(
        l_loss,
        model.parameters(),
        #create_graph=True,
        #retain_graph=True,
        allow_unused=True,
        )
    j = 0
    for name,_ in model.named_parameters():
        if i_grad[j] is not None and "exclude" not in name and "ll" not in name:
            #data = torch.norm(i_grad[j]*0.0001,2).cpu().item()  #l2
            #data = torch.mean(i_grad[j]*0.0001).cpu().item()
            var_, mean_ = torch.var_mean(i_grad[j]*0.0001)
            logger.add_scalar(name+'_var',var_.cpu().item(),frame)
            logger.add_scalar(name+'_mean',mean_.cpu().item(),frame)
        j += 1
        
bar.close()
logger.close()