import torch

def copy_gradient(global_model, local_model):
    for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
        global_param._grad = local_param.grad.clone().cpu()
