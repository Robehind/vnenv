import torch
def gpuify(tensor, gpu_id):
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            tensor = tensor.cuda()
    return tensor

def toFloatTensor(x, gpu_id):
    """ Convers x to a FloatTensor and puts on GPU.
        Support input as list or tuple"""
    if isinstance(x, tuple):
        return tuple(gpuify(torch.FloatTensor(x1), gpu_id) for x1 in x)
    
    if isinstance(x, list): 
        return list(gpuify(torch.FloatTensor(x1), gpu_id) for x1 in x)

    return gpuify(torch.FloatTensor(x), gpu_id)