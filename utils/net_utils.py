import torch
def gpuify(tensor, gpu_id):
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            tensor = tensor.cuda()
    return tensor

def toFloatTensor(x, gpu_id):
    """ Convers x to a FloatTensor and puts on GPU. """
    return gpuify(torch.FloatTensor(x), gpu_id)