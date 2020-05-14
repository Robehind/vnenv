import torch

def copy_gradient(global_model, local_model):
    for global_param, local_param in zip(filter(lambda p: p.requires_grad, global_model.parameters()),
                                       filter(lambda p: p.requires_grad, local_model.parameters())):
            #if local_param.grad is None:
                #global_param._grad = torch.zeros(global_param.shape)
            #else:
            global_param._grad = local_param.grad.clone().cpu()

def transfer_gradient_from_player_to_shared(player, shared_model, gpu_id):
    """ Transfer the gradient from the player's model to the shared model
        and step """
    for param, shared_param in zip(
        player.model.parameters(), shared_model.parameters()
    ):
        if shared_param.requires_grad:
            if param.grad is None:
                shared_param._grad = torch.zeros(shared_param.shape)
            elif gpu_id < 0:
                shared_param._grad = param.grad
            else:
                shared_param._grad = param.grad.cpu()