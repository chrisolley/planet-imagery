import torch.optim as optim

def create_optimizer(model, lr0):
    param_groups = [list(model.groups[i].parameters()) for i in range(3)]
    params = [{'params': p, 'lr': lr} for p, lr in zip(param_groups, [lr0/9, lr0/3, lr0] )]
    return optim.Adam(params)

def lr_scheduler(optimizer, epoch, factor, init_lr=0.01, lr_decay_epoch=7):
    """Decay learning rate by a factor every lr_decay_epoch epochs."""
    lr = init_lr * (factor**(epoch // lr_decay_epoch))
    
    if epoch % lr_decay_epoch == 0:
        print("Setting base LR to %.8f" % lr)
    return lr