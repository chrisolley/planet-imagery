import logging
import torch.optim as optim

logger = logging.getLogger("Planet-Imagery")

# def create_optimizer(model, lr0):
#     param_groups = [list(model.groups[i].parameters()) for i in range(3)]
#     params = [{'params': p, 'lr': lr} for p, lr in zip(param_groups, [lr0/9, lr0/3, lr0] )]
#     return optim.Adam(params)

def create_optimizer(model, optimizer, lr0, diff_lr_factors):
    '''
    Creates an optimizer for a NN segmented into groups, with a differential
    learning rate across groups according to a multiplicative factor for each
    group given by group_lrs
    '''
    n_groups = len(diff_lr_factors)
    param_groups = [list(model.groups[i].parameters()) for i in range(n_groups)]
    params = [{'params': p, 'lr': lr0/diff_factor} for p, diff_factor in zip(param_groups, diff_lr_factors)]
    return optimizer(params)

def lr_scheduler(epoch, factor, init_lr=0.01, lr_decay_epoch=7):
    '''
    Decay learning rate by a factor every lr_decay_epoch epochs.
    '''
    lr = init_lr * (factor**(epoch // lr_decay_epoch))
    if epoch % lr_decay_epoch == 0:
        logger.info("Setting base LR to %.8f" % lr)
    return lr
