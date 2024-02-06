import torch
import math
import random
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
def torch_load_cpu(load_path):
    return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU

def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) or isinstance(model, DDP) else model

def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)

def move_to_cuda(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.cuda(device)

def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for idx, group in enumerate(param_groups)
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_to_tb_train(tb_logger, grad_norms, entropy, approx_kl_divergence, reinforce_loss, baseline_loss, loss,current_step):
    tb_logger.log_value('train/approx_kl_divergence', approx_kl_divergence, current_step) 
    tb_logger.log_value('train/reinforce_loss', reinforce_loss, current_step) 
    tb_logger.log_value('train/baseline_loss', baseline_loss, current_step) 
    tb_logger.log_value('train/loss', loss, current_step) 

def log_to_val(tb_logger,epoch_id, avg_return, avg_actions, avg_gbest,peak_ratio,succ_rate, avg_len_archive, avg_reini_count, max_gbest, min_gbest, avg_returnp1, avg_returnp2,avg_returnp3):
    tb_logger.log_value('test/avg_return', avg_return, epoch_id)
    tb_logger.log_value('test/avg_returnp1', avg_returnp1, epoch_id)
    tb_logger.log_value('test/avg_returnp2', avg_returnp2, epoch_id)
    tb_logger.log_value('test/avg_returnp3', avg_returnp3, epoch_id)
    # log returnp1, returnp2, returnp3
    for act in range(len(avg_actions)):
        tb_logger.log_value('test/ratio_act_'+str(act),avg_actions[act] , epoch_id)
    tb_logger.log_value('test/avg_gbest',avg_gbest , epoch_id)
    tb_logger.log_value('test/max_gbest',max_gbest , epoch_id)
    tb_logger.log_value('test/min_gbest',min_gbest , epoch_id)
    tb_logger.log_value('test/avg_len_archive',avg_len_archive , epoch_id)
    tb_logger.log_value('test/avg_reini_count',avg_reini_count , epoch_id)
    
    for level in range(5):
        tb_logger.log_value('test/peak_ratio_'+str(level), peak_ratio[level], epoch_id)
        tb_logger.log_value('test/succ_rate_'+str(level), succ_rate[level], epoch_id)


