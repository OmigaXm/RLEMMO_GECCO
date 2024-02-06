import numpy as np
import torch
import os, sys
import time
import torch.nn as nn
import datetime
from options import MyOptions
from tensorboard_logger import Logger as TbLogger
from ppo.ppo import PPO
from utils import *

torch.set_num_threads(2)
os.environ["CUDA_VISIBLE_DEVICES"]="0"


### 最大值问题

def main():
    opts = MyOptions()
    set_seed(opts.seed)
    # 随机划分训练集和测试集
    funcs_1 = [1,2,3] # 2
    funcs_2 = [4,5,6,7,10,11,12,13] # 5
    funcs_3 = [8,9,14,15] # 2
    funcs_4 = [16,17] # 1
    funcs_5 = [18,19] # 1
    funcs_6 = [20] # 1
    random.shuffle(funcs_1)
    random.shuffle(funcs_2)
    random.shuffle(funcs_3)
    random.shuffle(funcs_4)
    random.shuffle(funcs_5)
    opts.training_dataset = funcs_1[:2].copy()
    opts.training_dataset.extend(funcs_2[:5].copy())
    opts.training_dataset.extend(funcs_3[:2].copy())
    opts.training_dataset.extend(funcs_4[:1].copy())
    opts.training_dataset.extend(funcs_5[:1].copy())
    opts.training_dataset.extend(funcs_6[:1].copy())
    opts.val_dataset = funcs_1[2:].copy()
    opts.val_dataset.extend(funcs_2[5:].copy())
    opts.val_dataset.extend(funcs_3[2:].copy())
    opts.val_dataset.extend(funcs_4[1:].copy())
    opts.val_dataset.extend(funcs_5[1:].copy())
    # RLEMMO论文中的划分结果
    # opts.training_dataset = np.array([1, 3, 4, 6, 8, 9, 10, 12, 13, 17, 19, 20])
    # opts.val_dataset =np.array([2, 5, 7, 11, 14, 15, 16, 18])
    
    tag = datetime.datetime.now()
    tag = tag.strftime("%Y_%m_%d_%H_%M_%S")

    savepath = opts.save_dir
    os.makedirs(savepath, exist_ok=True)

    
    ppo = PPO(opts)

    path = 'log/log_'+tag+'/'
    os.makedirs(path, exist_ok=True)
    tb_logger = TbLogger(os.path.join(path))

    ppo.start_training(tb_logger)



        
if __name__ == '__main__':
    main()

