
import warnings
import torch
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
from ppo.actor_network import Actor
from ppo.critic_network import Critic
from utils import *
import copy
from ppo.train_utils import *
from tensorboard_logger import Logger as TbLogger
from Env.env import Env
from options import MyOptions
from cec2013.cec2013 import *
from Env.problem import *
from utils import set_seed
from scipy.spatial import distance
import time
import random
import math

import os, sys
import torch.nn as nn
import datetime
from ppo.ppo import PPO

torch.set_num_threads(2)
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# interface for rollout
def test(dataloader,opts,agent=None):
    print('agent test')
    if agent:
        agent.eval()
        print('has agent')
    else:
        print('no agent!!!!!')
        assert False
    test_time = 50
    
    # set the same random seed before rollout for the sake of fairness
    set_seed(opts.seed)
    rollout_bar = tqdm(total = len(dataloader) / opts.val_batch_size * test_time, leave=True)
    rollout_bar.set_description(f'(test)')
    for bat_id in range(len(dataloader) // opts.val_batch_size):
        funcs = []
        for func_id in range(opts.val_batch_size):
            funcs.append(CEC2013(dataloader[bat_id * opts.val_batch_size + func_id]))
        problem = Problem(funcs, opts)

        peak_ratio = np.zeros((opts.val_batch_size, 5))
        succ_rate = np.zeros((opts.val_batch_size, 5))
        
        # list to store the final optimization result
        for i in range(test_time):
            # reset the backbone algorithm
            state=problem.reset()
            is_end = np.zeros(opts.val_batch_size, dtype='bool')
            
            if agent:
                state=torch.FloatTensor(state).to(opts.device)

            while True:
                
                if agent:
                    # if RL_agent is provided, the action is from the agent.actor
                    action,_,_to_critic = agent.actor(state,to_critic=True)
                    action=action.cpu().numpy()
                else:
                    assert False
                    # action=np.random.randint(0, opts.n_action, size=(opts.val_batch_size, opts.popsize))
                
                # put action into environment(backbone algorithm to be specific)
                next_state, rewards,is_end = problem.step(action)
                state=next_state.copy()
                if agent:
                    state=torch.FloatTensor(state).to(opts.device)
                    
                # store the rollout cost history
                if is_end.all():
                    # store the final cost in the end of optimization process
                    for env_id in range(opts.val_batch_size):
                        solu = np.array(problem.env_list[env_id].archive_pos)
                        # solu = np.vstack((solu, problem.env_list[env_id].cur_GA.population.copy()))
                        solu_val = np.array(problem.env_list[env_id].archive_val)
                        assert solu.shape[0] == solu_val.shape[0]
                        my_pkn = problem.env_list[env_id].func.get_no_goptima()
                        nfp = np.zeros(5)
                        accuracy = [0.1, 0.01, 0.001, 0.0001, 0.00001]
                        if len(solu) > 0:
                            for level in range(5):
                                final_archive = []
                                final_archive_val = []
                                for archive_choice in range(len(solu)):
                                    if (solu_val[archive_choice] - problem.env_list[env_id].cur_GA.best_so_far) <= accuracy[level]:
                                        final_archive.append(solu[archive_choice].copy())
                                        final_archive_val.append(solu_val[archive_choice]) 
                                if len(final_archive) > 0:
                                    true_op = problem.env_list[env_id].func.get_fitness_goptima()
                                    nfp[level], _ = how_many_goptima(np.array(final_archive), problem.env_list[env_id].func, accuracy[level], -(np.array(final_archive_val) - true_op))
                                    peak_ratio[env_id][level] += nfp[level] / my_pkn
                                    if nfp[level] >= my_pkn:
                                        succ_rate[env_id][level] += 1
                    break
            rollout_bar.update(1)

        peak_ratio /= (test_time + 0.0)
        succ_rate /= (test_time + 0.0)
        for sub_pro in range(opts.val_batch_size):
            sub_pro_id = dataloader[bat_id * opts.val_batch_size + sub_pro]
            print('problem: ', sub_pro_id, ' PR: ', peak_ratio[sub_pro], ' SR: ', succ_rate[sub_pro])

def main():
    opts = MyOptions()
    ppo = PPO(opts)
    ppo.load('forsave/epoch-60.pt')
    print("begin test")
    warnings.filterwarnings("ignore")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(opts.seed)

    test_dataloader= [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    print(test_dataloader)

    test(test_dataloader,opts,ppo)





        
if __name__ == '__main__':
    main()