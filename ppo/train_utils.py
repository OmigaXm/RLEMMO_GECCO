from utils import set_seed
from scipy.spatial import distance
import numpy as np
import torch
from tqdm import tqdm
from utils import *
import os
import time
import random
import math
import copy
from ppo.actor_network import Actor
from ppo.critic_network import Critic
from Env.env import Env
from options import MyOptions
from cec2013.cec2013 import *
from Env.problem import *

# memory for recording transition during training process
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]


def lr_sd(epoch, opts):
    return opts.lr_decay ** epoch



def train_batch(
        rank,
        problem,
        agent,
        pre_step,
        tb_logger,
        opts, epoch):
    
    # setup
    agent.train()
    memory = Memory()
    extra_log_step = 0

    # initial instances and solutions
    state =problem.reset()
    state=torch.FloatTensor(state).to(opts.device)
    # state=torch.where(torch.isnan(state),torch.zeros_like(state),state)
    assert state.shape == (opts.batch_size, opts.popsize, opts.node_dim)    
    
    # params for training
    gamma = opts.gamma
    n_step = opts.n_step
    
    K_epochs = opts.K_epochs
    eps_clip = opts.eps_clip
    
    t = 0
    # initial_cost = obj
    done=False
    
    # sample trajectory
    while not done:
        t_s = t
        entropy = []
        bl_val_detached = []
        bl_val = []

        # accumulate transition
        while t - t_s < n_step :  
            
            memory.states.append(state.clone())
            action, log_lh,_to_critic,  entro_p  = agent.actor(state,
                                                    require_entropy = True,
                                                    to_critic=True
                                                    )
            

            memory.actions.append(action.clone())
            memory.logprobs.append(log_lh)
            action=action.cpu().numpy()
            assert action.shape == (opts.batch_size, opts.popsize)

            entropy.append(entro_p.detach().cpu())

            baseline_val_detached, baseline_val = agent.critic(_to_critic)
            bl_val_detached.append(baseline_val_detached)
            bl_val.append(baseline_val)


            # state transient
            next_state,rewards,is_end = problem.step(action)

            if epoch % 5 == 0:
                avg_reward = []
                avg_rewardp1 = []
                avg_rewardp2 = []
                avg_rewardp3 = []
                avg_bsf = []
                avg_num_opti = []
                for sub_env in problem.env_list:
                    avg_reward.append(sub_env.rewards)
                    avg_bsf.append(sub_env.cur_GA.best_so_far)
                    avg_rewardp1.append(sub_env.rewardp1)
                    avg_rewardp2.append(sub_env.rewardp2)
                    avg_rewardp3.append(sub_env.rewardp3)
                    avg_num_opti.append(np.max(sub_env.labels) + 1)
                tb_logger.log_value('train_extra/'+str(epoch)+'_avg_reward_', np.average(avg_reward), extra_log_step)
                tb_logger.log_value('train_extra/'+str(epoch)+'_avg_rewardp1_', np.average(avg_rewardp1), extra_log_step)
                tb_logger.log_value('train_extra/'+str(epoch)+'_avg_rewardp2_', np.average(avg_rewardp2), extra_log_step)
                tb_logger.log_value('train_extra/'+str(epoch)+'_avg_rewardp3_', np.average(avg_rewardp3), extra_log_step)
                tb_logger.log_value('train_extra/'+str(epoch)+'_avg_bsf_', np.average(avg_bsf), extra_log_step)
                tb_logger.log_value('train_extra/'+str(epoch)+'_avg_num_opti_', np.average(avg_num_opti), extra_log_step)
                extra_log_step += 1

            memory.rewards.append(torch.FloatTensor(rewards).to(opts.device))
            # print('step:{},max_reward:{}'.format(t,torch.max(rewards)))

            # next
            t = t + 1
            state=torch.FloatTensor(next_state).to(opts.device)
            # state=torch.where(torch.isnan(state),torch.zeros_like(state),state)

            
            if is_end.any():
                done=True
                break
        
        # store info
        t_time = t - t_s

        assert len(memory.rewards) == len(memory.actions) == len(memory.states) == len(memory.logprobs)

        # begin update
        # 如果是madde这里的action就不能直接stack
        old_actions = torch.stack(memory.actions)
        old_states = torch.stack(memory.states).detach() #.view(t_time, bs, ps, dim_f)
        old_logprobs = torch.stack(memory.logprobs).detach().view(-1)

        # Optimize PPO policy for K mini-epochs:
        old_value = None
        for _k in range(K_epochs):
            if _k == 0:
                logprobs = memory.logprobs

            else:
                # Evaluating old actions and values :
                logprobs = []
                entropy = []
                bl_val_detached = []
                bl_val = []

                for tt in range(t_time):

                    # get new action_prob
                    _, log_p,_to_critic,  entro_p = agent.actor(old_states[tt],
                                                     fixed_action = old_actions[tt],
                                                     require_entropy = True,# take same action
                                                     to_critic = True
                                                     )

                    logprobs.append(log_p)
                    entropy.append(entro_p.detach().cpu())

                    baseline_val_detached, baseline_val = agent.critic(_to_critic)

                    bl_val_detached.append(baseline_val_detached)
                    bl_val.append(baseline_val)

            logprobs = torch.stack(logprobs).view(-1)
            entropy = torch.stack(entropy).view(-1)
            bl_val_detached = torch.stack(bl_val_detached).view(-1)
            bl_val = torch.stack(bl_val).view(-1)


            # get traget value for critic
            Reward = []
            reward_reversed = memory.rewards[::-1]
            # get next value
            R = agent.critic(agent.actor(state,only_critic = True))[0]

            critic_output=R.clone()
            for r in range(len(reward_reversed)):
                R = R * gamma + reward_reversed[r]
                Reward.append(R)
            # clip the target:
            Reward = torch.stack(Reward[::-1], 0)
            Reward = Reward.view(-1)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = Reward - bl_val_detached

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages
            reinforce_loss = -torch.min(surr1, surr2).mean()

            # define baseline loss
            if old_value is None:
                baseline_loss = ((bl_val - Reward) ** 2).mean()
                old_value = bl_val.detach()
            else:
                vpredclipped = old_value + torch.clamp(bl_val - old_value, - eps_clip, eps_clip)
                v_max = torch.max(((bl_val - Reward) ** 2), ((vpredclipped - Reward) ** 2))
                baseline_loss = v_max.mean()

            # check K-L divergence (for logging only)
            approx_kl_divergence = (.5 * (old_logprobs.detach() - logprobs) ** 2).mean().detach()
            approx_kl_divergence[torch.isinf(approx_kl_divergence)] = 0
            # calculate loss
            loss = baseline_loss + reinforce_loss

            # update gradient step
            agent.optimizer.zero_grad()
            loss.backward()

            # Clip gradient norm and get (clipped) gradient norms for logging
            current_step = int(pre_step + t//n_step * K_epochs  + _k)
            grad_norms = clip_grad_norms(agent.optimizer.param_groups, opts.max_grad_norm)

            # perform gradient descent
            agent.optimizer.step()

            # Logging to tensorboard
            if(not opts.no_tb) and rank == 0:
                if current_step % int(opts.log_step) == 0:
                    log_to_tb_train(tb_logger, grad_norms, entropy, approx_kl_divergence, reinforce_loss, baseline_loss, loss,current_step)

            # end update
        

        memory.clear_memory()

    # return learning steps
    return ( t // n_step + 1) * K_epochs


# interface for rollout
def rollout(dataloader,opts,agent=None,tb_logger=None, epoch_id=0):

    if agent:
        agent.eval()
    else:
        print('!!!!!!!!!!!!!! no agent !!!!!!!!!!!!!!!!!!!!1')

    peak_ratio = np.zeros(5)
    succ_rate = np.zeros(5)
    all_return = 0
    all_returnp1 = 0
    all_returnp2 = 0
    all_returnp3 = 0
    all_actions = np.zeros(opts.n_action)
    all_gbest = []
    all_len_archive = 0
    all_reini_count = 0
    
    
    # set the same random seed before rollout for the sake of fairness
    set_seed(opts.seed)
    rollout_bar = tqdm(total = len(dataloader) / opts.val_batch_size * opts.per_eval_time, leave=True)
    rollout_bar.set_description(f'(Validating) Epoch [{epoch_id}/{opts.epoch_end}]')
    for bat_id in range(len(dataloader) // opts.val_batch_size):
        # see if there is agent to aid the backbone
        origin=True
        if agent:
            origin=False

        funcs = []
        for func_id in range(opts.val_batch_size):
            funcs.append(CEC2013(dataloader[bat_id * opts.val_batch_size + func_id]))
        problem = Problem(funcs, opts)

        # list to store the final optimization result
        for i in range(opts.per_eval_time):
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
                    action=np.random.randint(0, opts.n_action, size=(opts.val_batch_size, problem.popsize))
                
                # put action into environment(backbone algorithm to be specific)
                next_state, rewards,is_end = problem.step(action)
                state=next_state.copy()
                if agent:
                    state=torch.FloatTensor(state).to(opts.device)
                    
                # store the rollout cost history
                if is_end.all():
                    # store the final cost in the end of optimization process
                    for env_id in range(opts.val_batch_size):
                        all_gbest.append(problem.env_list[env_id].cur_GA.best_so_far)
                        all_return += problem.env_list[env_id].returns
                        all_returnp1 += problem.env_list[env_id].returnsp1
                        all_returnp2 += problem.env_list[env_id].returnsp2
                        all_returnp3 += problem.env_list[env_id].returnsp3
                        all_actions += problem.env_list[env_id].chosen_actions
                        all_len_archive += len(problem.env_list[env_id].archive_pos)
                        all_reini_count += problem.env_list[env_id].reini_count

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
                                    peak_ratio[level] += nfp[level] / my_pkn
                                    if nfp[level] >= my_pkn:
                                        succ_rate[level] += 1
                    break
            rollout_bar.update(1)

    for i in range(5):
        peak_ratio[i] /= (opts.per_eval_time * len(dataloader) + 0.0)
        succ_rate[i] /= (opts.per_eval_time * len(dataloader) + 0.0)

    avg_return = all_return / (len(dataloader) * opts.per_eval_time)
    avg_returnp1 = all_returnp1 / (len(dataloader) * opts.per_eval_time)
    avg_returnp2 = all_returnp2 / (len(dataloader) * opts.per_eval_time)
    avg_returnp3 = all_returnp3 / (len(dataloader) * opts.per_eval_time)
    avg_gbest = np.average(all_gbest)
    max_gbest = np.max(all_gbest)
    min_gbest = np.min(all_gbest)
    avg_actions = all_actions / np.sum(all_actions)
    avg_len_archive = all_len_archive / (len(dataloader) * opts.per_eval_time)
    avg_reini_count = all_reini_count / (len(dataloader) * opts.per_eval_time)


    # log to tensorboard if needed
    if tb_logger:
        log_to_val(tb_logger,epoch_id, avg_return, avg_actions, avg_gbest,peak_ratio,succ_rate, avg_len_archive, avg_reini_count, max_gbest, min_gbest, avg_returnp1, avg_returnp2,avg_returnp3)
    
    # calculate and return the mean and std of final cost
    return avg_gbest.item(), peak_ratio, succ_rate
    

