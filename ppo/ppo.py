import os
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



class PPO:
    def __init__(self, opts):

        # figure out the options
        self.opts = opts
        # figure out the actor network
        self.actor = Actor(
            embedding_dim = opts.embedding_dim,
            hidden_dim = opts.hidden_dim,
            n_heads_actor = opts.encoder_head_num,
            n_layers = opts.n_encode_layers,
            normalization = opts.normalization,
            node_dim=opts.node_dim,
            hidden_dim1=opts.hidden_dim1_actor,
            hidden_dim2=opts.hidden_dim2_actor,
            output_dim=opts.output_dim,
            global_dim = opts.global_dim,
            local_dim = opts.local_dim,
            ind_dim = opts.ind_dim,
            no_attn=opts.no_attn
        )
        
        if not opts.test:
            # for the sake of ablation study, figure out the input_dim for critic according to setting
            input_critic=opts.embedding_dim

            # figure out the critic network
            self.critic = Critic(
                input_dim = input_critic,
                hidden_dim1 = opts.hidden_dim1_critic,
                hidden_dim2 = opts.hidden_dim2_critic,
            )

            # figure out the optimizer
            self.optimizer = torch.optim.Adam(
                [{'params': self.actor.parameters(), 'lr': opts.lr_model}] +
                [{'params': self.critic.parameters(), 'lr': opts.lr_model}])
            # figure out the lr schedule
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, opts.lr_decay, last_epoch=-1,)

        if opts.use_cuda:
            # move to cuda
            self.actor.to(opts.device)
            if not opts.test:
                self.critic.to(opts.device)


    # load model from load_path
    def load(self, load_path):

        assert load_path is not None
        load_data = torch_load_cpu(load_path)

        # load data for actor
        model_actor = get_inner_model(self.actor)
        model_actor.load_state_dict({**model_actor.state_dict(), **load_data.get('actor', {})})

        if not self.opts.test:
            # load data for critic
            model_critic = get_inner_model(self.critic)
            model_critic.load_state_dict({**model_critic.state_dict(), **load_data.get('critic', {})})
            # load data for optimizer
            self.optimizer.load_state_dict(load_data['optimizer'])
            # load data for torch and cuda
            torch.set_rng_state(load_data['rng_state'])
            if self.opts.use_cuda:
                torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # done
        print(' [*] Loading data from {}'.format(load_path))

    # save trained model
    def save(self, epoch):
        print('Saving model and state...')
        torch.save(
            {
                'actor': get_inner_model(self.actor).state_dict(),
                'critic': get_inner_model(self.critic).state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
            },
            os.path.join(self.opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    # change working mode to evaling
    def eval(self):
        torch.set_grad_enabled(False)  ##
        self.actor.eval()
        if not self.opts.test: self.critic.eval()

    # change working mode to training
    def train(self):
        torch.set_grad_enabled(True)  ##
        self.actor.train()
        if not self.opts.test: self.critic.train()


    def start_training(self, tb_logger):
        train(0, self, tb_logger)

# inference for training
def train(rank, agent, tb_logger):  
    print("begin training")
    opts = agent.opts
    warnings.filterwarnings("ignore")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(opts.seed)

    # move optimizer's data onto chosen device
    for state in agent.optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(opts.device)


    # generatate the train_dataset and test_dataset
    training_dataloader = opts.training_dataset
    print('train_dataload', training_dataloader)
    test_dataloader=opts.val_dataset
    print('val_dataload', test_dataloader  )


    best_epoch=None
    best_peak_ratio=None
    best_epoch_list=[]
    pre_step=0

    # modify the learning ray after resume(if needed)
    if opts.resume:
        for e in range(opts.epoch_start):
            agent.lr_scheduler.step(e)

    stop_training=False

    # rollout in the 0 epoch model to get the baseline data
    ini_avg_gbest, ini_peak_ratio, ini_succ_rate =rollout(test_dataloader,opts,agent,tb_logger,-1)

    # Start the actual training loop
    for epoch in range(opts.epoch_start, opts.epoch_end):
        # Training mode
        # set_seed(opts.seed)
        agent.train()
        agent.lr_scheduler.step(epoch)

        # logging
        if rank == 0:
            print('\n\n')
            print("|",format(f" Training epoch {epoch} ","*^60"),"|")
            print("Training with actor lr={:.3e} critic lr={:.3e}".format(agent.optimizer.param_groups[0]['lr'],
                                                                                     agent.optimizer.param_groups[1]['lr']) , flush=True)

        # start training
        # episode_step=8500
        pbar = tqdm(range(len(training_dataloader)), leave=True)
        pbar.set_description(f'(Training) Epoch [{epoch}/{opts.epoch_end}]')

        for batch_id in pbar:
            funcs = []
            for func_id in range(opts.batch_size):
                funcs.append(CEC2013(training_dataloader[batch_id]))
            problem = Problem(funcs, opts)

            batch_step=train_batch(rank,
                                problem,
                                agent,
                                pre_step,
                                tb_logger,
                                opts, epoch)
            pre_step += batch_step
            # see if the learning step reach the max_learning_step, if so, stop training
            if pre_step>=opts.max_learning_step:
                stop_training=True
                break
        pbar.close()

        # save new model after one epoch
        if rank == 0 and not opts.distributed:
            if not opts.no_saving and (( opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or \
                                       epoch == opts.epoch_end - 1): agent.save(epoch)
                                       
        if (epoch-opts.epoch_start) % opts.update_best_model_epochs==0 or epoch == opts.epoch_end-1:
            # validate the new model
            avg_gbest, peak_ratio, succ_rate=rollout(test_dataloader,opts,agent,tb_logger,epoch)
            if epoch==opts.epoch_start:
                best_peak_ratio=peak_ratio[3]
                best_epoch=epoch
            elif peak_ratio[3]>best_peak_ratio:
                best_peak_ratio=peak_ratio[3]
                best_epoch=epoch
            best_epoch_list.append(best_epoch)


        # logging
        print('current_epoch:{}, best_epoch:{}'.format(epoch,best_epoch))
        print('best_epoch_list:{}'.format(best_epoch_list))
        print(f'init_peak_ratio:{ini_peak_ratio}')
        print(f'cur_peak_ratio:{peak_ratio}')
        print(f'best_peak_ratio:{best_peak_ratio}')
        
        if stop_training:
            print('Have reached the maximum learning steps')
            break
    print(best_epoch_list)














