from torch import nn
import torch
from ppo.graph_layers import *
from torch.distributions import Normal
import torch.nn.functional as F
from options import MyOptions
import numpy as np

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class Actor(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 n_heads_actor,
                 n_layers,
                 normalization,
                 node_dim,
                 hidden_dim1,
                 hidden_dim2,
                 output_dim,
                 global_dim,
                 local_dim,
                 ind_dim,
                 no_attn=False
                 ):
        super(Actor, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads_actor = n_heads_actor    
        self.n_layers = n_layers
        self.normalization = normalization
        self.no_attn=no_attn
        self.node_dim = node_dim 
        self.output_dim=output_dim
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.ind_dim = ind_dim

        # figure out the Actor network
        if not self.no_attn:
            print('has attention')
            # figure out the embedder for feature embedding
            self.embedder1 = EmbeddingNet(
                                int(self.global_dim + self.local_dim),
                                int(self.embedding_dim / 2))
            self.embedder2 = EmbeddingNet(
                                self.ind_dim,
                                int(self.embedding_dim / 2))
            # figure out the fully informed encoder
            self.encoder = mySequential(*(
                    MultiHeadEncoder(self.n_heads_actor,
                                    self.embedding_dim,
                                    self.hidden_dim,
                                    self.normalization,)
                for _ in range(self.n_layers))) # stack L layers

            self.softmax_net = MLP(self.embedding_dim ,hidden_dim1,hidden_dim2, output_dim, 0) 
            self.softmax_out = nn.Softmax(dim=-1)
        else:
            assert False
            # print('no attention')
            # self.embedder = EmbeddingNet(
            #                     self.node_dim,
            #                     self.embedding_dim)
            # self.softmax_net = MLP(self.embedding_dim ,hidden_dim1,hidden_dim2, output_dim, 0) 
            # self.softmax_out = nn.Softmax(dim=-1)
        print(self.get_parameter_number())

    def get_parameter_number(self):
        
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def forward(self, x_in,fixed_action = None, require_entropy = False,to_critic=False,only_critic=False):
        if not self.no_attn:
            population_feature=x_in[:,:,:(self.global_dim + self.local_dim)].clone()
            # pass through embedder
            h_em_1 = self.embedder1(population_feature)

            ind_feature=x_in[:,:,(self.global_dim + self.local_dim):].clone()
            # pass through embedder
            h_em_2 = self.embedder2(ind_feature)
            h_em = torch.cat((h_em_1, h_em_2),dim=-1)
            assert h_em.shape == (x_in.shape[0],x_in.shape[1], self.embedding_dim)

            # pass through encoder
            logits = self.encoder(h_em)
        
            # share logits to critic net, where logits is from the decoder output 
            if only_critic:
                return logits  # .view(bs, dim, ps, -1)
            probs = self.softmax_out(self.softmax_net(logits))
        else:
            assert False
            # population_feature=x_in[:,:,:self.node_dim].clone()
            # feature = self.embedder(population_feature)
            # if only_critic:
            #     return feature
            # probs = self.softmax_out(self.softmax_net(feature))

        # don't share the network between actor and critic if there is no attention mechanism
        if self.no_attn:
            assert False
            # _to_critic=feature
        else:
            _to_critic=logits

        policy = torch.distributions.Categorical(probs)
        
        if fixed_action is not None:
            action = torch.tensor(fixed_action)
        else:
            action = policy.sample() 
        assert action.shape == (x_in.shape[0], x_in.shape[1])
        # get log probability
        log_prob=policy.log_prob(action)

        # The log_prob of each instance is summed up, since it is a joint action for a population
            
		# todo: if self.output_dim > 1:
        #    log_prob=torch.sum(torch.sum(log_prob,dim=-1),dim=-1)
        log_prob=torch.sum(log_prob,dim=-1)
        
        if require_entropy:
            entropy = policy.entropy() # for logging only 
            
            out = (action,
                   log_prob,
                   _to_critic if to_critic else None,
                   entropy)
        else:
            out = (action,
                   log_prob,
                   _to_critic if to_critic else None,
                   )
        return out