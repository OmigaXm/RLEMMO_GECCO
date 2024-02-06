import numpy as np

class MyOptions:
    def __init__(self):
        super(MyOptions, self).__init__()
        self.FF = 0.5
        self.CR = 0.9
        self.popsize = 100 # todo: 不同函数不同？
        self.n_action = 5
        self.save_dir = 'forsave/'
        
        self.global_dim = 5
        self.local_dim = 5
        self.ind_dim = 12
        self.node_dim = int(self.global_dim + self.local_dim + self.ind_dim)
        self.embedding_dim = 64
        self.hidden_dim = 64
        self.encoder_head_num = 4
        self.n_encode_layers = 1
        self.normalization = 'layer'
        
        self.hidden_dim1_actor = 16
        self.hidden_dim2_actor = 8
        self.output_dim = self.n_action
        self.no_attn = False
        self.test = False
        self.hidden_dim1_critic = 16
        self.hidden_dim2_critic = 8
        
        self.use_cuda = True
        self.device = 'cuda'
        self.seed = 12

        self.batch_size = 4
        self.val_batch_size = 1

        self.training_dataset = None
        self.val_dataset = None
        self.test_dataset = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        self.resume = False
        self.epoch_start = 0
        self.epoch_end = 100
        self.max_learning_step = np.inf
        self.distributed = False
        self.no_saving = False
        self.checkpoint_epochs = 1
        self.update_best_model_epochs = 1
        self.gamma = 0.99
        self.max_grad_norm = 0.1

        self.lr_model = 5e-4
        self.lr_decay = 0.9862327
        self.n_step = 10
        self.K_epochs = 3
        self.eps_clip = 0.1
        
        self.no_tb = False
        self.log_step = 1000
        self.per_eval_time = 1

        self.min_samples = 3
        self.k_neis = 4  


