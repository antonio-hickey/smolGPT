import torch 


batch_size = 64
block_size = 256
max_iters = 5000
eval_iters = 200
eval_inter = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embds = 384
n_head = 6
n_layers = 6
dropout = 0.2