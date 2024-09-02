import torch
#嵌入层，batch_size,block_size之间的关系是什么？它们分别代表什么？
batch_size=32#how many independent sequences will we process in parallel? (Batch Dimension)
block_size=8#what is the maximum context length for prediction (Time Dimension)
max_iters=1000
eval_interval = 100#这个参数用于？
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters=200#迭代次数
embedding_dim=64
num_heads=6
num_layers=6
dropout=0.2#每个前向、后向，超过20%的所有这些中间计算都被禁用并降至0
n_embd=12#number of embedding (channel)
