import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size=32
block_size=8
max_iters=3000
eval_interval = 100#这个参数用于？
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters=200#迭代次数
embedding_dim=64
num_heads=4
num_layers=4
dropout_rate=0.0

# Set the random seed for reproduciblity
torch.manual_seed(1337)

# Load dataset
with open('data/shakespeare_input.txt','r', encoding='utf-8') as f:
    text=f.read()

# Character mappings(简单分词器实现)
'''
将所有出现过的字符重排序，映射成数字，存储在数组中（哈希表实现）
'''
chars=sorted(list(set(text)))#去重和排序
vocab_size=len(chars)
stoi={ch:i for i,ch in enumerate(chars)}
itos={i:ch for i,ch in enumerate(chars)}

encode = lambda s:[stoi[c] for c in s]#编码器实现
decode = lambda l:''.join([itos[i] for i in l])#解码器实现

# Prepare train and valid data splits(训练集，验证集划分)
data = torch.tensor(encode(text), dtype=torch.long)
train_size = int(0.9*len(data))
train_data = data[:train_size]#可复用划分技巧
val_data = data[train_data:]


# data loading
def get_batch(split):
    # generate a small batch of data of input x and targets y
    data= train_data if split=='train' else val_data

    # 在合理范围内，随机生成batch_size个sequence(每个sequence长度为block_size)
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])#为什么y是i+1:block_size+1
    #x,y = x.to(device), y.to(device)#将数据放到GPU上
    return x,y

@torch.no_grad()#上下文管理器,告诉PyTorch，我们不会调用.backward来处理此函数内部发生的所有事情（没懂）
def estimate_loss():
    out={}
    model.eval()#评估阶段

    # 通过平均计算batch平均，减少loss计算的噪音
    for split in['train','val']:
        losses=torch.zeros(eval_iters)#待检验，torch.zeros(..)会生成什么->一个长为eval_iters,值全为0的数组
        for k in range(eval_iters):
            X, y = get_batch(split)
            logits, loss = model(X, y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()#训练阶段
    return out

# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # token embedding table -> 标记嵌入表,例如上述tensor中[[0,24,43,...],[...]..];
        # 值为24的元素将在嵌入表中找到第24行，然后呢...?
        # nn.Embedding -> 一个非常薄的包装器，基本上是一个形状为vocab_size*vocab_size的张量
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):  # 这是一个封装起来的强制函数
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)  # （B,T,C),在本例中,B(batch)批次为4，T(time)为8，C(chanel)为vocabSize即65
        if targets is None:
            loss = None
        else:
            # 利用负对数似然损失(negative log likelihood loss)衡量损失或预测质量，它也在PyTorch中以交叉熵的名称实现
            # 但是因为Pytorch对于cross_engropy的入参格式有要求，所以这里需要变换下维度
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # 将原先的4*8*65的三维数组扁平化至32*65
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

        # 但是这里没有调用logits函数啊，怎么计算的呢

        # 如果无视Batch的话，每个矩阵是一个8*65的矩阵。好像有点感觉，8代表时间维度（为什么叫做时间维度？），65代表
        # 总之得到的结果是每个senquece对于自己下一个词的预测分数？
        # 然后现在需要一个方法来衡量损失

    def generate(self, idx, max_new_tokens):  # generate函数的主要目的是利用训练好的模型进行文本生成
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)  # 为什么这里会调用
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # sample form the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


