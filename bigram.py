import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
#batch代表的难道不是多个句子的组合吗
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

# batch_size=64#how many independent sequences will we process in parallel?
# block_size=256#what is the maximum context length for prediction
# max_iters=100
# eval_interval = 500#这个参数用于？
# learning_rate = 3e-4
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters=200#迭代次数
# embedding_dim=384
# num_heads=6
# num_layers=6
# dropout=0.2#每个前向、后向，超过20%的所有这些中间计算都被禁用并降至0
# n_embd=384#number of embedding

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
val_data = data[train_size:]


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

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size,bias=False)
        self.query = nn.Linear(n_embd, head_size,bias=False)
        self.value = nn.Linear(n_embd, head_size,bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)#(B,T,C)
        q = self.query(x)#(B,T,C)
        # compute attention scores("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 #(B,T,C) @ (B,C,T) ---> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))#(B,T,T)
        wei = F.softmax(wei, dim=-1) #(B,T,T)
        wei = self.dropout(wei)
        # perform weight aggregation of the values
        v = self.value(x) #(B,T,C)
        out = wei @ v #(B,T,T) @ (B,T,C) -----> (B,T,C)
        return out

class FeedForwad(nn.Module):
    """ a simple linear layer followed by a non-linearity"""

    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)#线性变换 ？
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        #out = self.proj(out)
        return out


class Block(nn.Module):
    """ Transformer block: communication followed by computation"""
    #Q->这个block在架构中对应哪个部分？
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimensions,
        # n_head:the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head#why
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForwad(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)#Layer Normalization
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # （此处实现和attention is all you need论文中不太一样；论文中LayerNorm在sa和前馈层之后，但是如今
        #  业界流行将LayerNorm放在sa和前馈层执行 ---> why?
        x = x + self.sa(self.ln1(x))#残差连接，将x拆分成初始x（利用自注意力通信）和分叉x（原始输入）
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # token embedding table -> 标记嵌入表,例如上述tensor中[[0,24,43,...],[...]..];
        # 值为24的元素将在嵌入表中找到第24行，然后呢...?
        # nn.Embedding -> 一个非常薄的包装器，基本上是一个形状为vocab_size*vocab_size的张量
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)#位置编码表 为什么是block_size, n_embd
        # self.blocks = nn.Sequential(
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     nn.LayerNorm(n_embd),
        # )
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(n_embd)#在Transformer的末尾，在解码成词汇的最终线性层之间也有一个层规范
        #self.sa_heads = MultiHeadAttention(4, n_embd/4) # i.e. 4 heads of 8-dimensional self-attention
        self.lm_head = nn.Linear(n_embd, vocab_size)  # linear layer（线性层），从标记嵌入转到对数

    def forward(self, idx, targets=None):  # 这是一个封装起来的强制函数
        B,T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # （B,T,C),在本例中,B(batch)批次为4，T(time)为8，C(chanel)为vocabSize即65
        pos_emb = self.positional_embedding_table(torch.arange(T,device=device))#位置编码
        x = tok_emb + pos_emb
        #x = self.sa_heads(x)#apple mutiple heads of self-attention, (B,T,C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)#(B,T,vocab_size)

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
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)  # 为什么这里会调用
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # sample form the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Create Model
model = BigramLanguageModel()
# Create Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval==0 or iter == max_iters-1:
        losses = estimate_loss()
        print(f"step {iter}: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")
    # sample a batch of data
    xb, yb = get_batch('train')

    ###经典的训练循环
    # evalulate the loss
    logits, loss = model(xb, yb)
    # 将所有梯度归零->why?
    optimizer.zero_grad(set_to_none=True)
    loss.backward()  # 获取所有参数的梯度
    optimizer.step()  # 使用这些梯度来更新我们的参数

# generate from the model
context=torch.zeros((1, 1), dtype=torch.long)
output = decode(model.generate(context, max_new_tokens=100)[0].tolist())
print(output)