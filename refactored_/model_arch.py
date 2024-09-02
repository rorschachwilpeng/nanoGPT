import torch.nn as nn
from torch.nn import functional as F
import torch
import config as mp

n_embd=mp.n_embd
block_size=mp.block_size
dropout=mp.dropout
num_heads=mp.num_heads
num_layers=mp.num_layers

###Stupid Refactoring
# Set the random seed for reproduciblity
torch.manual_seed(1337)
# Load dataset
with open('../data/shakespeare_input.txt','r', encoding='utf-8') as f:
    text=f.read()

# Character mappings(简单分词器实现)
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
###


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
        self.ffwd = FeedForwad(n_embd)#Feed Forward
        self.ln1 = nn.LayerNorm(n_embd)#Layer Normalization
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # （此处实现和attention is all you need论文中不太一样；论文中LayerNorm在sa和前馈层之后，但是如今
        #  业界流行将LayerNorm放在sa和前馈层执行 ---> why?
        x = x + self.sa(self.ln1(x))#残差连接，将x拆分成初始x（利用自注意力通信）和分叉x（原始输入）
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class nanoGPT(nn.Module):
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
        pos_emb = self.positional_embedding_table(torch.arange(T,device=mp.device))#位置编码
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
