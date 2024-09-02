import torch


# hyperparameters
import config as mp
# model architecture
import model_arch as ma


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


# data loading
def get_batch(split):
    # generate a small batch of data of input x and targets y
    data= train_data if split=='train' else val_data
    block_size = mp.block_size
    batch_size = mp.batch_size

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
    eval_iters=mp.eval_iters

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

# Create Model
model = ma.nanoGPT()
# Create Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=mp.learning_rate)


for iter in range(mp.max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % mp.eval_interval==0 or iter == mp.max_iters-1:
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