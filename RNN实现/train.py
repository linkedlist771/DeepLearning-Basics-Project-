import torch
from torch import nn
import DataCollect
import math
import time
import numpy as np
import utils
from model import *
from predict import predict_next_text
import tensorboard
from tensorboard import notebook
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


def grad_clipping(net, theta):
    """清除一下梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def train_net_epoch(net, train_iter, loss, updater, device, use_random_iter):
    """在每一个epoch中进行网络的训练"""
    state, timer = None, utils.Timer()
    start_times = time.time()
    metric = utils.Accumulator(2)  # 记录损失值的和
    index = 0
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 初始化模型，并进行随机取样
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                # `state` is a tuple of tensors for `nn.LSTM` and
                # for our custom scratch implementation
                for s in state:
                    s.detach_()

        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        updater.zero_grad()
        l.backward()
        grad_clipping(net, 1)
        updater.step()
        if index % 100 == 0:
            print(f"loss:{l.item()}")
            SummaryWriter.add_scalar(writer, "loss", l.item(), index)

        index += 1
        metric.add(l.item() * utils.size(y), utils.size(y))
    return np.exp(metric[0] / (metric[1]+1e-8)), metric[1] /((time.time()-start_times)+1e-8)

def train_net(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False, save_path=None):
    """训练模型"""
    loss = nn.CrossEntropyLoss()
    #animator = utils.Animator(xlabel='epoch', ylabel='perplexity',
    #                        legend=['train'], xlim=[10, num_epochs])
    # Initialize
    updater = torch.optim.Adam(net.parameters(), lr)
    predict = lambda prefix: predict_next_text(prefix, 50, net, vocab, device)
    # 训练与预测
    for epoch in range(num_epochs):
        ppl, speed = train_net_epoch(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 1 == 0:
            print(predict('地球'))
            if save_path:
                torch.save(net.state_dict(), save_path)
            # animator.add(epoch + 1, [ppl])
        SummaryWriter.add_scalar(writer, "perplexity", ppl, epoch)
        # # 使用tensorboard 记录训练过程的loss 和 perplexity

        print(f'epoch:{epoch},perplexity {ppl:.1f}.')

    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')



def main():
    batch_size, num_steps = 32, 28
    train_iter, vocab = DataCollect.load_data_fiction(batch_size, num_steps, use_random_iter=False)
    num_hiddens = 1300 # 吞噬星空的隐藏层是1300
    rnn_layer = nn.RNN(len(vocab), num_hiddens)
    device = torch.device("cuda")
    net = RNNModel(rnn_layer, vocab_size=len(vocab))
    net = net.to(device)
    print(predict_next_text('地球', 100, net, vocab, device))
    num_epochs, lr = 500, 1e-3
    load_weight = False
    weigh_path = "RNN weight.pth"
    if load_weight:
        net.load_state_dict(torch.load(weigh_path))
    train_net(net, train_iter, vocab, lr, num_epochs, device, save_path=weigh_path)
    writer.close()


if __name__=="__main__":
    main()