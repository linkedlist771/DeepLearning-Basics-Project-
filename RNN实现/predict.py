import torch
from torch import nn
import DataCollect
import math
import time
import numpy as np
import utils
from model import *
from createFile import *

def predict_next_text(prefix, num_preds, net, vocab, device):
    """根据前缀来生成后面内容"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    for y in prefix[1:]:  # Warm-up period\
        input = torch.reshape(torch.tensor([outputs[-1]], device=device), (1, 1))
        _, state = net(input, state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Predict `num_preds` steps
        input = torch.reshape(torch.tensor([outputs[-1]], device=device), (1, 1))
        y, state = net(input, state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])



def main():
    data_iter, vocab = DataCollect.load_data_fiction(1, 50)
    device = torch.device("cuda")
    num_hiddens = 1300
    rnn_layer = nn.RNN(len(vocab), num_hiddens)
    net = RNNModel(rnn_layer, vocab_size=len(vocab))
    net = net.to(device)
    net.load_state_dict(torch.load("RNN weight.pth"))
    predict_string = predict_next_text('“罗峰师兄，我，我有事想请罗峰师兄帮忙。”壮硕男忐忑道。', 200, net, vocab, device)
    print(predict_string)
    pdf = TextFileWriter(predict_string)

if __name__=="__main__":
    main()