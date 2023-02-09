
"""

Section 1 !
"""
# import torch
# import tensorwatch as tw
# from model import *
# import DataCollect
#
#
# import torch
# from torchvision.models import AlexNet
# from torchviz import make_dot
# import math
# import time
# import numpy as np
# import utils
# from model import *
# from createFile import *
#
# def predict_next_text(prefix, num_preds, net, vocab, device):
#     """根据前缀来生成后面内容"""
#     state = net.begin_state(batch_size=1, device=device)
#     outputs = [vocab[prefix[0]]]
#     for y in prefix[1:]:  # Warm-up period\
#         input = torch.reshape(torch.tensor([outputs[-1]], device=device), (1, 1))
#         print(input)
#         _, state = net(input, state)
#         return input, state
#
# def main():
#     data_iter, vocab = DataCollect.load_data_fiction(1, 50)
#     device = torch.device("cpu")
#     num_hiddens = 1300
#     rnn_layer = nn.RNN(len(vocab), num_hiddens)
#     net = RNNModel(rnn_layer, vocab_size=len(vocab))
#     net = net.to(device)
#     predict_string = predict_next_text('地球的华夏国，由于具有两大人类强者', 2000, net, vocab, device)
#     print(predict_string)
#     pdf = TextFileWriter(predict_string)
#
# # 其实就两句话
# batch_size, num_steps = 32, 28
# train_iter, vocab = DataCollect.load_data_fiction(batch_size, num_steps, use_random_iter=False)
# num_hiddens = 1300 # 吞噬星空的隐藏层是1300
# rnn_layer = nn.RNN(len(vocab), num_hiddens)
# device = torch.device("cpu")
# model=RNNModel(rnn_layer, vocab_size=len(vocab))
# model.load_state_dict(torch.load("RNN weight.pth"))
#
# # 以AlexNet为例，前向传播
# input, state = predict_next_text('地球的华夏国，由于具有两大人类强者', 2000, model, vocab, device)
# s, state = model(input, state)
#
# # 构造图对象，3种方式
# g = make_dot(state)
# # g = make_dot(y, params=dict(model.named_parameters()))
# # g = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
#
# # 保存图像
# # g.view()  # 生成 Digraph.gv.pdf，并自动打开
# g.render(filename='graph1', view=False)  # 保存为 graph.pdf，参数view表示是否打开pdf
# g2 = make_dot(s)
#
# g2.render(filename='graph2', view=False)  # 保存为 graph.pdf，参数view表示是否打开pdf
#
#

"""

Section 2 !
"""


import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from d2l import torch as d2l
from sklearn.metrics.cluster import  mutual_info_score
import numpy as np
df = pd.read_csv("GPT2.csv")
df = df.iloc[:, 2:]
df.plot.line()
# use the two y axis to plot the two cloumns separately
df1 = df.iloc[:, 0]
df2 = df.iloc[:, 1]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
p2 = ax2.plot(df2, 'b-')
p1 = ax1.plot(df1, 'g-')
ax1.set_xlabel('epoch', fontsize=16)
ax1.set_ylabel('accuracy', color='g', fontsize=16)
ax2.set_ylabel('loss', color='b', fontsize=16)
plt.legend(p1+p2, ['accuracy', 'loss'], fontsize=16)
plt.grid()
plt.title('GPT2 loss and accuracy', fontsize=16)
plt.savefig('GPT2 loss.png', dpi=400)
plt.show()