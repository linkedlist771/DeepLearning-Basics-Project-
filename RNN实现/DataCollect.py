# 导入所需要的库
import os
import sys
import matplotlib.pyplot as plt
import random
import torch
import collections
import re



def remove_non_chinese_english_punctuation_numbers(string):
  # 匹配所有中文、英文、标点符号和数字, 我认为在中文小说中，其他内容含义不大
  pattern = r'[\u4e00-\u9fa5a-zA-Z0-9“”，。\n（）-]+'
  
  # 使用正则表达式匹配出所有符合条件的字符
  matches = re.findall(pattern, string)

  # 将所有匹配到的字符连接起来
  result = ''.join(matches)
  # 将回车改为空格
  # result = result.replace("\n", " ")


  return result

class FictionReader :
    """ 读取小说的文本内容，可以读取全部的小说，或者读取某一个小说, 某前些部分"""
    def __init__(self, fiction_name, percentage=1.0) -> None:
        self.fiction_name = fiction_name
        self.percentage = percentage
    
    def read_data(self):
        root_path = "C:/Users/23174/Desktop/GitHub项目/DeepLearningBasics/DeepLearning-Basics/大作业/DataText/"
        lines = []
        if self.fiction_name == "all":
            # 如果全部的文本都要读入的话
            for file_name in os.listdir(root_path):
                file_name = os.path.join(root_path, file_name)
                with open(file_name, 'r', encoding="UTF-8") as f:
                    lines = [*lines, *f.readlines()]
        elif self.fiction_name != "":
            # 如果文本还存在的话
            file_name = os.path.join(root_path, f"{self.fiction_name}.txt")
            with open(file_name, 'r', encoding="UTF-8") as f:
                lines = [*lines, *f.readlines()]
       
        assert self.fiction_name, f"{self.fiction_name} is not a valid fiction name !"

        lines = [remove_non_chinese_english_punctuation_numbers(line) for line in lines]
        # lines = list(filter(lambda x: x != '', lines))
        # 未做处理
        return lines 




def tokenize(lines, token='word'):
    """将句子依据单个汉字字符或者通过命名实体进行拆分"""
    # 命名实体
    if token == 'word':
        raise NotImplementedError()
        pass #return [line.split() for line in lines]
    # 根据单个汉字字符进行拆分
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type: ' + token)

def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用随机采样生成小批量的迭代器."""
    # 从 corpus 中随机选择一个起始位置
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减 1 是因为需要给标签留出空间
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为 num_steps 的子序列的起始位置的列表
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 随机采样中，迭代过程中相邻两个随机小批量中的子序列在原始序列中不一定是相邻的
    random.shuffle(initial_indices)
    def data(pos):
        # 返回从 `pos` 开始长度为 `num_steps` 的序列
        return corpus[pos: pos + num_steps]
    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 这里，`initial_indices` 包含了子序列的随机起始位置
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.Tensor(X), torch.Tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使用连续采样生成小批量的迭代器."""
    # Start with a random offset to partition a sequence
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.Tensor(corpus[offset: offset + num_tokens])
    Ys = torch.Tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y
   
def load_corpus_fiction(max_tokens=-1):
    """返回该文本数据集的token 和 Voc  `"""
    # 读取小说的文本行
    fictionReader = FictionReader(fiction_name="吞噬星空")
    lines = fictionReader.read_data()
    # 对文本行进行字符型分词
    tokens = tokenize(lines, 'char')
    # 创建词汇表
    vocab = Vocab(tokens)
    # 由于数据集中的每一行文本不一定是一个句子或者一段段落，因此需要将所有文本行扁平化为单个列表
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        # 如果指定了最大标记数，则只保留前max_tokens个标记
        corpus = corpus[:max_tokens]
    return corpus, vocab


def count_corpus(tokens):
    """统计tokens频率"""
    # 这里，`tokens` 是一个一维列表或二维列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将令牌列表的列表展平成令牌列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

class Vocab:
    """文本的词元类"""
    def __init__(self, tokens=None, min_freq=10, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # print(self._token_freqs)
        # 未知令牌的索引为 0
        self.idx_to_token = reserved_tokens#['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    # 返回词典中的数量
    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        # 如果 tokens 是一个标量，则返回其索引
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        # 如果 tokens 是一个列表或元组，则递归调用 __getitem__ 将其转化为索引列表
        return [self.__getitem__(token) for token in tokens]

    # 将索引列表转化为列表
    def to_tokens(self, indices):
        # 如果 indices 是一个标量，则返回其对应的令牌
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        # 如果 indices 是一个列表或元组，则递归调用 to_tokens 将其转化为令牌列表
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知令牌的索引
        return 0

    @property
    def token_freqs(self):  # 频率的索引
        return self._token_freqs

class SeqDataLoader:
    """序列数据迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        # 根据 use_random_iter 的值选择使用随机采样或顺序采样的迭代器
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        # 加载时光机数据集并创建词典
        self.corpus, self.vocab = load_corpus_fiction(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps


    def __iter__(self):
        # 使用 data_iter_fn 函数返回序列数据迭代器
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_fiction(batch_size, num_steps,
                           use_random_iter=False, max_tokens=50000):#500000):
    """返回小说的迭代器和词元。"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

# batch_size, num_steps = 32, 35
# daiter, voca = load_data_fiction(batch_size, num_steps,)