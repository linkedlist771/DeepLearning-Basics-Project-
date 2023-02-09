# 打个🦶先
from DataCollect import *
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch

class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        input_ids = self.data_list[index]
        return input_ids

    def __len__(self):
        return len(self.data_list)

def get_dataloader(args, shuffle, collate_fn):
    fr = FictionReader(fiction_name="all")
    data = fr.read_raw_data()
    data = data.replace("\n", "[sep]")#[-500000:]
    tokenizer = BertTokenizer(vocab_file="gpt2通用中文模型/vocab.txt")
    indexed_text = tokenizer.encode(data)
    dataset_cut = []
    for i in range(len(indexed_text)//512):
        # 将字符串分段成长度为 512
        dataset_cut.append(indexed_text[i*512:i*512+512])
    del(indexed_text)
    dataset = MyDataset(dataset_cut)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=shuffle,
                            collate_fn=collate_fn)
    return dataloader

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    arg = parser.parse_args()
    dataloader = get_dataloader(arg, shuffle=True, collate_fn=None)
    for batch in dataloader:
        print(batch)



