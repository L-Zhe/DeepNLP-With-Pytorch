import  torch
import  random
import  nltk
from    collections import Counter
from    copy import deepcopy
import os

flatten = lambda l: [item for sublist in l for item in sublist]

random.seed(1024)

device = torch.device("cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def prepare_seq(seq, word2idx):
    idxs = list(map(lambda w: word2idx[w] if word2idx.get(w) is not None else word2idx['<UNK>'], seq))
    return torch.LongTensor(idxs)

def creatDataSet(path, filename):

    corpus = open(os.path.join(path, filename), 'r', encoding='utf-8').readlines()
    corpus = flatten([co.strip().split() + ['<\s>'] for co in corpus])
    random.shuffle(corpus)
    vocab = list(set(corpus))
    return corpus, vocab

def word2index(vocab):
    word2idx = {}
    idx2word = {}
    word2idx['<UNK>'] = 0
    idx2word[0] = '<UNK>'

    for vo in vocab:
        if word2idx.get(vo) is None:
            idx2word[len(word2idx)] = vo
            word2idx[vo] = len(word2idx)

    return word2idx, idx2word

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)

    data = data.view(bsz, -1).contiguous()
    data = data.to(device)
    return data

def getBatch(trainSet, seq_len):
    for i in range(0, trainSet.size(1) - seq_len, seq_len):
        inputs = torch.LongTensor(trainSet[:, i:i+seq_len])
        targets = torch.LongTensor(trainSet[:, (i + 1):(i + 1) + seq_len].contiguous())
        yield (inputs, targets)




