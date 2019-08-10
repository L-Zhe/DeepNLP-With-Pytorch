import  torch
import  random
import  re
import  numpy as np
import  os

random.seed(1024)
flatten2 = lambda l: [word.lower() for sublist in l for word in sublist]
flatten1 = lambda l: [word for word in l]

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor


class lang():
    def __init__(self, path):
        self.train, self.target = zip(*self.creatDataSet(path))

        self.data2idx, self.idx2data = self.word2index(self.train, False)
        self.tar2idx, self.idx2tar = self.word2index(self.target, True)

        self.train_data = [self.prepare_sequence(seq, self.data2idx) for seq in self.train]
        self.train_target = [self.prepare_target(tar, self.tar2idx) for tar in self.target]

        self.dataset = list(zip(self.train_data, self.train_target))
        self.valid_data, self.test_data, self.train_data = self.splitDataSet(self.dataset)
        # print(self.data[:1])

    def creatDataSet(self, path):
        f = open(path, 'r', encoding='UTF-8')
        corpus = [re.split('[:, ]', co.strip('/n')) for co in f.readlines()]
        f.close()
        maxN = max([len(co) - 1 for co in corpus])

        corpus = [(self.padding(co[1:], maxN), co[0]) for co in corpus]

        return corpus
    def splitDataSet(self, dataset):
        random.shuffle(dataset)
        length = len(dataset)
        valid_len = int(length * 0.2)
        test_len = int(length * 0.3)
        return dataset[:valid_len], dataset[valid_len:test_len], dataset[test_len:]

    def padding(self, data, maxN):
        return data + ['<PAD>'] * (maxN - len(data))

    def prepare_sequence(self, seq, word2idx):
        idxs = list(map(lambda w: word2idx[w] if word2idx.get(w) is not None else word2idx['<UNK>'], seq))
        return idxs

    def prepare_target(self, target, tar2idx):
        return tar2idx[target]

    def word2index(self, data, is_tar):

        word2idx = {}
        idx2word = {}
        if is_tar:
            vocab = list(set(flatten1(data)))
        else:
            vocab = list(set(flatten2(data)))
            word2idx['<PAD>'] = 0; idx2word[0] = '<PAD>'
            word2idx['<UNK>'] = 1; idx2word[1] = '<UNK>'

        for vo in vocab:
            if word2idx.get(vo) is None:
                word2idx[vo] = len(word2idx)
                idx2word[len(idx2word)] = vo

        return word2idx, idx2word

    def getBatch(self, data, BATCH_SIZE):

        random.shuffle(data)
        sindex = 0
        eindex = BATCH_SIZE
        while(eindex < len(data)):
            yield data[sindex:eindex]
            sindex = eindex
            eindex += BATCH_SIZE

        if eindex > len(data):
            return data[sindex:]

if __name__ == '__main__':
    a = lang('./data/train_5500.label.txt')

