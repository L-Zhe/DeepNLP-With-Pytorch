import  torch
import  os
import  random

flatten = lambda l: [item for sublist in l for item in sublist]
random.seed(1024)

device = torch.device("cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LongTensor = torch.LongTensor
# LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

def creatDataSet(path, filename):
    f = open(os.path.join(path, filename), 'r', encoding='utf-8')
    corpus = f.readlines()
    f.close()
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

def prepare_sequence(seq, word2idx):
    idxs = list(map(lambda w: word2idx[w] if word2idx.get(w) is not None else word2idx['<UNK>'], seq))

    return LongTensor(idxs)

def getBatch(train_data, seq_length):

    for i in range(0, train_data.size(1) - seq_length, seq_length):
        inputs = LongTensor(train_data[:, i:i+seq_length])
        targets = LongTensor(train_data[:, (i+1):(i+1)+seq_length])
        yield (inputs, targets)


def batchify(train_data, batch_size):
    nbatch = len(train_data) // batch_size
    train_data = train_data.narrow(0, 0, nbatch*batch_size)
    train_data = train_data.reshape(batch_size, -1)

    return train_data.to(device)

if __name__ == '__main__':
    copus, vocab = creatDataSet('./data', 'ptb.train.txt')
