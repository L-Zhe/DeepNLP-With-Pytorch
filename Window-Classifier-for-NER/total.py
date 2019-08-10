import torch
from torch import nn, optim
from torch.nn import functional as F
import random
from collections import Counter
import nltk
import numpy as np
from tqdm import tqdm
from math import ceil
import time
import sys

random.seed(1024)
flatten = lambda list: [word for sublist in list for word in sublist]
device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def creatCorpus():
    nltk.download("conll2002")
    corpus = nltk.corpus.conll2002.iob_sents()
    return corpus


def creatData(corpus):
    sents = []
    target = []
    for cor in corpus:
        sent, _, tag = list(zip(*cor))
        sents.append(sent)
        target.append(tag)

    data = list(zip(sents, target))

    return data


def getBatch(train_data, BATCH_SIZE):
    random.shuffle(train_data)
    sindex = 0
    eindex = BATCH_SIZE
    while eindex < len(train_data):
        yield train_data[sindex:eindex]
        sindex = eindex
        eindex = eindex + BATCH_SIZE

    if eindex >= len(train_data):
        yield train_data[sindex:]


def prepare_word(word, word2idx):
    return torch.LongTensor([word2idx[word] if word2idx.get(word) is not None
                             else word2idx['<UNK>']]).to(device)


def prepare_seq(seq, word2idx):
    idx = list(map(lambda w: word2idx[w] if word2idx.get(w) is not None
    else word2idx['<UNK>'], seq))
    return torch.LongTensor([idx]).to(device)


def prepare_tag(tag, tag2idx):
    return torch.LongTensor([tag2idx[tag]]).to(device)


def word2index(vocab):
    word2idx = {'<UNK>': 0, '<DUMMY>': 1}
    idx2word = {0: '<UNK>', 1: '<DUMMY>'}

    for vo in vocab:
        if word2idx.get(vo) is None:
            idx2word[len(word2idx)] = vo
            word2idx[vo] = len(word2idx)

    return word2idx, idx2word


def tag2index(target):
    tag2idx = {}
    idx2tag = {}
    for tag in target:

        if tag2idx.get(tag) is None:
            idx2tag[len(tag2idx)] = tag
            tag2idx[tag] = len(tag2idx)

    return tag2idx, idx2tag


def creatTrainData(WINDOW_SIZE):
    corpus = creatCorpus()
    data = creatData(corpus)
    sents, targets = list(zip(*data))
    vocab = list(set(flatten(sents)))
    target = list(set(flatten(targets)))

    word2idx, _ = word2index(vocab)
    tag2idx = tag2index(target)

    windows = []
    for sample in data:
        dummy = ['<DUMMY>'] * WINDOW_SIZE
        window = list(nltk.ngrams(dummy + list(sample[0]) + dummy, WINDOW_SIZE * 2 + 1))
        windows.extend([[list(window[i]), sample[1][i]] for i in range(len(sample[0]))])

    return vocab, target, windows


def splitDataSet(windows, alpha):
    random.shuffle(windows)
    index = int(len(windows) * alpha)
    test_data = windows[:index]
    train_data = windows[index:]

    return train_data, test_data


def crossValidation(windows, k):
    batch_size = len(windows) / k
    alpha = k / len(windows)
    sindex = 0
    eindex = batch_size

    while eindex < len(windows):
        test_data = windows[sindex:(sindex + batch_size)]
        train_data = windows[(sindex + batch_size), eindex]
        yield train_data, test_data

        sindex = eindex
        eindex = eindex + batch_size

    if eindex < len(windows):
        test_data = windows[sindex:(sindex + batch_size)]
        train_data = windows[(sindex + batch_size):]
        yield train_data, test_data


class WindowClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_size, window_size, hidden_size, output_size):

        super(WindowClassifier, self).__init__()

        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.hidden_layer1 = nn.Linear(embedding_size * (window_size * 2 + 1), hidden_size)
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, inputs, is_training=False):
        embeds = self.embed(inputs)
        concated = embeds.view(-1, embeds.size(1) * embeds.size(2))
        h0 = self.relu(self.hidden_layer1(concated))
        if is_training:
            h0 = self.dropout(h0)
        h1 = self.relu(self.hidden_layer2(h0))
        if is_training:
            h1 = self.dropout(h1)

        out = self.softmax(self.output_layer(h1))

        return out


BATCH_SIZE = 128
EMBEDDING_SIZE = 300  # x (WINDOW_SIZE*2+1) = 250
HIDDEN_SIZE = 300
EPOCH = 100
LEARNING_RATE = 0.1
WINDOW_SIZE = 2

vocab, target, windows = creatTrainData(WINDOW_SIZE)
word2idx, idx2word = word2index(vocab)
tag2idx, idx2tag = tag2index(target)
train_data, test_data = splitDataSet(windows, 0.1)


def view_bar(num, total, epoch):
    rate = num / total
    rate_num = int(ceil(rate * 100))
    r = '\r[%d/%d][%s%s]%d%% ' % (epoch, EPOCH, "=" * rate_num, " " * (100 - rate_num), rate_num,)
    sys.stdout.write(r)
    sys.stdout.flush()


def trainModel(model, EPOCH, train_data, test_data, criterion,
               optimizer, RESULTS_PATH, scheduler=None, MODEL_PATH=None):
    total = int(ceil(len(train_data) / BATCH_SIZE))
    #     print(total)
    for epoch in range(EPOCH):
        losses = []
        model.train()
        for i, batch in enumerate(getBatch(train_data, BATCH_SIZE)):
            view_bar(i, total, epoch + 1)
            sents, target = list(zip(*batch))

            inputs = torch.cat([prepare_seq(sent, word2idx) for sent in sents])
            targets = torch.cat([prepare_tag(tag, tag2idx) for tag in target])
            # print(inputs.shape, targets.shape)
            model.zero_grad()
            preds = model(inputs, is_training=True)

            loss = criterion(preds, targets)
            losses.append(loss.data.tolist())
            loss.backward()
            optimizer.step()
        if scheduler:
            scheduler.step()

        print("\t mean_loss : %0.2f" % np.mean(losses), end='')
        losses = []
        testModel(test_data)


def testModel(test_data):
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for i, test in enumerate(test_data):

            sent, target = test[0], test[1]
            input = prepare_seq(sent, word2idx)
            index = model(input).max(1)[1]
            pred = idx2tag[index.data.tolist()[0]]

            if pred == target:
                accuracy += 1

    acc = float(accuracy / len(test_data)) * 100
    print("\t test_acc : %0.2f%%" % acc)


if __name__ == '__main__':
    momentum = 0.9
    weight_decay = 1e-4
    gamma = 0.1
    milestones = [82, 123]
    model = WindowClassifier(len(word2idx),
                             EMBEDDING_SIZE,
                             WINDOW_SIZE,
                             HIDDEN_SIZE,
                             len(tag2idx)).to(device)

    criterion = nn.CrossEntropyLoss()
    #     optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay=weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=gamma, last_epoch=-1)

    trainModel(model, EPOCH, train_data, test_data, criterion, optimizer, scheduler)
#     testModel(test_data)