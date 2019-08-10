import  torch
import  random
from    collections import Counter
import  nltk

random.seed(1024)
flatten = lambda list: [word for sublist in list for word in sublist]
device = torch.device("cpu")
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        test_data = windows[sindex:(sindex+batch_size)]
        train_data = windows[(sindex+batch_size), eindex]
        yield train_data, test_data

        sindex = eindex
        eindex = eindex + batch_size

    if eindex < len(windows):
        test_data = windows[sindex:(sindex+batch_size)]
        train_data = windows[(sindex+batch_size):]
        yield train_data, test_data


