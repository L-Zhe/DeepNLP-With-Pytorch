import  torch
from collections import Counter
import  nltk
import random

flatten = lambda l: [item for sublist in l for item in sublist]
random.seed(1024)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

nltk.download("gutenberg")

def getBatch(train_data, batch_size):
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while(eindex < len(train_data)):
        yield train_data[sindex:eindex]
        sindex = eindex
        eindex = eindex + batch_size

    if eindex >= len(train_data):
        yield train_data[sindex:]


def prepare_sequence(seq, word2idx):

    idx = list(map(lambda w: word2idx[w] if word2idx.get(w) is not None
                                        else word2idx['<UNK>'], seq))
    return torch.LongTensor(idx).to(device)

def prepare_word(word, word2idx):
    return torch.LongTensor([word2idx[word] if word2idx.get(word) is not None
                             else word2idx['<UNK>']]).to(device)


def creatCorpus():
    corpus = list(nltk.corpus.gutenberg.sents('melville-moby_dick.txt'))[:100]
    corpus = [[word.lower() for word in sent] for sent in corpus]

    return corpus

def creatVocab(corpus):
    word_count = Counter(flatten(corpus))
    border = int(len(word_count) * 0.01)
    stopWords = word_count.most_common()[:border] \
                + list(reversed(word_count.most_common()))[:border]
    stopWords = [s[0] for s in stopWords]

    vocab = list(set(flatten(corpus)) - set(stopWords))
    vocab .append('<UNK>')

    return vocab

def word2index(vocab):
    word2idx = {'<UNK>': 0}
    idx2word = {0: '<UNK>'}

    for vo in vocab:
        if word2idx.get(vo) is None:
            idx2word[len(word2idx)] = vo
            word2idx[vo] = len(word2idx)

    return word2idx, idx2word

def creatTrainData(corpus, WINDOW_SIZE = 3):
    vocab = creatVocab(corpus)
    windows = flatten([list(nltk.ngrams(
        ['<DUMMY>'] * WINDOW_SIZE + c + ['<DUMMY>']
		* WINDOW_SIZE, WINDOW_SIZE * 2 + 1)) for c in corpus])

    word2idx, _ = word2index(vocab)
    train_data = []

    for window in windows:
        center_index = prepare_word(window[WINDOW_SIZE], word2idx)

        for i in range(WINDOW_SIZE * 2 + 1):
            if i == WINDOW_SIZE or window[i] == '<DUMMY>':
                continue
            outer_index = prepare_word(window[i], word2idx)
            train_data.append((center_index.view(1, -1), outer_index.view(1, -1)))

    # for window in windows:
    #     for i in range(WINDOW_SIZE * 2 + 1):
    #         if i == WINDOW_SIZE or window[i] == '<DUMMY>':
    #             continue
    #         train_data.append((window[WINDOW_SIZE], window[i]))
    # X_p = []
    # y_p = []
    # for tr in train_data:
    #     X_p.append(prepare_word(tr[0], word2idx).view(1, -1))
    #     y_p.append(prepare_word(tr[1], word2idx).view(1, -1))
    # train_data = list(zip(X_p, y_p))
    return train_data

if __name__ == '__main__':
    corpus = creatCorpus()
    train_data = creatTrainData(corpus)
    print(train_data[:5])
    a, b = list(zip(*train_data))
    print(a[:5])
    print(b[:5])




