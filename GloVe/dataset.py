import  torch
from collections import Counter
import  nltk
import random
from itertools import combinations_with_replacement

flatten = lambda l: [item for sublist in l for item in sublist]
random.seed(1024)

device = torch.device("cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

nltk.download("gutenberg")

def getBatch(window_data, batch_size):
    random.shuffle(window_data)
    sindex = 0
    eindex = batch_size
    while(eindex < len(window_data)):
        yield window_data[sindex:eindex]
        sindex = eindex
        eindex = eindex + batch_size

    if eindex >= len(window_data):
        yield window_data[sindex:]


def prepare_sequence(seq, word2idx):

    idx = list(map(lambda w: word2idx[w] if word2idx.get(w) is not None
                                        else word2idx['<UNK>'], seq))
    return torch.LongTensor(idx).to(device)

def prepare_word(word, word2idx):
    return torch.LongTensor([word2idx[word] if word2idx.get(word) is not None
                             else word2idx['<UNK>']]).to(device)


def creatCorpus():
    corpus = list(nltk.corpus.gutenberg.sents('melville-moby_dick.txt'))[:500]
    corpus = [[word.lower() for word in sent] for sent in corpus]

    return corpus

def creatVocab(corpus):
    vocab = list(set(flatten(corpus)))
    vocab .append('<UNK>')

    return vocab

def word2index(vocab):
    word2idx = {'<UNK>': 0}
    idx2word = {0: '<UNK>'}
    index = 0
    for vo in vocab:
        if word2idx.get(vo) is None:
            word2idx[vo] = index
            idx2word[index] = vo
            index += 1

    return word2idx, idx2word

def creatWindowData(corpus, WINDOW_SIZE = 3):
    vocab = creatVocab(corpus)
    windows = flatten([list(nltk.ngrams(['<DUMMY>'] * WINDOW_SIZE
                                        + c + ['<DUMMY>'] * WINDOW_SIZE, WINDOW_SIZE * 2 + 1)) for c in corpus])

    word2idx, _ = word2index(vocab)
    window_data = []

    for window in windows:

        for i in range(WINDOW_SIZE * 2 + 1):
            if i == WINDOW_SIZE or window[i] == '<DUMMY>':
                continue
            window_data.append((window[WINDOW_SIZE], window[i]))

    return window_data


#Calculate f(x_ij)
def weighting(w_i, w_j, X_ik):
    try:
        x_ij = X_ik[(w_i, w_j)]
    except:
        x_ij = 1

    x_max = 100
    alpha = 0.75

    if x_ij < x_max:
        result = (x_ij/x_max) ** alpha
    else:
        result = 1

    return result

def creatWeight(corpus, window_data):
    vocab = creatVocab(corpus)
    X_i = dict(Counter(flatten(corpus)))
    X_ik_windows = dict(Counter(window_data))
    X_ik = {}
    weighting_dic = {}

    for bigram in combinations_with_replacement(vocab, 2):
        if X_ik_windows.get(bigram) is not None:
            co_occur = X_ik_windows[bigram]
            X_ik[bigram] = co_occur + 1
            X_ik[(bigram[1], bigram[0])] = co_occur + 1

        weighting_dic[bigram] = weighting(bigram[0], bigram[1], X_ik)
        weighting_dic[(bigram[1], bigram[0])] = weighting(bigram[1], bigram[0], X_ik)
    return X_ik, weighting_dic


def creatTrainData(corpus, WINDOW_SIZE = 3):
    vocab = creatVocab(corpus)
    word2idx, _ = word2index(vocab)
    window_data = creatWindowData(corpus, WINDOW_SIZE)

    X_ik, weighting_dic = creatWeight(corpus, window_data)
    X_p = []; y_p = []
    co_p = []; weight_p = []
    word2idx, _ = word2index(vocab)
    for pair in window_data:
        X_p.append(prepare_word(pair[0], word2idx).view(1, -1))
        y_p.append(prepare_word(pair[1], word2idx).view(1, -1))
        try:
            cooc = X_ik[pair]
        except:
            cooc = 1
        co_p.append(torch.log(torch.Tensor([cooc])).to(device).view(1, -1))
        weight_p.append(torch.FloatTensor([weighting_dic[pair]]).to(device).view(1, -1))

    return list(zip(X_p, y_p, co_p, weight_p))


if __name__ == '__main__':
    corpus = creatCorpus()
    window_data = creatWindowData(corpus)
    x = (0, 1, 2)
    # print(window_data[:5])
    # a, b = list(zip(*window_data))
    # print(a[:5])
    # print(b[:5])
    P = creatWeight(corpus, window_data)
    print(creatTrainData(corpus, 5)[:5])




