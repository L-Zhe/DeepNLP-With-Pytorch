from    dataset import prepare_sequence
import  random
import  torch
from    collections import Counter

flatten = lambda l: [item for sublist in l for item in sublist]


def creatUnigramable(corpus, vocab, power=0.75, z=0.001):
    '''Creat an unigram set to negative sample.
    :param
            corpus: corpus set.
            vocab: vocab set.
            power: an power to product an unigram distribution of corpus set.
            z:
    :return:
            unigran_table: a distribution word set to be sample.
    '''
    word_count = Counter(flatten(corpus))
    num_total_words = len(vocab)
    unigram_table = []
    for vo in vocab:
        unigram_table.extend([vo] * int((word_count[vo]
                                         / num_total_words) ** power / z))
    random.shuffle(unigram_table)
    return unigram_table

def negative_sample(targets, unigram_table, word2idx, k):
    '''Sample some negative sample from unirgam set
    :param
            targets: positive words.
            unigram_table: a distribution word set to be sample.
            word2idx: a mapping from word to index.
            k: the num of negative word to be sample.
    :return:
            Tensor: a vector matrix of negative sample words.
    '''

    batch_size = targets.size(0)
    neg_samples = []
    for i in range(batch_size):
        nsample = []

        target_index = targets[i].data.tolist()

        while len(nsample) < k:  # num of sampling
            neg = random.choice(unigram_table)

            if word2idx[neg] == target_index:
                continue
            nsample.append(neg)
        neg_samples.append(prepare_sequence(nsample, word2idx).view(1, -1))

    return torch.cat(neg_samples)