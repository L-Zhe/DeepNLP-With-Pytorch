import  torch
import  random
from    torch import nn, optim
from    torch.nn import functional as F
from    GloVe import GloVe
import  dataset
from    dataset import creatTrainData, getBatch, prepare_word
import numpy as np

EMBEDDING_SIZE = 30
BATCH_SIZE = 256
EPOCH = 50
WINDOW_SIZE = 5

device = torch.device("cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

corpus = dataset.creatCorpus()
vocab = dataset.creatVocab(corpus)
word2idx, idx2word = dataset.word2index(vocab)
train_data = creatTrainData(corpus, WINDOW_SIZE)

model = GloVe(len(vocab), EMBEDDING_SIZE).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

def trainModel(train_data):
    losses = []

    for epoch in range(EPOCH):
        for i, batch in enumerate(getBatch(train_data, BATCH_SIZE)):

            inputs, targets, coocs, weights  = zip(*batch)

            inputs = torch.cat(inputs)
            targets = torch.cat(targets)
            coocs = torch.cat(coocs)
            weights = torch.cat(weights)

            model.zero_grad()
            loss = model(inputs, targets, coocs, weights)

            loss.backward()
            optimizer.step()

            losses.append(loss.data.tolist())
        if epoch % 10 == 0:
            print("Epoch : %d, mean_loss : %.02f" % (epoch, np.mean(losses)))
            losses = []


def word_similarity(target, vocab):
    target_V = model.prediction(prepare_word(target, word2idx)).to(device)
    similarities = []
    for i in range(len(vocab)):
        if vocab[i] == target:
            continue

        vector = model.prediction(prepare_word(list(vocab)[i], word2idx)).to(device)

        cosine_sim = F.cosine_similarity(target_V, vector).data.tolist()[0]
        similarities.append([vocab[i], cosine_sim])
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:10]

if __name__ == '__main__':
    trainModel(train_data)
    test = random.choice(list(vocab))
    print(test)
    print(word_similarity(test, vocab))