import  torch
from    torch import optim
from    torch.nn import functional as F
from    skip_gram import Skipgram
import  numpy as np
import  random
from    dataset import creatCorpus, creatVocab, creatTrainData, word2index, \
    prepare_sequence, prepare_word, getBatch
from    negativeSample import negative_sample, creatUnigramable
import  time

#Parameters:

EMBEDDING_SIZE = 300
BATCH_SIZE = 256
EPOCH = 100
learning_rate = 0.01
negativeSample = True
negSampleNum = 10

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
corpus = creatCorpus()
vocab = creatVocab(corpus)
word2idx, idx2word = word2index(vocab)
train_data = creatTrainData(corpus)

unigram_table = creatUnigramable(corpus, vocab)

losses = []
model = Skipgram(len(word2idx), EMBEDDING_SIZE).to(device)

optimizer = optim.Adam(model.parameters(), lr = learning_rate)
start_time = time.clock()



for epoch in range(EPOCH):
    for i, batch in enumerate(getBatch(train_data, BATCH_SIZE)):

        inputs, targets = zip(*batch)
        inputs = torch.cat(inputs) # B X 1
        targets = torch.cat(targets) # B X 1

        if  negativeSample:
            vocabs = negative_sample(targets, unigram_table, word2idx, negSampleNum)

        else:
            vocabs = prepare_sequence(list(vocab), word2idx).expand(len(inputs), len(vocab))

        model.zero_grad()

        loss = model(inputs, targets, vocabs)

        loss.backward()
        optimizer.step()

        losses.append(loss.data.tolist())
        # print(loss)

    if  epoch % 10 == 0:
        print(time.clock() - start_time)
        print("Epoch : %d, mean_loss : %.02f" % (epoch, np.mean(losses)))
        losses = []

def word_similarity(target, vocab):
    target_V = model.prediction(prepare_word(target, word2idx))

    similarities = []
    for i in range(len(vocab)):

        if vocab[i] == target:  continue
        vector = model.prediction(prepare_word(list(vocab)[i], word2idx))
        cosine_sim = F.cosine_similarity(target_V, vector).data.tolist()[0]
        similarities.append([vocab[i], cosine_sim])

    return sorted(similarities, key=lambda x: x[1], reverse=True)[:10] # sort by similarity

test = random.choice(list(vocab))
print(test)
print(word_similarity(test, vocab))

