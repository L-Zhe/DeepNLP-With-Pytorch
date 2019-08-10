import  torch
from    torch import nn, optim
import  numpy as np

from    model import LanguageModel
from    dataset import creatDataSet, word2index, prepare_seq, getBatch, batchify
from    utils import view_bar
from    math import ceil

device = torch.device("cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EMBED_SIZE = 128
HIDDEN_SIZE = 1024
NUM_LAYER = 1
LR = 0.01
SEQ_LENGTH = 30 # for bptt
BATCH_SIZE = 20
EPOCH = 40
RESCHEDULED = False

train_data, vocab_train = creatDataSet('./data', 'ptb.train.txt')
valid_data, _ = creatDataSet('./data', 'ptb.valid.txt')
test_data, _ = creatDataSet('./data', 'ptb.test.txt')

vocab = list(set(vocab_train))
word2idx, idx2word = word2index(vocab)

trainSet = batchify(prepare_seq(train_data, word2idx), BATCH_SIZE)
testSet = batchify(prepare_seq(test_data, word2idx), BATCH_SIZE//2)
validSet = batchify(prepare_seq(valid_data, word2idx), BATCH_SIZE//2)


model = LanguageModel(len(word2idx), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYER, 0.5).to(device)
model.init_weight()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = LR)

def trainModel(trainSet, validSet):
    for epoch in range(EPOCH):
        total_loss = 0
        losses = []
        hidden = model.init_hidden(BATCH_SIZE)
        total = ceil((trainSet.size(1) - SEQ_LENGTH) / SEQ_LENGTH)
        model.train()
        for i, batch in enumerate(getBatch(trainSet, SEQ_LENGTH)):
            view_bar(i, total, epoch + 1, EPOCH)
            inputs, targets = batch
            hidden = model.detach_hidden(hidden)

            model.zero_grad()
            preds, hidden = model(inputs, hidden, True)

            loss = criterion(preds, targets.reshape(-1))
            losses.append(loss.data)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        print("\tmean_loss : %0.2f" % np.mean(losses), end='')
        losses = []
        testModel(validSet)


def testModel(testSet):
    model.eval()
    hidden = model.init_hidden(BATCH_SIZE // 2)
    with torch.no_grad():
        losses = []
        for i, batch in enumerate(getBatch(trainSet, SEQ_LENGTH)):
            inputs, targets = batch

            hidden = model.detach_hidden(hidden)
            model.zero_grad()
            preds, hidden = model(inputs, hidden)
            loss = inputs.size(1) * criterion(preds, targets.view(-1)).data
            losses.append(loss)
        print("\ttest_loss : %0.2f" % np.mean(losses))
        losses = []

if __name__ == '__main__':
    trainModel(trainSet, validSet)
    testModel(testSet)