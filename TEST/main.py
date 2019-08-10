import  torch
from    torch import nn, optim
import  random
import  numpy as np
from    model import LanguageModel
from    dataset import creatDataSet, getBatch, batchify, word2index, prepare_sequence
from    math import ceil
from    utils import view_bar

random.seed(1024)
device = torch.device("cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def trainModel(model, train_data, valid_data, BATCH_SIZE, SEQ_LENGTH, EPOCH):

    model.train()
    for epoch in range(EPOCH):
        losses = []
        total = ceil((train_data.size(1) - SEQ_LENGTH) / SEQ_LENGTH)
        hidden = model.hidden_init(BATCH_SIZE)

        for i, batch in enumerate(getBatch(train_data, SEQ_LENGTH)):
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

        print("\tmean loss : %0.2f" % np.mean(losses), end='\t')
        losses = []

        testModel(model, valid_data, BATCH_SIZE, SEQ_LENGTH)


def testModel(model, test_data, BATCH_SIZE, SEQ_LENGTH):

    model.eval()
    with torch.no_grad():
        hidden = model.hidden_init(BATCH_SIZE)
        criterion = nn.CrossEntropyLoss()
        losses = []
        for i, batch in getBatch(test_data, SEQ_LENGTH):

            inputs, targets = batch
            hidden = model.detach_hidden(hidden)

            model.zero_grad()
            preds, hidden = model(test_data, hidden)

            loss = criterion(preds, targets.reshape(-1))
            losses.append(loss.data)

        print("test loss : %0.2f" % np.mean(losses))


if __name__ == '__main__':

    trainSet, vocab = creatDataSet('./data', 'ptb.train.txt')
    testSet, _ = creatDataSet('./data', 'ptb.test.txt')
    validSet, _ = creatDataSet('./data', 'ptb.valid.txt')

    word2idx, idx2word = word2index(vocab)

    ### Parameters Set ##########
    VOCAB_SIZE = len (word2idx)
    EMBEDDING_SIZE = 128
    HIDDEN_SIZE = 1024
    N_LAYERS = 1
    DOPROUT_P = 0.5
    BATCH_SIZE = 20
    SEQ_LENGTH = 30
    EPOCH = 40
    LEARNING_RATE = 0.01
    #############################

    train_data = batchify(prepare_sequence(trainSet, word2idx), BATCH_SIZE)
    test_data = batchify(prepare_sequence(testSet, word2idx), BATCH_SIZE)
    valid_data = batchify(prepare_sequence(validSet, word2idx), BATCH_SIZE)

    model = LanguageModel(VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE, N_LAYERS, DOPROUT_P).to(device)
    model.weight_init()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    trainModel(model, train_data, valid_data, BATCH_SIZE, SEQ_LENGTH, EPOCH)
    testModel(model, test_data, BATCH_SIZE, SEQ_LENGTH)