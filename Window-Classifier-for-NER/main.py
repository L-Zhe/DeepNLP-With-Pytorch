import  torch
from    torch import nn, optim
from    model import WindowClassifier
from    dataset import getBatch, creatData, prepare_word, prepare_seq, \
        prepare_tag, word2index, tag2index, creatTrainData, splitDataSet, \
        crossValidation
from    math import ceil
import  numpy as np
import  time
import sys

device = torch.device("cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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