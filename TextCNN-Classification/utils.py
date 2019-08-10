import  torch
from    torch import nn, optim
import  numpy as np
from model import *


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor


def trainModel(model, LANG, EPOCH, BATCH_SIZE, EMBEDDING_SIZE, KERNEL_DIM,
               KERNEL_SIZE, LR, DROPOUT_P=0.5, STEP_SIZE = 15, GAMMA=0.5,
               momentum=0.9, weight_decay=1e-4):

    # optimizer = optim.SGD(model.parameters(), lr=LR, momentum=momentum, weight_decay=weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    for epoch in range(EPOCH):
        losses = []
        bestModel = None
        maxAcc = 0
        model.train()
        for i, batch in enumerate(LANG.getBatch(LANG.train_data, BATCH_SIZE)):

            inputs, targets = zip(*batch)
            inputs = LongTensor(inputs).to(device)
            targets = LongTensor(targets).to(device)

            model.zero_grad()
            preds = model(inputs, True)

            loss = criterion(preds, targets)
            losses.append(loss.tolist())
            loss.backward()
            optimizer.step()

        scheduler.step()

        valid_acc = testModel(model, LANG, state='valid')
        if valid_acc >= maxAcc:
            maxAcc = valid_acc
            bestModel = model
        print("[%d/%d]\ttrain_loss : %0.2f\tvalid_acc : %0.2f%%" % (epoch+1, EPOCH, np.mean(losses), valid_acc))
        losses = []

    return bestModel


def testModel(model, LANG, state):
    model.eval()
    dataset = LANG.test_data if state == 'test' else LANG.valid_data
    with torch.no_grad():
        acc = 0
        for data in dataset:
            inputs, targets = data
            inputs = LongTensor([inputs]).to(device)

            preds = int(model(inputs, False).max(1)[1])

            if preds == targets:
                acc += 1
    return acc / len(dataset) * 100