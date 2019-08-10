from    dataset import *
from    utils import *

PATH = './data/train_5500.label.txt'

if __name__ == '__main__':
    corpus = lang(PATH)
    ############################################################################################
    EPOCH = 50
    BATCH_SIZE = 50
    EMBEDDING_SIZE = 300
    KERNEL_DIM = 100
    KERNEL_SIZE = range(1, 5)
    LR = 0.001
    DROPOUT_P = 0.5
    ############################################################################################

    model = TestCNN(len(corpus.data2idx), EMBEDDING_SIZE, len(corpus.tar2idx),
                    KERNEL_DIM, KERNEL_SIZE, DROPOUT_P).to(device)
    ############################################################################################

    bestModel = trainModel(model, corpus, EPOCH, BATCH_SIZE, EMBEDDING_SIZE,
           KERNEL_DIM, KERNEL_SIZE, LR, DROPOUT_P)
    test_acc = testModel(bestModel, corpus, state='test')

    print("test_acc : %0.2f%%" % (test_acc))
