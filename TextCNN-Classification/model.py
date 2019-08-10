import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor

class Resnet(nn.Module):
    def __init__(self, input_size, hidden_size):

        super(Resnet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):

        f1 = self.fc1(inputs)
        f2 = self.fc2(f1)
        out = self.fc3(f2) + inputs

        return out

class TestCNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size,
                 kernel_dim=100, kernel_sizes=(3, 4, 5), dropout_p=0.5):
        super(TestCNN, self).__init__()
        self.embeds = nn.Embedding(vocab_size, embedding_size)
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_dim, (K, embedding_size)) for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(len(kernel_sizes) * kernel_dim, output_size)
        self.resnet = Resnet(output_size, 100)

    def forward(self, inputs, is_train=False):
        inputs = self.embeds(inputs).unsqueeze(1) # (B, L, D) => (B, C=1, L, D)
        inputs = [F.relu(conv(inputs)).squeeze(3) for conv in self.convs] # (B, 1, L, D) => (N_c, B, N_k, L', 1) => (N_c, B, N_k, L')
        inputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs] # (N_c, B, N_k, L') => (N_c, B, N_k, 1) => (N_c, B, N_k)

        concated = torch.cat(inputs, 1) # (N_c, B, N_k) => (B, N_c * N_k)

        if is_train:
            concated = self.dropout(concated)

        out = self.fc(concated)

        for i in range(6):
           out = self.resnet(out)

        return F.log_softmax(out, 1)

