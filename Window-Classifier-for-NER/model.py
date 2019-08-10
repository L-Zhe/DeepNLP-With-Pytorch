import  torch
from    torch import nn
from    torch.nn import functional as F

class WindowClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_size, window_size, hidden_size, output_size):

        super(WindowClassifier, self).__init__()

        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.hidden_layer1 = nn.Linear(embedding_size * (window_size * 2 + 1), hidden_size)
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, inputs, is_training=False):
        embeds = self.embed(inputs)
        concated = embeds.view(-1, embeds.size(1)*embeds.size(2))
        h0 = self.relu(self.hidden_layer1(concated))
        if is_training:
            h0 = self.dropout(h0)
        h1 = self.relu(self.hidden_layer2(h0))
        if is_training:
            h1 = self.dropout(h1)

        out = self.softmax(self.output_layer(h1))

        return out
