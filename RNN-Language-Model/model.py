import  torch
from    torch import nn
import  os
device = torch.device("cpu")
# device = torch.device("cuda;0" if torch.cuda.is_available() else "cpu")

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, n_layers = 1, dropout_p = 0.5):

        super(LanguageModel, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout_p)

    def init_weight(self):
        nn.init.xavier_uniform_(self.embed.weight)
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        context = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        return (hidden.to(device), context.to(device))

    def detach_hidden(self, hiddens):
        return list([hidden.detach() for hidden in hiddens])

    def forward(self, inputs, hidden, is_train=False):

        embeds = self.embed(inputs)
        if is_train:
            embeds = self.dropout(embeds)
        out, hidden = self.rnn(embeds, hidden)
        out = self.linear(out.contiguous().view(out.size(0) * out.size(1), -1))
        return out, hidden

