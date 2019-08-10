import  torch
from    torch import nn
import  torch.nn.functional as F

class Skipgram(nn.Module):

    def __init__(self, vocab_size, projection_dim):
        super(Skipgram, self).__init__()
        self.embedding_v = nn.Embedding(vocab_size, projection_dim)
        self.embedding_u = nn.Embedding(vocab_size, projection_dim)
        self.logsigmoid = nn.LogSigmoid()

        initrange = (2.0 / (vocab_size + projection_dim)) ** 0.5
        self.embedding_v.weight.data.uniform_(-initrange, initrange) # init
        self.embedding_u.weight.data.uniform_(0, 0) # init
        #self.out = nn.Linear(projection_dim,vocab_size)

    def forward(self, center_words, target_words, outer_words):

        center_embeds = self.embedding_v(center_words) # B x 1 x D
        target_embeds = self.embedding_u(target_words) # B x 1 x D
        outer_embeds = self.embedding_u(outer_words) # B x V x D

        positive_score = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2) # Bx1xD * BxDx1 => Bx1
        negative_score = torch.sum(outer_embeds.bmm
                                (center_embeds.transpose(1, 2)).squeeze(2), 1) # BxVxD * BxDx1 => BxV)

        loss = self.logsigmoid(positive_score) + self.logsigmoid(negative_score)

        return -torch.mean(loss)

    def prediction(self, inputs):

        embeds = self.embedding_v(inputs)
        return embeds