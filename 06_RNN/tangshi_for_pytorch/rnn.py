import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
        # print("inital linear weight ")

class word_embedding(nn.Module):
    def __init__(self,vocab_length , embedding_dim):
        super(word_embedding, self).__init__()
        w_embeding_random_intial = np.random.uniform(-1,1,size=(vocab_length ,embedding_dim))
        self.word_embedding = nn.Embedding(vocab_length,embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(w_embeding_random_intial))
    def forward(self,input_sentence):
        sen_embed = self.word_embedding(input_sentence)
        return sen_embed


class RNN_model(nn.Module):
    def __init__(self, batch_sz ,vocab_len ,word_embedding,embedding_dim, lstm_hidden_dim):
        super(RNN_model,self).__init__()
        self.word_embedding_lookup = word_embedding
        self.batch_size = batch_sz
        self.vocab_length = vocab_len
        self.word_embedding_dim = embedding_dim
        self.lstm_dim = lstm_hidden_dim
        self.rnn_lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, num_layers = 2)
        self.fc = nn.Linear(lstm_hidden_dim, vocab_len )
        self.apply(weights_init)
        self.softmax = nn.functional.log_softmax
        self.tanh = nn.Tanh()
        
    def forward(self,sentence,is_test = False):
        batch_input = self.word_embedding_lookup(sentence).view(1,-1 ,self.word_embedding_dim)
        h0 = Variable(torch.zeros(2, batch_input.size(1), self.lstm_dim))
        c0 = Variable(torch.zeros(2, batch_input.size(1), self.lstm_dim))
        output, _ = self.rnn_lstm(batch_input, (h0, c0))
        out = output.contiguous().view(-1,self.lstm_dim)
        out = F.relu(self.fc(out))
        out = self.softmax(input=out, dim=1)
        if is_test:
            prediction = out[ -1, : ].view(1,-1)
            output = prediction
        else:
           output = out
        return output
