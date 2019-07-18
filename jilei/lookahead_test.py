import torch
import math
from torch.autograd import Variable
from torch import nn

class Lookahead(nn.Module):
    # Wang et al 2016 - Lookahead Convolution Layer for Unidirectional Recurrent Neural Networks
    # input shape - sequence, batch, feature - TxNxH
    # output shape - same as input

    def __init__(self, n_features, context):
        # should we handle batch_first=True?
        super().__init__()
        self.n_features = n_features
        self.weight = nn.Parameter(torch.Tensor(n_features, context + 1))
        assert context > 0
        self.context = context
        self.register_parameter('bias', None)
        self.init_parameters()

    def init_parameters(self):  # what's a better way initialiase this layer?
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        seq_len = input.size(0)
        # pad the 0th dimension (T/sequence) with zeroes whose number = context
        # Once pytorch's padding functions have settled, should move to those.
        padding = torch.zeros(self.context, *(input.size()[1:])).type_as(input.data)
        x = torch.cat((input, Variable(padding)), 0)

        # add lookahead windows (with context+1 width) as a fourth dimension
        # for each seq-batch-feature combination
        x = [x[i:i + self.context + 1] for i in range(seq_len)]  # TxLxNxH - sequence, context, batch, feature
        x = torch.stack(x)
        x = x.permute(0, 2, 3, 1)  # TxNxHxL - sequence, batch, feature, context

        x = torch.mul(x, self.weight).sum(dim=3)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'n_features=' + str(self.n_features) \
               + ', context=' + str(self.context) + ')'

if __name__=='__main__':
    W0 = 129
    # C0 = 2 * input_folding
    W1 = (W0 - 41 + 2 * 20) // 2 + 1  # 65
    C1 = 16
    W2 = (W1 - 21 + 2 * 10) // 2 + 1  # 33
    C2 = C1 * 2
    W3 = (W2 - 11 + 2 * 5) // 2 + 1  # 17
    C3 = C2 * 2

    rnn_hidden_size = 3

    H0 = [C3 * W3, rnn_hidden_size, rnn_hidden_size]
    H1 = rnn_hidden_size
    context = 20

    lookahead = Lookahead(H1, context)
    x = torch.randn(3,1,3)
    y = lookahead(x)
    print(x)
    print(y)
    # print(y.shape)