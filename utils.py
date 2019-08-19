import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchRNN(nn.Module):
    """
    RNN layer with BatchNormalization and parameter reset,
    convert input: (batch, windows, in_features) into output: (batch, windows, out_features)
    optional: bidirectional default set to be True which means using BIRNN
    """
    def __init__(self, in_features, out_features, windows=200, rnn_type=nn.LSTM, dropout=0.3, bidirectional=True, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.out_features = out_features
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.batch_norm = batch_norm
        self.in_features = in_features
        self.dropout = dropout
        self.rnn = rnn_type(input_size=in_features, hidden_size=out_features,
                            bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.BatchNorm1d = nn.BatchNorm1d(windows)

    def reset_parameters(self):
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_normal_(param.data)
            elif 'weight_hh' in name:
                nn.init.xavier_normal_(param.data)
            else:
                param.data.fill_(0)

    def forward(self, x):
        self.reset_parameters()
        if self.batch_norm:
            x = self.BatchNorm1d(x)  # (batch_size, windows, features)
        x, _ = self.rnn(x)
        # x = self.dropout(x)
        if self.bidirectional:
            # sum two directions which means (B,W,H*2) => (B,W,H)
            x = x.view(x.shape[0], x.shape[1], 2, -1).sum(2).view(x.shape[0], x.shape[1], -1)
        return x


def fclayer(in_features, out_features):
    """
    fully connected layers or dense layer
    :param in_features: input_dimension => channels * features
    :param out_features: output_dimension
    :return: (batch, windows, channels * features) => (*, *, output_dimension)
    """
    fc = nn.Linear(in_features, out_features)
    nn.init.kaiming_normal_(fc.weight)
    return fc


class ShallowCNN(nn.Module):

    def __init__(self, ):