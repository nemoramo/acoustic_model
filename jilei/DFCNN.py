import math
import torch
import torch.nn as nn
from torchsummary import summary
from torch.autograd import Variable
import torch.nn.functional as F



def convolution(in_channels, filter_size, bias=True):
    # if input shape = (32,32,3) means input_size = 32, in_channels = 3
    # so suppose you have a filter_size = 10 , kernel_size = 3, padding = 1, stride = 1
    # then your layer output_size = (32-3+2)/1+1 = 32
    # which means you keep the shape consistent with input shape
    # return shape (batch_size, filter_size, input_size, input_size)
    conv = nn.Conv2d(in_channels=in_channels, out_channels=filter_size, kernel_size=(3, 3), padding=1, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    return conv


def normalization(num_features):
    # batch normalization
    # suppose input being of shape (batch_size , channels, input_size, input_size)
    # num_features should be channels
    return nn.BatchNorm2d(num_features)


def maxpooling(kernel_size):
    return nn.MaxPool2d(kernel_size=kernel_size)


def fclayer(in_features, out_features):
    fc = nn.Linear(in_features, out_features)
    nn.init.kaiming_normal_(fc.weight)
    return fc


def GRU(in_features, out_features, bidirectional=True, num_layers=1, dropout=0.1):
    rnn = nn.GRU(input_size=in_features, hidden_size=out_features,
                 bidirectional=bidirectional, dropout=dropout,
                 num_layers=num_layers, batch_first=True)
    for name, param in rnn.named_parameters():
        if 'weight_ih' in name:
            nn.init.kaiming_normal_(param.data)
        elif 'weight_hh' in name:
            nn.init.xavier_uniform_(param.data)
        else:
            param.data.fill_(0)
    return rnn


def LSTM(in_features, out_features, bidirectional=True, num_layers=1, dropout=0.1):
    rnn = nn.LSTM(input_size=in_features, hidden_size=out_features,
                  bidirectional=bidirectional, dropout=dropout,
                  num_layers=num_layers, batch_first=True)
    # print(rnn.all_weights)
    for name, param in rnn.named_parameters():
        if 'weight_ih' in name:
            nn.init.kaiming_normal_(param.data)
        elif 'weight_hh' in name:
            nn.init.xavier_normal_(param.data)
        else:
            param.data.fill_(0)
    return rnn

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
        input = input.transpose(0,1)
        # print("input:", input.shape)
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
        # print("x:", x.shape)
        x = x.transpose(0,1)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'n_features=' + str(self.n_features) \
               + ', context=' + str(self.context) + ')'



class AcousticModel(nn.Module):
    """
    4 layers of convolution and 4 layers of lstm with 3 fully connected layers.
    """
    def __init__(self, vocab_size, input_dimension):
        super(AcousticModel, self).__init__()
        self.vocab_size = vocab_size
        self.input_dimension = input_dimension
        self.dropout = nn.Dropout(p=0.4)

        conv1 = nn.Sequential()
        conv1.add_module('conv1_conv1', convolution(in_channels=1, filter_size=32, bias=False))
        conv1.add_module('conv1_norm1', normalization(num_features=32))
        conv1.add_module('conv1_relu1', nn.ReLU())
        conv1.add_module('conv1_dropout1', nn.Dropout(p=0.1))
        conv1.add_module('conv1_conv2', convolution(in_channels=32, filter_size=32))
        conv1.add_module('conv1_norm2', normalization(num_features=32))
        conv1.add_module('conv1_relu2', nn.ReLU())
        conv1.add_module('conv1_maxpool', maxpooling(2))
        conv1.add_module('conv1_dropout2', nn.Dropout(p=0.1))
        self.conv1 = conv1

        conv2 = nn.Sequential()
        conv2.add_module('conv2_conv1', convolution(in_channels=32, filter_size=64))
        conv2.add_module('conv2_norm1', normalization(num_features=64))
        conv2.add_module('conv2_relu1', nn.ReLU())
        conv2.add_module('conv2_dropout1', nn.Dropout(p=0.2))
        conv2.add_module('conv2_conv2', convolution(in_channels=64, filter_size=64))
        conv2.add_module('conv2_norm2', normalization(num_features=64))
        conv2.add_module('conv2_relu2', nn.ReLU())
        conv2.add_module('conv2_maxpool', maxpooling(2))
        conv2.add_module('conv2_dropout2', nn.Dropout(p=0.2))
        self.conv2 = conv2

        conv3 = nn.Sequential()
        conv3.add_module('conv3_conv1', convolution(in_channels=64, filter_size=128))
        conv3.add_module('conv3_relu1', nn.ReLU())
        conv3.add_module('conv3_dropout1', nn.Dropout(p=0.3))
        conv3.add_module('conv3_norm1', normalization(num_features=128))
        conv3.add_module('conv3_conv2', convolution(in_channels=128, filter_size=128))
        conv3.add_module('conv3_norm2', normalization(num_features=128))
        conv3.add_module('conv3_relu2', nn.ReLU())
        conv3.add_module('conv3_maxpool', maxpooling(2))
        conv3.add_module('conv3_dropout2', nn.Dropout(p=0.3))
        self.conv3 = conv3

        conv4 = nn.Sequential()
        conv4.add_module('conv4_conv1', convolution(in_channels=128, filter_size=128))
        conv4.add_module('conv4_norm1', normalization(num_features=128))
        conv4.add_module('conv4_relu1', nn.ReLU())
        conv4.add_module('conv4_dropout1', nn.Dropout(p=0.4))
        conv4.add_module('conv4_conv2', convolution(in_channels=128, filter_size=128))
        conv4.add_module('conv4_relu2', nn.ReLU())
        conv4.add_module('conv4_conv3', convolution(in_channels=128, filter_size=128))
        conv4.add_module('conv4_norm2', normalization(num_features=128))
        conv4.add_module('conv4_relu3', nn.ReLU())
        conv4.add_module('conv4_dropout2', nn.Dropout(p=0.4))
        self.conv4 = conv4  # no maxpooling

        # conv5 = nn.Sequential()
        # conv5.add_module('conv5_conv1', convolution(in_channels=128, filter_size=256))
        # conv5.add_module('conv5_relu1', nn.ReLU())
        # # conv4.add_module('conv4_norm1', normalization(num_features=128))
        # # conv4.add_module('conv4_dropout1', nn.Dropout(p=0.4))
        # conv5.add_module('conv5_conv2', convolution(in_channels=256, filter_size=256))
        # conv5.add_module('conv5_relu2', nn.ReLU())
        # conv5.add_module('conv5_conv3', convolution(in_channels=256, filter_size=256))
        # conv5.add_module('conv5_relu3', nn.ReLU())
        # # conv4.add_module('conv4_norm2', normalization(num_features=128))
        # # conv4.add_module('conv4_dropout2', nn.Dropout(p=0.4))
        # self.conv5 = conv5  # no maxpooling

        self.fc_features = int(input_dimension / 8 * 128)  # due to three times of pooling 2**3 = 8
        self.fc1 = fclayer(in_features=self.fc_features, out_features=128)
        self.fc2 = fclayer(in_features=256, out_features=128)
        self.fc3 = fclayer(in_features=128, out_features=vocab_size)
        self.rnn = LSTM(in_features=128, out_features=128, num_layers=2)
        self.lookahead = Lookahead(n_features=128, context=20)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        # print(conv4.shape)
        # conv5 = self.conv5(conv4)
        # shape : (batch_size, channels, windows, dimension) => (batch size, windows, channels, dimension)
        # remember that when you transpose your tensor it only changes your stride which means you should
        # make this tensor contiguous by adding .contiguous()

        """
            [3, 128, 200, 25] -> [3, 200, 3200]

        """
        conv4 = conv4.transpose(1, 2).contiguous()

        # print(conv5.shape)
        out = conv4.view(-1, conv4.shape[1], self.fc_features)
        print(out.shape)
        # conv4 = self.dropout(conv4)
        # print(conv4.shape)

        out = self.fc1(out)
        out = F.relu(out)
        print("out:",out.shape)
        out = self.lookahead(out)
        print("out:", out.shape)
        out, (h_n, c_n) = self.rnn(out)
        # print(out.shape)
        # out = self.dropout(out)
        out = self.fc2(out)
        out = F.relu(out)
        # print(out.shape)
        out = self.dropout(out)
        out = self.fc3(out)
        out = F.log_softmax(out, dim=-1)
        out = out.transpose(0, 1).contiguous()  # (input_length, batch_size, number_classes) for ctc loss
        return out


if __name__ == "__main__":
    model = AcousticModel(1000, 200)
    # print(model)
    for para in model.named_parameters():
        print(para[0],para[1].shape)
    print(model(torch.randn(3, 1, 1600, 200)).shape)
    # if torch.cuda.is_available():
    #     model = model.cuda()

    # summary(model, (1, 1600, 200))





