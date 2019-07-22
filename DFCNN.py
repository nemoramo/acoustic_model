import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
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


def normalLSTM(in_features, out_features, num_layers=2, dropout=0.1):
    rnn = nn.LSTM(in_features, out_features, num_layers=num_layers,
                  dropout=dropout, bidirectional=True, batch_first=True)
    for name, param in rnn.named_parameters():
        if 'weight_ih' in name:
            nn.init.kaiming_normal_(param.data)
        elif 'weight_hh' in name:
            nn.init.xavier_normal_(param.data)
        else:
            param.data.fill_(0)
    return rnn


class BatchRNN(nn.Module):
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


class AcousticModel(nn.Module):
    """
    4 layers of convolution and 4 layers of lstm with 2 fully connected layers.
    """
    def __init__(self, vocab_size, input_dimension):
        super(AcousticModel, self).__init__()
        self.vocab_size = vocab_size
        self.input_dimension = input_dimension
        self.dropout = nn.Dropout(p=0.5)
        self.num_rnn_layers = 4

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
        conv2.add_module('conv2_dropout1', nn.Dropout(p=0.1))
        conv2.add_module('conv2_conv2', convolution(in_channels=64, filter_size=64))
        conv2.add_module('conv2_norm2', normalization(num_features=64))
        conv2.add_module('conv2_relu2', nn.ReLU())
        conv2.add_module('conv2_maxpool', maxpooling(2))
        conv2.add_module('conv2_dropout2', nn.Dropout(p=0.1))
        self.conv2 = conv2

        conv3 = nn.Sequential()
        conv3.add_module('conv3_conv1', convolution(in_channels=64, filter_size=128))
        conv3.add_module('conv3_relu1', nn.ReLU())
        conv3.add_module('conv3_dropout1', nn.Dropout(p=0.2))
        conv3.add_module('conv3_norm1', normalization(num_features=128))
        conv3.add_module('conv3_conv2', convolution(in_channels=128, filter_size=128))
        conv3.add_module('conv3_norm2', normalization(num_features=128))
        conv3.add_module('conv3_relu2', nn.ReLU())
        conv3.add_module('conv3_maxpool', maxpooling(2))
        conv3.add_module('conv3_dropout2', nn.Dropout(p=0.2))
        self.conv3 = conv3

        conv4 = nn.Sequential()
        conv4.add_module('conv4_conv1', convolution(in_channels=128, filter_size=128))
        conv4.add_module('conv4_norm1', normalization(num_features=128))
        conv4.add_module('conv4_relu1', nn.ReLU())
        conv4.add_module('conv4_dropout1', nn.Dropout(p=0.2))
        conv4.add_module('conv4_conv2', convolution(in_channels=128, filter_size=128))
        conv4.add_module('conv4_relu2', nn.ReLU())
        conv4.add_module('conv4_conv3', convolution(in_channels=128, filter_size=128))
        conv4.add_module('conv4_norm2', normalization(num_features=128))
        conv4.add_module('conv4_relu3', nn.ReLU())
        conv4.add_module('conv4_dropout2', nn.Dropout(p=0.2))
        self.conv4 = conv4  # no maxpooling

        self.fc_features = int(input_dimension / 8 * 128)  # due to three times of pooling 2**3 = 8
        self.fc1 = fclayer(in_features=self.fc_features, out_features=128)
        self.fc2 = fclayer(in_features=256, out_features=128)
        self.fc3 = fclayer(in_features=128, out_features=vocab_size)
        # self.fc3 = fclayer(in_features=512, out_features=vocab_size)
        # self.StackBatchRNN = nn.Sequential()
        # self.StackBatchRNN.add_module('BatchRNN_1', BatchRNN(in_features=128,
        #                                                      out_features=512,
        #                                                      windows=input_dimension))
        # for i in range(2, 2+self.num_rnn_layers-1):
        #     self.StackBatchRNN.add_module('BatchRNN_'+str(i), BatchRNN(in_features=512,
        #                                                                out_features=512,
        #                                                                windows=input_dimension))

        self.rnn = normalLSTM(in_features=128, out_features=128, num_layers=4)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        # shape : (batch_size, channels, windows, dimension) => (batch size, windows, channels, dimension)
        # remember that when you transpose your tensor it only changes your stride which means you should
        # make this tensor contiguous by adding .contiguous()
        conv4 = conv4.transpose(1, 2).contiguous()
        out = conv4.view(-1, conv4.shape[1], self.fc_features)
        # conv4 = self.dropout(conv4)
        # print(conv4.shape)

        out = self.fc1(out)
        out = F.relu(out)
        # out shape: (batch_size,200,128)
        out, _ = self.rnn(out)
        # out shape: (batch_size,200,256)
        out = self.fc2(out)
        out = self.fc3(out)
        out = F.log_softmax(out, dim=-1)
        out = out.transpose(0, 1).contiguous()  # (input_length, batch_size, number_classes) for ctc loss
        return out

    def convert(self, input_lengths):
        return input_lengths//8 + 1


if __name__ == "__main__":
    model = AcousticModel(1000, 200)
    print(model)
    dummy_input = torch.randn(3, 1, 1600, 200)
    with SummaryWriter(comment="DFCNN_LSTM") as w:
        w.add_graph(model, (dummy_input,))
    # if torch.cuda.is_available():
    #     model = model.cuda()

    # summary(model, (1, 1600, 200))





