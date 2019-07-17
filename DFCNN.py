import torch
import torch.nn as nn
from torchsummary import summary
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

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        # conv5 = self.conv5(conv4)
        # shape : (batch_size, channels, windows, dimension) => (batch size, windows, channels, dimension)
        # remember that when you transpose your tensor it only changes your stride which means you should
        # make this tensor contiguous by adding .contiguous()
        conv4 = conv4.transpose(1, 2).contiguous()
        # print(conv5.shape)
        out = conv4.view(-1, conv4.shape[1], self.fc_features)
        # conv4 = self.dropout(conv4)
        # print(conv4.shape)

        out = self.fc1(out)
        out = F.relu(out)
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
    print(model)
    print(model(torch.randn(3, 1, 1600, 200)).shape)
    # if torch.cuda.is_available():
    #     model = model.cuda()

    # summary(model, (1, 1600, 200))





