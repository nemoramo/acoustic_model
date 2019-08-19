import os
import numpy as np
from file_wav import read_wav_data,GetFrequencyFeature3
from collections import defaultdict
from tqdm import tqdm
import random
from torch.utils.data import Dataset
from prepare import WordDict


class SpeechData(Dataset):
    """
    Implements torch dataset which contains
        __getitem__
        __len__
    """
    def __init__(self, path, WordDict=WordDict, type='train', dataset='thchs', audio_length=1600):
        super(SpeechData, self).__init__()
        self.data_path = path
        self.type = type
        self.dataset = dataset
        self.audio_length = audio_length
        self.wavs = []
        self.transcripts = []
        self.encoding = []
        self.WordDict = WordDict(['data_config/aishell_train.txt', 'data_config/thchs_train.txt'])
        self.WordDict.compose()
        self.w2i, self.i2w = self.WordDict.get_dict()
        self.load_status = 0
        self.data_num = self.__len__()

    def __len__(self):
        if self.load_status == 0:
            print("loading %s data:" % self.type)
            self.load_data()
        return self.data_num

    def label_nums(self):
        if self.load_status == 0:
            print("loading %s data:" % self.type)
            self.load_data()
        return len(self.w2i)

    def load_data(self):
        assert self.type in ['train', 'dev', 'test'], "data type(train, dev, test) should be defined correctly"
        if self.type == 'train':
            filename = self.data_path + '/' + self.dataset + "_train.txt"
        elif self.type == 'dev':
            filename = self.data_path + '/' + self.dataset + "_dev.txt"
        else:
            filename = self.data_path + '/' + self.dataset + "_test.txt"
        with open(filename, 'r') as f:
            for line in f:
                items = line.rstrip('\n').split('\t')
                self.wavs.append('../' + items[0])
                self.transcripts.append(items[1])
        self.data_num = len(self.wavs)
        for trn in tqdm(self.transcripts):
            encoding = [self.w2i[word] for word in trn.split(' ')]
            self.encoding.append(encoding)
        self.load_status = 1

    def words2ids(self, trn):
        return [self.w2i[word] for word in trn]

    def ids2words(self, encoding):
        return [self.i2w[index] for index in encoding]

    def __getitem__(self, index):
        # container X,y
        X = np.zeros((self.audio_length, 200, 1))
        y = np.zeros((64,), dtype=np.int16)

        wavfile = self.wavs[index]
        transcripts = self.transcripts[index]
        labels = self.encoding[index]

        wavsignal, fs = read_wav_data(wavfile) # fs = 16000
        # generate features here
        features = GetFrequencyFeature3(wavsignal, fs)
        features = features.reshape(features.shape[0], features.shape[1], 1) # 增加一维


        input_length = len(features)//8+1 # fs/length_of_dataline default 200

        # remember that input length is not converted, should use model.convert(input_lengths) method

        label_length = len(labels)
        X[0:len(features)] = features
        y[0:len(labels)] = labels
        X = np.transpose(X, [2, 0, 1])  # X should fit pytorch data format ( batch_size, channel, H, W)
        return X, y, input_length, label_length, transcripts


if __name__ == "__main__":
    data = SpeechData('data_config')
    # print(list(data.i2w.values()))
    # print(len(data))
    print(data.label_nums())
    from torch.utils.data import DataLoader
    loader = DataLoader(data, batch_size=4, shuffle=True)
    for batch_idx, sample in enumerate(loader):
        # pass
        # print(batch_idx)
        X, y, input_lengths, label_lengths, transcripts = sample
        # # print(X.shape)
        # y = y.reshape((256,))
        # y = y[y!=0]
        # print(index.numpy())
        # print(data.transcripts[index.numpy()[1]])
        # print(transcripts[1])
        # print(input_lengths.shape)
        # print(label_lengths.shape)
        # print(y)
        print(input_lengths)
        break