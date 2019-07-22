from collections import defaultdict
import tqdm
import os


class WordDict:
    def __init__(self, path_list):
        self.paths = path_list
        self.w2i = defaultdict(lambda: len(self.w2i))
        self.i2w = defaultdict()

    def compose(self):
        pad = self.w2i['<PAD>']  # means blank 0
        unk = self.w2i['<UNK>']
        print("Using padding as <PAD> as %d" % pad)
        print("Using unknown as <UNK> as %d" % unk)
        for path in self.paths:
            print("Handling %s" % path)
            with open(path, 'r') as f:
                for line in tqdm.tqdm(f):
                    if line:
                        line = line.split('\t')
                        trns = line[1]
                        for pny in trns.split(' '):
                            _ = self.w2i[pny]
        self.i2w = {v: k for k, v in self.w2i.items()}
        self.w2i = defaultdict(lambda: unk, self.w2i)
        self.i2w = defaultdict(lambda: '<UNK>', self.i2w)

    def get_dict(self):
        return self.w2i, self.i2w

    def encode(self, pnys):
        return [self.w2i[pny] for pny in pnys]

    def decode(self, encoding):
        return [self.i2w[enc] for enc in encoding]


if __name__ == "__main__":
    paths = ['data_config/aishell_train.txt', 'data_config/thchs_train.txt']
    dic = WordDict(paths)
    dic.compose()
    print(dic.w2i)
