"""
    Citing from Aggregation-Cross-Entropy
    https://github.com/summerlvsong/Aggregation-Cross-Entropy
    Basically this code is enlightened by ACE idea however
    I modify this code to fit in ASR condition.
"""

import torch
import torch.nn as nn


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()

    def result_analysis(self, iteration):
        pass


class ACE(Sequence):

    def __init__(self, dictionary):
        super(ACE, self).__init__()
        self.softmax = None
        self.label = None
        self.dict = dictionary  # i2w
        self.bs = None
        self.T_ = None
        self.vocab_size = len(self.dict)

    def forward(self, x, label):
        # input shape : (T, B, C), label shape : (B,T) while in scene text recognition (B, H, W, C) (B,HW)
        # here label is still seemed as [pnyid1, pnyid2, ... pnyid_length]
        self.T_, self.bs, _ = x.size()  # modified to adapt into self-contained models

        x = x.transpose(0, 1).contiguous()  # input size (B, T, C)
        x = x + 1e-10

        self.softmax = x
        label = self.reform(label)
        self.label = label

        # ACE Implementation (four funda
        # mental formulas)
        x = torch.sum(x, 1)  # shape : (B,C)
        x = torch.div(x, self.T_)
        label = torch.div(label, self.T_)
        # print(label.shape)
        # print(x.shape)
        assert x.shape == label.shape, "shape not comparable between input and label"
        batch_total_loss = torch.neg(torch.sum(torch.mul(label, torch.log(x))))
        loss = torch.div(batch_total_loss, self.bs)
        # loss = (-torch.sum(torch.log(input)*label))/self.bs
        return loss

    def reform(self, label):
        container = torch.zeros(self.bs, self.vocab_size)
        container = container.float()
        # print(label)
        for batch_idx in range(self.bs):
            length = 0
            for l in label[batch_idx]:
                if int(l.item()) != 0:
                    container[batch_idx][int(l.item())] += 1
                    length += 1
            container[batch_idx][0] = length
        container[:, 0] = self.T_ - container[:, 0]
        # print(container)
        return container

    # def decode_batch(self):
    #     out_best = torch.max(self.softmax, 2)[1].data.cpu().numpy()
    #     pre_result = [0]*self.bs
    #     for j in range(self.bs):
    #         pre_result[j] = out_best[j][out_best[j]!=0]
    #     return pre_result


    # def vis(self,iteration):
    #
    #     sn = random.randint(0,self.bs-1)
    #     print('Test image %4d:' % (iteration*50+sn))
    #
    #     pred = torch.max(self.softmax, 2)[1].data.cpu().numpy()
    #     pred = pred[sn].tolist() # sample #0
    #     pred_string = ''.join(['%2s' % self.dict[pn] for pn in pred])
    #     pred_string_set = [pred_string[i:i+self.w*2] for i in xrange(0, len(pred_string), self.w*2)]
    #     print('Prediction: ')
    #     for pre_str in pred_string_set:
    #         print(pre_str)
    #     label = ''.join(['%2s:%2d'%(self.dict[idx],pn) for idx, pn in enumerate(self.label[sn]) if idx != 0 and pn != 0])
    #     label = 'Label: ' + label
    #     print(label)

if __name__ == "__main__":
    from DFCNN import AcousticModel
    from readdata import SpeechData
    from torch.utils.data import DataLoader
    dev_data = SpeechData('data_config', type='test', dataset="aishell")
    test_loader = DataLoader(dev_data, batch_size=4, shuffle=True)
    model = AcousticModel(dev_data.label_nums(), 200)
    model = model.cuda()
    loss = ACE(dev_data.w2i)
    # print(len(dev_data.w2i))
    for sample in test_loader:
        X, y, input_lengths, label_lengths, transcripts = sample
        X = X.float().cuda()
        out = model(X)
        out = torch.exp(out).cpu()
        l = loss(out, y)
        print(l)
        break
