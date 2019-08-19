import torch
from torch import nn
import numpy as np
from .utils.misc import *
from collections import OrderedDict

class SequenceWise(nn.Module):

    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super().__init__()
        self.module = module

    def forward(self, x, *args, **kwargs):
        t, n = x.size(0), x.size(1)
        x = x.contiguous().view(t * n, -1)
        x = self.module(x, *args, **kwargs)
        x = x.contiguous().view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr

class MaskedSoftmax(nn.Module):

    def __init__(self, dim=-1, epsilon=1e-5):
        super().__init__()
        self.dim = dim
        self.epsilon = epsilon

        self.softmax = nn.Softmax(dim=dim)

    def forward(self, e, mask=None):
        # e: Bx1xTh, mask: BxTh
        if mask is None:
            return self.softmax(e)
        else:
            # masked softmax only in input_seq_len in batch
            # for stability, refered to https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
            shift_e = e - e.max()
            exps = torch.exp(shift_e) * mask
            sums = exps.sum(dim=self.dim, keepdim=True) + self.epsilon
            return (exps / sums)

class Attention(nn.Module):

    def __init__(self, state_vec_size, listen_vec_size, apply_proj=True, proj_hidden_size=256, num_heads=1):
        super().__init__()
        self.apply_proj = apply_proj
        self.num_heads = num_heads

        if apply_proj:
            self.phi = nn.Linear(state_vec_size, proj_hidden_size * num_heads, bias=True)
            # psi should have no bias since h was padded with zero
            self.psi = nn.Linear(listen_vec_size, proj_hidden_size, bias=False)
        else:
            assert state_vec_size == listen_vec_size * num_heads

        if num_heads > 1:
            input_size = listen_vec_size * num_heads
            self.reduce = nn.Linear(input_size, listen_vec_size, bias=True)

        self.normal = SequenceWise(MaskedSoftmax(dim=-1))

    def score(self, m, n):
        """ dot product as score function """
        return torch.bmm(m, n.transpose(1, 2))

    def forward(self, s, h, len_mask=None):
        # s: Bx1xHs -> m: Bx1xHe
        # h: BxThxHh -> n: BxThxHe
        if self.apply_proj:
            m = self.phi(s)
            n = self.psi(h)
        else:
            m = s
            n = h

        # <m, n> -> a, e: Bx1xTh -> c: Bx1xHh
        if self.num_heads > 1:
            proj_hidden_size = m.size(-1) // self.num_heads
            ee = [self.score(mi, n) for mi in torch.split(m, proj_hidden_size, dim=-1)]
            aa = [self.normal(e, len_mask) for e in ee]
            c = self.reduce(torch.cat([torch.bmm(a, h) for a in aa], dim=-1))
            a = torch.stack(aa).transpose(0, 1)
        else:
            e = self.score(m, n)
            a = self.normal(e, len_mask)
            c = torch.bmm(a, h)
            a = a.unsqueeze(dim=1)
        # c: context (Bx1xHh), a: Bxheadsx1xTh
        return c, a

class Speller(nn.Module):

    def __init__(self, listen_vec_size, label_vec_size, max_seq_lens=256, sos=None, eos=None,
                 rnn_type=nn.LSTM, rnn_hidden_size=512, rnn_num_layers=2,
                 proj_hidden_size=256, num_attend_heads=1, masked_attend=True):
        super().__init__()

        assert sos is not None and 0 <= sos < label_vec_size
        assert eos is not None and 0 <= eos < label_vec_size
        assert sos is not None and eos is not None and sos != eos

        self.label_vec_size = label_vec_size
        self.sos = label_vec_size - 2 if sos is None else sos
        self.eos = label_vec_size - 1 if eos is None else eos
        self.max_seq_lens = max_seq_lens
        self.num_eos = 3
        self.tfr = 1.

        Hs, Hc, Hy = rnn_hidden_size, listen_vec_size, label_vec_size

        self.rnn_num_layers = rnn_num_layers
        self.rnns = rnn_type(input_size=(Hy + Hc), hidden_size=Hs, num_layers=rnn_num_layers,
                             bias=True, bidirectional=False, batch_first=True)
        self.norm = nn.LayerNorm(Hs, elementwise_affine=False)

        self.attention = Attention(state_vec_size=Hs, listen_vec_size=Hc,
                                   proj_hidden_size=proj_hidden_size, num_heads=num_attend_heads)

        self.masked_attend = masked_attend

        self.chardist = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(Hs + Hc, 128, bias=True)),
            ('fc2', nn.Linear(128, label_vec_size, bias=False)),
        ]))

        self.softmax = nn.Softmax(dim=-1)

    def get_mask(self, h, seq_lens):
        bs, ts, hs = h.size()
        mask = h.new_ones((bs, ts), dtype=torch.float)
        for b in range(bs):
            mask[b, seq_lens[b]:] = 0.
        return mask

    def _is_sample_step(self):
        return np.random.random_sample() < self.tfr

    def forward(self, h, x_seq_lens, y=None, y_seq_lens=None):
        batch_size = h.size(0)
        sos = int2onehot(h.new_full((batch_size, 1), self.sos), num_classes=self.label_vec_size).float()
        eos = int2onehot(h.new_full((batch_size, 1), self.eos), num_classes=self.label_vec_size).float()

        hidden = None
        y_hats = list()
        attentions = list()

        in_mask = self.get_mask(h, x_seq_lens) if self.masked_attend else None
        x = torch.cat([sos, h.narrow(1, 0, 1)], dim=-1)

        y_hats_seq_lens = torch.ones((batch_size, ), dtype=torch.int) * self.max_seq_lens

        bi = torch.zeros((self.num_eos, batch_size, )).byte()
        if x.is_cuda:
            bi = bi.cuda()

        for t in range(self.max_seq_lens):
            s, hidden = self.rnns(x, hidden)
            s = self.norm(s)
            c, a = self.attention(s, h, in_mask)
            y_hat = self.chardist(torch.cat([s, c], dim=-1))
            y_hat = self.softmax(y_hat)

            y_hats.append(y_hat)
            attentions.append(a)

            # check 3 conjecutive eos occurrences
            bi[t % self.num_eos] = onehot2int(y_hat.squeeze()).eq(self.eos)
            ri = y_hats_seq_lens.gt(t)
            if bi.is_cuda:
                ri = ri.cuda()
            y_hats_seq_lens[bi.prod(dim=0, dtype=torch.uint8) * ri] = t + 1

            # early termination
            if y_hats_seq_lens.le(t + 1).all():
                break

            if y is None or not self._is_sample_step():     # non sampling step
                x = torch.cat([y_hat, c], dim=-1)
            elif t < y.size(1):                             # scheduled sampling step
                x = torch.cat([y.narrow(1, t, 1), c], dim=-1)
            else:
                x = torch.cat([eos, c], dim=-1)

        y_hats = torch.cat(y_hats, dim=1)
        attentions = torch.cat(attentions, dim=2)

        return y_hats, y_hats_seq_lens, attentions
