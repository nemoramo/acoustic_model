import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Conv2dEncoder
from readdata import SpeechData
from torch.utils.data import DataLoader

NEG_INF = float('-inf')


def make_mask(seqs, seq_lengths):
    max_frames = torch.max(seq_lengths)
    seqs = seqs[:, :max_frames]
    n_frames = seqs.shape[1]
    print("max_frames : %d, n_frames : %d" % (max_frames, n_frames))
    assert max_frames <= n_frames, "sequence lengths should be smaller than container frames"
    # mask = torch.zeros(n_frames, n_frames).fill_(NEG_INF)
    # mask[:max_frames, :max_frames] = torch.zeros(max_frames, max_frames)
    mask = torch.zeros(max_frames, max_frames)
    key_mask = torch.zeros(seqs.shape[0], max_frames)
    for i in range(len(seq_lengths)):
        key_mask[i, :seq_lengths[i]] = 1
    return seqs, mask.cuda(), (key_mask == 0).cuda()


class ASRTransformer(nn.Module):
    """
    Since Pytorch 1.2.* has already provided torch.nn.Transformer, it is quite useful in this repo.
    """
    def __init__(self, vocab_size, feature_dimension, dmodel, elayers=12, dlayers=6, num_head=4, dropout=0.1):
        super(ASRTransformer, self).__init__()

        self.nhead = num_head
        self.dmodel = dmodel
        assert self.dmodel % self.nhead == 0, "key or value size should equal attention_dim//num_head"
        self.elayers = elayers
        self.dlayers = dlayers

        self.conv_encoder = Conv2dEncoder(indim=feature_dimension, outdim=dmodel, dropout=dropout)
        self.decoder_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=dmodel)
        self.transformer = nn.Transformer(d_model=self.dmodel, nhead=self.nhead, num_encoder_layers=elayers,
                                          num_decoder_layers=dlayers, dropout=dropout)

    def forward(self, src, tgt, seq_lengths=None, target_lengths=None):
        # remember here src size should be (B, C, T, F)
        # x.size => (B, T, attention_dim)

        xs = self.conv_encoder(src)
        seq_lengths = self.conv_encoder.convert(seq_lengths)
        xs, src_mask, src_key_padding_mask = make_mask(xs, seq_lengths)

        xs = xs.transpose(0, 1)

        ys, tgt_mask, tgt_key_padding_mask = make_mask(tgt, target_lengths)
        ys = self.decoder_embedding(ys)
        ys = ys.transpose(0, 1)
        out = self.transformer(xs, ys,
                               src_mask=src_mask, tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)
        return out


if __name__ == "__main__":
    dev_data = SpeechData('data_config', type='test', dataset="aishell")
    dev_loader = DataLoader(dev_data, batch_size=2, shuffle=True)
    model = ASRTransformer(dev_data.label_nums(), 200, dmodel=512).cuda()
    for samples in dev_loader:
        X, y, seq_lengths, target_lengths, trns = samples
        X = X.float().cuda()
        y = y.long().cuda()
        seq_lengths = seq_lengths.cuda()
        target_lengths = target_lengths.cuda()
        out = model(src=X, tgt=y, seq_lengths=seq_lengths, target_lengths=target_lengths)
        print(out)
        break


