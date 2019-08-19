import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from fastNLP.modules.encoder.transformer import TransformerEncoder
from utils import CNNFeatureExtractor


class TransformerDecoder(nn.Module):
    def __init__(self):
        super(TransformerDecoder, self).__init__()


class ASRTransformer(nn.Module):
    def __init__(self, vocab_size, feature_dimension):
        super(ASRTransformer, self).__init__()
        self.featurizer = CNNFeatureExtractor()
        self.fc = nn.Linear(feature_dimension, )
        self.encoder = TransformerEncoder()
        self.decoder = TransformerDecoder()