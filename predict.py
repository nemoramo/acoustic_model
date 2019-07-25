from torch.utils.data import DataLoader
import numpy as np
import torch
from readdata import SpeechData
import Levenshtein as Lev
import time


def moveblank(probs_seq):
    if probs_seq.ndim == 2:
        probs_seq = np.hstack((probs_seq[:, 1:], probs_seq[:, 0].reshape(-1, 1)))
    else:  # with batch
        probs_seq = np.hstack((probs_seq[:, :, 1:], probs_seq[:, :, 0].reshape(probs_seq.shape[0], -1, 1)))
    return probs_seq


def predict(dataloader, classes):
    # self.contained decoder function which is not efficient
    from BeamSearch import decode
    model = torch.load("logs/redrop_am.pt")
    model.eval()
    model = model.cuda()
    print(model)
    dis = 0.0
    total_words = 0
    for samples in dataloader:
        X, y, _, _, trns = samples
        X = X.float().cuda()
        with torch.no_grad():
            outputs = model(X).cpu()

        y = y[y != 0]
        outputs = outputs.reshape((outputs.shape[0], -1)).numpy()
        start = time.time()
        beam_results, _ = decode(outputs)
        print(list(beam_results))
        print(list(y.numpy()))
        print(time.time()-start)
        break
    #         pred = [chr(p) for p in pred]
    #         trns = [chr(t) for t in y]
    #         d = Lev.distance(''.join(pred), ''.join(trns))
    #         total_words += len(pred)
    #         dis += d
    # print(dis/total_words)


def predict2(dataloader, classes):
    # Thanks to baidu's DeepSpeech, little modification made this function quite useful
    from ctcDecode import ctc_beam_search_decoder
    model = torch.load("logs/redrop_am.pt")
    model.eval()
    model = model.cuda()
    print(model)
    dis = 0.0
    total_words = 0
    for samples in dataloader:
        X, y, _, _, trns = samples
        X = X.float().cuda()
        with torch.no_grad():
            outputs = model(X).cpu()

        y = y[y != 0]
        outputs = outputs.reshape((outputs.shape[0], -1)).numpy()
        outputs = moveblank(outputs)
        outputs = np.exp(outputs)
        start = time.time()
        beam_results = ctc_beam_search_decoder(outputs, beam_size=30, vocabulary=classes[1:])
        print(list(beam_results))
        print(list(trns)[0].split(' '))
        print("Prediction using %.5fs" % (time.time() - start))
        break


if __name__ == "__main__":
    dev_data = SpeechData('data_config', type='test', dataset="aishell")
    test_loader = DataLoader(dev_data, batch_size=1, shuffle=True)
    # predict(test_loader, list(dev_data.i2w.keys()))
    predict2(test_loader, list(dev_data.i2w.values()))
