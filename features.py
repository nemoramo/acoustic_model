import numpy as np
from scipy.io import wavfile as wav
from python_speech_features import mfcc
from scipy.fftpack import fft


def compute_mfcc(file,numcep):
    fs, audio = wav.read(file)
    mfcc_feat = mfcc(audio, samplerate=fs, numcep=numcep)
    mfcc_feat = mfcc_feat[::3]
    mfcc_feat = np.transpose(mfcc_feat)
    return mfcc_feat


def compute_fbank(file, window_size=400, time_window=25, step=10):
    # Hamming window
    # original solution
    ##
    # x = np.linspace(0, window_size - 1, window, dtype=np.int64)
    # w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (window_size - 1))
    ##
    # numpy build-in solution
    w = np.hamming(window_size)
    fs, wavsignal = wav.read(file)
    wav_array = np.array(wavsignal)
    # number of windows should be calculated as:
    windows = int(len(wavsignal) / fs * 1000 - time_window)//step+1
    data_input = np.zeros((windows, int(window_size/2)), dtype=np.float)  # feature matrix container
    for i in range(windows):
        w_start = int(i * window_size*step/time_window)  # window_size/w_start = time_window/shift
        w_end = w_start + window_size
        data_line = wav_array[w_start:w_end]
        data_line = data_line * w  # multiply hamming window
        data_line = np.abs(fft(data_line))
        data_input[i] = data_line[0:int(window_size/2)]  # because the data are symmetric
    data_input = np.log(data_input+1)  # log level according to NTU DIGITAL SPEECH PROCESSING
    return data_input[:data_input.shape[0]//8*8, :]  # dimensions should be 8x
