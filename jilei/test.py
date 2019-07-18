from .file_wave_test import *

if (__name__ == '__main__'):
    wave_data, framerate = read_wav_data("./test.wav")
    print(wave_data.shape)