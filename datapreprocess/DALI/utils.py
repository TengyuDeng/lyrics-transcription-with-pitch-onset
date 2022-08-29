import os
import yaml
import numpy as np
import torch
import librosa

def read_yaml(config_yaml: str):

    with open(config_yaml, "r") as fr:
        configs = yaml.load(fr, Loader=yaml.FullLoader)

    return configs

def wav2spec(waveform, **paras):
    try:
        spec = np.abs(librosa.stft(waveform, **paras))
    except Exception as e:
        print(f"Error for transforming to Spectrogram!")
        print(self.indexes[i])
        return [0]
    return spec

def wav2mel(waveform, sr, **paras):
    try:
        # mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=2048, hop_length=512, win_length=2048)
        mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sr, **paras)
    except Exception as e:
        print(f"Error for transforming to Mel spectrogram!")
        return [0]
    return mel_spec

# 128 classes + 1 rest
def freq2pitch(freq):
    return int(np.round(12 * (np.log2(freq) - np.log2(440.)))) + 69

def pitch2freq(class_id):
    return 440. * 2 ** ((class_id - 69) / 12)

    