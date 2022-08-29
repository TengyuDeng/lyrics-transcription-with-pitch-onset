import librosa
import numpy as np

def get_transform(feature_type, params, harmonics_shift=False):

    if feature_type == "melspectrogram":
        def transform(waveform, sr):
            input_features = wav2mel(waveform, sr=sr, **params)
            return input_features

    elif feature_type == "logspectrogram":
        def transform(waveform, sr):
            spectrogram = wav2spec(waveform, **params)
            if harmonics_shift:
                input_features = harmonics_logspectrogram(spectrogram, sr, **params)
            else:
                log_filter = logspectrogram_filter(sr, **params)
                input_features = np.dot(log_filter, spectrogram)
                return input_features

    elif feature_type == "cqt":
        def transform(waveform, sr):
            input_features = wav2cqt(waveform, sr=sr, harmonics_shift=harmonics_shift, **params)
            return input_features

    else:
        raise ValueError("No such feature type!")

    return transform

def wav2spec(waveform, n_fft, hop_length, **args):
    try:
        spec = np.abs(librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)) ** 2
    except Exception as e:
        print(f"Error for transforming to Spectrogram!")
        print(e)
        return [0]
    return spec

def wav2cqt(waveform, sr, harmonics_shift=True, **cqt_params):
    # Harmonics
    
    if harmonics_shift:
        hs = [0.5, 1, 2, 3, 4, 5, 6]
        cqt_params['fmin'] *= hs[0]
        hs = np.round(np.log2(hs) * cqt_params['bins_per_octave']).astype(int)
        hs -= hs[0]
        old_bins = cqt_params['n_bins']
        cqt_params['n_bins'] = int(cqt_params['n_bins'] + hs[-1]) 
        cqt_features = np.abs(librosa.cqt(y=waveform, sr=sr, **cqt_params)) ** 2

        cqt_features = librosa.power_to_db(cqt_features)
        cqt_h = [cqt_features[shift: shift + old_bins] for shift in hs]
        return np.stack(cqt_h)
    else:
        cqt = librosa.power_to_db(np.abs(librosa.cqt(y=waveform, sr=sr, **cqt_params)) ** 2)
        return cqt

def wav2mel(waveform, sr, **mel_params):
    # Harmonics
    try:
        mel_features = librosa.power_to_db(librosa.feature.melspectrogram(y=waveform, sr=sr, **mel_params))
    except Exception as e:
        print(f"Error for transforming to melspectrogram!")
        print(e)
        return [0]
    
    return mel_features

def logspectrogram_filter(sr=22050, n_fft=2048, pitch_min=0, pitch_max=127, n_bins=128, **args):
    freqs = pitch2freq(np.linspace(pitch_min - 0.5, pitch_max + 0.5, n_bins + 1))
    fft_bins = np.round(freqs / sr * n_fft).astype(int)
    log_filter = np.zeros((n_bins, 1 + n_fft//2))
    for i in range(n_bins):
        if fft_bins[i] == fft_bins[i+1]:
            log_filter[i][fft_bins[i]] = 1
        else:
            log_filter[i][fft_bins[i]:fft_bins[i+1]] = 1
    return log_filter

def harmonics_logspectrogram(spectrogram, sr, hs=[0.5, 1, 2, 3, 4, 5], **filter_params):
    features_h = []
    for n in range(len(hs)):
        new_filter_params = filter_params.copy()
        new_filter_params['pitch_min'] += 12 * np.log2(hs[n])
        new_filter_params['pitch_max'] += 12 * np.log2(hs[n])
        log_filter = logspectrogram_filter(sr, **filter_params)
        features = np.dot(log_filter, spectrogram)
        features_h.append(features)
    return np.stack(features_h)
