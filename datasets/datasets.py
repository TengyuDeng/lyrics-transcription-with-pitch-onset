import os
import sys
sys.path.append("..")

import numpy as np
import torch, torchaudio
import librosa
import pickle
import h5py
from decoders import Mycodec
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

seed = 65324

class ConvFilter:
    def __init__(self, half_length=5, v_max=1):
        self.padding = half_length - 1
        self.conv_weights = self._get_conv_weight(half_length, v_max)

    def __call__(self, x):
        # Apply the filter on the last dimension
        # x: (batch_size, in_channel=1, length)
        # output: (batch_size, out_channel=1, length)

        return torch.nn.functional.conv1d(x, self.conv_weights, padding=self.padding)
    
    def _get_conv_weight(self, half_length, v_max):
        weights = np.linspace(0, v_max, half_length)
        weights = np.append(weights, weights[-2::-1])[None, None, :]
        weights = torch.tensor(weights, dtype=torch.float32)
        return weights

class MyDataset(torch.utils.data.Dataset):
    
    def __init__(self, name, indexes_path, hdf5s_dir, transform,
        text_only=False,
        target_type="word",
        frame_based=False,
        pitch_shift=False, 
        separated=False,
        separated_and_mix=False,
        with_offset=False,
        onset_filter=None,
        pitch_filter=None,
        min_pitch_label=0,
        max_pitch_label=128,
        **args):

        super().__init__()
        self.name = name
        self.indexes = pickle.load(open(indexes_path, 'rb'))

        self.hdf5s_dir = hdf5s_dir
        self.target_type = target_type
        if self.target_type == "with_pitch":
            self.codec = Mycodec(self.target_type, min_pitch_label, max_pitch_label)
        else:
            self.codec = Mycodec(self.target_type)
        self.transform = transform

        self.text_only = text_only
        self.frame_based = frame_based
        self.pitch_shift = pitch_shift
        self.separated = separated
        self.separated_and_mix = separated_and_mix
        self.with_offset = with_offset
        self.onset_filter = ConvFilter(**onset_filter) if onset_filter is not None else None
        self.pitch_filter = ConvFilter(**pitch_filter) if pitch_filter is not None else None
        self.rand = np.random.RandomState(seed)

    def __len__(self):

        if self.name == "RWC" and self.pitch_shift:
            return len(self.indexes) * 25
        else:
            return len(self.indexes)

    def __getitem__(self, i):
        # start_time = time.time()
        if self.name == "RWC":
            index_id = i // 25 if self.pitch_shift else i
            track_id = self.indexes[index_id]['track_id']
            if 'frames' in self.indexes[index_id]:
                frames = self.indexes[index_id]['frames']
                tatum_ids = None
            else:
                frames = None
                tatum_ids = self.indexes[index_id]['tatum_ids']
            text = " "
            pitch_offset = 0
            hdf5_path = os.path.join(self.hdf5s_dir, f"p{track_id:0>3}.h5")
            hdf5s_dir_sep = self.hdf5s_dir[:self.hdf5s_dir.rfind(".")] + "_separated" + self.hdf5s_dir[self.hdf5s_dir.rfind("."):]
            hdf5_path_sep = os.path.join(hdf5s_dir_sep, f"p{track_id:0>3}.h5")
        
        elif self.name == "DALI":
            dali_id = self.indexes[i]['dali_id']
            if 'frames' in self.indexes[i]:
                frames = self.indexes[i]['frames']
                tatum_ids = None
            else:
                frames = None
                tatum_ids = self.indexes[i]['tatum_ids']
            if self.target_type == "with_pitch":
                text = self.indexes[i]['text_with_pitch'] 
            else:
                text = self.indexes[i]['text']
            pitch_offset = self.indexes[i]['offset']
            hdf5_path = os.path.join(self.hdf5s_dir, f"{dali_id}.h5")

        else:
            raise RuntimeError("Task not supported!")
        
        # if self.target_type == "phoneme":
        #     text = self.codec.phonemize(text)
        text = self.codec.encode(text)

        with h5py.File(hdf5_path, 'r') as hf:
            if self.name == "RWC":
                with h5py.File(hdf5_path_sep, 'r') as hf_sep:
                    input_features, pitches, onsets, sr, hop_length, tatum_time = self._get_data_from_hdf5(hf, pitch_offset, tatum_ids, frames, hf_sep=hf_sep)
            else:
                input_features, pitches, onsets, sr, hop_length, tatum_time = self._get_data_from_hdf5(hf, pitch_offset, tatum_ids, frames)
        
        # tatum frames:
        tatum_frames = librosa.time_to_frames(tatum_time, sr=sr, hop_length=hop_length)
        # print(f"getitem using time {time.time() - start_time}")
        
        # targets:
        if self.text_only:
            return (
                input_features,
                torch.tensor(text, dtype=torch.int),
                )
        else:
            return (
                input_features, pitches, onsets,
                torch.tensor(text, dtype=torch.int),
                )

    def _get_data_from_hdf5(self, hf, pitch_offset, tatum_ids, frames, hf_sep=None):
        sr = hf.attrs['sample_rate']
        hop_length = hf.attrs['hop_length']
        pitch_shift = list(range(-12, 13))[i % 25] if self.pitch_shift else 0

        # waveform:
        if tatum_ids is not None:
            samples = librosa.time_to_samples(hf['tatum_time'][tatum_ids], sr=sr)
            frames = librosa.time_to_frames(hf['tatum_time'][tatum_ids], sr=sr, hop_length=hop_length)
        else:
            samples = librosa.frames_to_samples(frames, hop_length=hop_length)
            tatum_ids = [0, 80]
        if self.name == "RWC":
            if pitch_shift == 0:
                waveform_mix = hf['waveform'][samples[0]: samples[1]]
                waveform_sep = hf_sep['waveform'][samples[0]: samples[1]]
            else:
                waveform_mix = hf[f"waveform_shifted_{pitch_shift}"][samples[0]: samples[1]]
                waveform_sep = hf_sep[f"waveform_shifted_{pitch_shift}"][samples[0]: samples[1]]
            
            if self.separated:
                waveform = waveform_sep
            else:
                waveform = waveform_mix

        elif self.name == "DALI":
            waveform_mix = hf['waveform'][samples[0]: samples[1]]
            waveform_sep = hf['waveform_separated'][samples[0]: samples[1]]
            if self.separated:
                waveform = waveform_sep
            else:
                waveform = waveform_mix

        if len(waveform) < samples[1] - samples[0]:
            waveform = np.pad(waveform, (0, samples[1] - samples[0]))

        # input features:
        if self.separated_and_mix:
            input_features_sep = torch.tensor(self.transform(waveform_sep, sr), dtype=torch.float)
            input_features_mix = torch.tensor(self.transform(waveform_mix, sr), dtype=torch.float)
            if input_features_sep.ndim == 3:
                input_features = torch.cat([input_features_sep, input_features_mix], dim=0)
            else:
                input_features = torch.stack([input_features_sep, input_features_mix])
        else:
            input_features = torch.tensor(self.transform(waveform, sr), dtype=torch.float)
            
        # tatum:
        tatum_time = hf['tatum_time'][tatum_ids[0]: tatum_ids[1] + 1]
        tatum_time -= tatum_time[0]

        # targets:
        if self.frame_based:
            midi_notes = hf['midi_notes'][:] if self.name == "RWC" else hf['annot_notes'][:]
            pitches, onsets = self._create_frame_based_targets(midi_notes, sr, hop_length, input_features.shape[-1], frames, pitch_shift, pitch_offset)
        else:
            pitches = hf['pitch_tatums'][tatum_ids[0]: tatum_ids[1]]
            pitches[pitches != 128] += pitch_shift + pitch_offset
            pitches = np.clip(pitches, 0, 128)
            pitches = self._pitch_to_zeroone(pitches)
            onsets = hf['onset_tatums'][tatum_ids[0]: tatum_ids[1]]
            
            pitches = torch.tensor(pitches, dtype=torch.float)
            onsets = torch.tensor(onsets, dtype=torch.float)

        return input_features, pitches, onsets, sr, hop_length, tatum_time

    def _create_frame_based_targets(self, midi_notes, sr, hop_length, frame_length, frames, pitch_shift, pitch_offset):
        pitches_frame = np.full(frame_length, 128, dtype=int)
        onsets_frame = np.full(frame_length, 0, dtype=float)
        for note in midi_notes:
            start_frame, end_frame = librosa.time_to_frames(note[:2], sr=sr, hop_length=hop_length)
            if start_frame >= frames[0] and start_frame < frames[1]:
                start_frame -= frames[0]
                end_frame -= frames[0]
                pitches_frame[start_frame:end_frame] = note[-1]
                onsets_frame[start_frame] = 1
                if self.with_offset and end_frame < frame_length:
                    onsets_frame[end_frame] = 1

        pitches_frame[pitches_frame != 128] += pitch_shift + pitch_offset
        pitches_frame = np.clip(pitches_frame, 0, 128)
        pitches_frame = self._pitch_to_zeroone(pitches_frame)
        
        pitches_frame = torch.tensor(pitches_frame, dtype=torch.float)
        onsets_frame = torch.tensor(onsets_frame, dtype=torch.float)

        if self.pitch_filter is not None:
            pitches_frame_value = self.pitch_filter(pitches_frame[:128, :].T.unsqueeze(1)).squeeze().T
            pitches_frame_silence = pitches_frame[(128,), :]
            pitches_frame = torch.cat([pitches_frame_value, pitches_frame_silence], dim=0)
        if self.onset_filter is not None:
            onsets_frame = self.onset_filter(onsets_frame[None, None, :]).squeeze()
        
        return pitches_frame, onsets_frame

    def _pitch_to_zeroone(self, pitches):
        new_pitches = np.zeros((129, len(pitches)))
        new_pitches[pitches, range(len(pitches))] = 1
        return new_pitches


class MyLibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, urls=["train-clean-360"], text_only=True, channels=1, **args):
        super().__init__()
        self.LibriSpeechDataset = torch.utils.data.ConcatDataset([
            torchaudio.datasets.LIBRISPEECH(
                root=root,
                url=url,
                download=True,
            ) for url in urls
        ])
        self.target_type = "word"
        self.codec = Mycodec(self.target_type)
        self.text_only = text_only
        self.transform = transform
        self.channels=channels

    def __len__(self):
        return len(self.LibriSpeechDataset)

    def __getitem__(self, i):
        # waveform, sr, text, _ , _ , _ = self.LibriSpeechDataset[self.indexes[i]['index']]
        waveform, sr, text, _ , _ , _ = self.LibriSpeechDataset[i]
        text = text.lower()
        
        waveform = waveform.squeeze().numpy()
        input_features = torch.tensor(self.transform(waveform, sr), dtype=torch.float)
        if self.channels > 1:
            input_features = input_features.unsqueeze(0).repeat_interleave(self.channels, dim=0)
        
        if self.target_type == "phoneme":
            text = self.codec.phonemize(text)
        text = self.codec.encode(text)
        
        if self.text_only:
            return (
                input_features,
                torch.tensor(text, dtype=torch.int),
                )
        else:
            frame_length = input_features.shape[-1]
            pitches_frame = np.zeros((129, frame_length))
            onsets_frame = np.zeros((frame_length,))
            return (
                input_features,
                torch.tensor(pitches_frame, dtype=torch.float),
                torch.tensor(onsets_frame, dtype=torch.float),
                torch.tensor(text, dtype=torch.int),
                )
