import argparse, os, shutil
import numpy as np
import DALI as dali_code
import h5py
from tqdm import tqdm
import librosa
import pickle
import time

from utils import read_yaml, freq2pitch, wav2spec, wav2mel

def midi_to_tatums(midi_notes, tatum_time):
    pitch_tatums = np.full(len(tatum_time), 128, dtype=int)
    onset_tatums = np.full(len(tatum_time), 0, dtype=int)
    i = 0
    j = 0
    pitch_start = 0
    while True:
        if i >= len(midi_notes) or j >= len(tatum_time) - 1:
            break
    
        while True:
            if i >= len(midi_notes) or j >= len(tatum_time) - 1:
                break
        
            start_diff1 = midi_notes[i][0] - tatum_time[j]
            start_diff2 = tatum_time[j + 1] - midi_notes[i][0]
            start_in_tatum = start_diff1 >= 0 and start_diff2 >= 0

            if start_in_tatum:
                pitch_start = j if start_diff1 < start_diff2 else j + 1
                onset_tatums[pitch_start] = 1
                if midi_notes[i][1] > tatum_time[j + 1]:
                    break
            end_diff1 = midi_notes[i][1] - tatum_time[j]
            end_diff2 = tatum_time[j + 1] - midi_notes[i][1]
            end_in_tatum = end_diff1 >= 0 and end_diff2 >= 0
            if end_in_tatum:
                pitch_end = j if end_diff1 < end_diff2 else j + 1
                pitch_tatums[pitch_start: pitch_end] = midi_notes[i][2]
                onset_tatums[pitch_end] = 1
                i += 1
            else:
                break

        j += 1

    return pitch_tatums, onset_tatums

def main(args):
    workspace = args.workspace
    dali_dir = args.dataset_dir
    gt_path = "/n/work1/deng/data/DALI/info/gt_v1.0_22:11:18.gz"
    config_yaml = args.config_yaml
    configs = read_yaml(config_yaml)
    sr = configs['sample_rate']
    hop_length = configs['hop_length']
    dali_ids = pickle.load(open("./datapreprocess/DALI/data_ids.pkl", 'rb'))
    hdf5s_dir = os.path.join(workspace, "hdf5s", "DALI", f"config={os.path.split(config_yaml)[1]}")
    print(f"hdf5s_dir={hdf5s_dir}")
    os.makedirs(hdf5s_dir, exist_ok=True)
    dali_data = dali_code.get_the_DALI_dataset(dali_dir, gt_path)

    print(f"Creating targets for {len(dali_ids)} downloaded tracks.")
    
    for dali_id in tqdm(dali_ids, unit="file"):
        entry = dali_data[dali_id]
        hdf5_path = os.path.join(hdf5s_dir, f"{dali_id}.h5")
        waveform, _ = librosa.load(os.path.join(dali_dir, "wav", f"sr={sr}", f"{dali_id}.wav"), sr=None, mono=True)
        waveform_sep, _ = librosa.load(os.path.join(dali_dir, "separated", f"{dali_id}.wav"), sr=sr, mono=True)
        create_target(entry, hdf5_path, waveform, waveform_sep, configs)

    shutil.copy(config_yaml, os.path.join(hdf5s_dir, "config.yaml"))

def create_target(entry, hdf5_path, waveform, waveform_sep, configs):
    # start_time = time.time()
    sr = configs['sample_rate']
    hop_length = configs['hop_length']
    
    length = len(waveform) / sr
    
    annot_notes = []
    for annot in entry.annotations['annot']['notes']:
        freq = annot['freq'][0]
        pitch_id = freq2pitch(freq)
        time_seg = annot['time']

        if time_seg[1] < length and time_seg[1] > time_seg[0] and time_seg[0] >= 0:
            if pitch_id >= 0 and pitch_id < 128:
                note_item = [time_seg[0], time_seg[1], pitch_id]
                annot_notes.append(note_item)
        else:
            print(f"Index error! annot=\n{annot}")
    
    tempo, beats = librosa.beat.beat_track(y=waveform, sr=sr, hop_length=hop_length, units="time")
    tatum_rate = 4
    tatum_time = np.interp(
        np.linspace(1, len(beats), (len(beats) - 1) * tatum_rate + 1),
        xp = np.linspace(1, len(beats), len(beats)),
        fp = beats,
        )
    pitch_tatums, onset_tatums = midi_to_tatums(annot_notes, tatum_time)
    
    with h5py.File(hdf5_path, 'w') as hf:
        hf.attrs.create("sample_rate", data=sr, dtype=np.int64)
        hf.attrs.create("hop_length", data=hop_length, dtype=np.int64)
        hf.create_dataset(name="waveform", data=waveform, dtype=np.float32)
        hf.create_dataset(name="waveform_separated", data=waveform_sep, dtype=np.float32)
        hf.create_dataset(name="annot_notes", data=annot_notes)
        hf.create_dataset(name="tatum_time", data=tatum_time, dtype=np.float32)
        hf.create_dataset(name="pitch_tatums", data=pitch_tatums, dtype=np.int64)
        hf.create_dataset(name="onset_tatums", data=onset_tatums, dtype=np.int64)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--workspace", type=str, default="/n/work1/deng/workspaces/", help="Directory of workspace.")
    parser.add_argument("--dataset_dir", type=str, default="/n/work1/deng/data/DALI", help="Directory of DALI dataset.")
    parser.add_argument("--config_yaml", type=str, default="./datapreprocess/configs/create_hdf5s.yaml", help="Path to configs.")
    args = parser.parse_args()

    main(args)
