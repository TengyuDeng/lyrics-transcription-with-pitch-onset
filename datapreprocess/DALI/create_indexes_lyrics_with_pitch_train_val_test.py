import argparse, os, re, shutil
import numpy as np
import DALI as dali_code
import pickle
import h5py
import librosa
import pandas as pd
from tqdm import tqdm

from utils import read_yaml

def main(args):
    workspace = args.workspace
    dali_dir = args.dataset_dir
    gt_path = "/n/work1/deng/data/DALI/info/gt_v1.0_22:11:18.gz"
    config_yaml = args.config_yaml
    configs = read_yaml(config_yaml)
    csv_path = args.csv_path
    df = pd.read_csv(csv_path, index_col=0)
    
    split = configs['split']
    # hdf5s_dir = configs['hdf5s_dir']
    # print(f"hdf5s_dir={hdf5s_dir}")
    threshold = configs['threshold']
    # dali_ids = pickle.load(open("./datapreprocess/DALI/data_ids.pkl", 'rb')) 
    pitch_offsets = df[df['best_res'] < threshold]['best_offset'].astype(int)
    dali_ids = pitch_offsets.index.to_list()
    dali_data = dali_code.get_the_DALI_dataset(dali_dir, gt_path)
    print(f"Creating indexes for {len(dali_ids)} tracks.")

    idx_dir = os.path.join(workspace, "indexes", "DALI", f"config={os.path.split(config_yaml)[1]}")
    print(f"idx_dir={idx_dir}")
    os.makedirs(idx_dir, exist_ok=True)

    random_stat = np.random.RandomState(3756)
    random_stat.shuffle(dali_ids)
    
    train_num = int(len(dali_ids) / np.sum(split) * split[0])
    val_num = int(len(dali_ids) / np.sum(split) * split[1])
    test_num = len(dali_ids) - train_num - val_num
    print(f"{len(dali_ids)} tracks in total, split by {split[0]}:{split[1]}:{split[2]}.\ntrain set: {train_num} tracks, val set: {val_num} tracks, test set: {test_num} tracks.")
    
    train_ids = dali_ids[:train_num]
    val_ids = dali_ids[train_num:train_num + val_num]
    test_ids = dali_ids[train_num + val_num:]
    assert train_num == len(train_ids) and val_num == len(val_ids) and test_num == len(test_ids)
    train_path = os.path.join(idx_dir, "train_idx.pkl")
    val_path = os.path.join(idx_dir, "val_idx.pkl")
    test_path = os.path.join(idx_dir, "test_idx.pkl")
    print("Creating indexes for train set.")
    create_indexes_for_list(dali_data, train_ids, pitch_offsets, train_path, configs)
    print("Creating indexes for val set.")
    create_indexes_for_list(dali_data, val_ids, pitch_offsets, val_path, configs)
    print("Creating indexes for test set.")
    create_indexes_for_list(dali_data, test_ids, pitch_offsets, test_path, configs)
    
    shutil.copy(config_yaml, os.path.join(idx_dir, "config.yaml"))

def create_indexes_for_list(dali_data, dali_ids, pitch_offsets, idx_path, configs):
    indexes = []
    segment_frames = configs['segment_frames']
    sr = configs['sample_rate']
    hop_length = configs['hop_length']
    segment_time = librosa.frames_to_time(segment_frames, sr=sr, hop_length=hop_length)

    for dali_id in tqdm(dali_ids, unit="file"):
        offset = pitch_offsets[dali_id]
        annots_notes = dali_data[dali_id].annotations['annot']['notes']
        annots_words = dali_data[dali_id].annotations['annot']['words']
        # (notes, 2)
        note_time = np.array([annot['time'] for annot in annots_notes]) 
        
        # hdf5_path = os.path.join(hdf5s_dir, f"{dali_id}.h5")
        # with h5py.File(hdf5_path, 'r') as hf:
            # tatum_time = hf['tatum_time'][:]
            # pitch_tatums = hf['pitch_tatums'][:]

        start_frame = 0
        last_time = note_time[-1, -1]
        while True:
            end_frame = start_frame + segment_frames
            start_time, end_time = librosa.frames_to_time((start_frame, end_frame), sr=sr, hop_length=hop_length)
            if end_time > last_time:
                break

            notes_in_segment = np.argwhere((note_time[:, 0] >= start_time) * (note_time[:, 1] < end_time)).flatten()
            if np.sum(note_time[notes_in_segment,1] - note_time[notes_in_segment,0]) / segment_time > 0.1:
                texts_with_pitch = []
                try:
                    current_word_index = annots_notes[notes_in_segment[0]]['index']
                except KeyError:
                    current_word_index = 0
                try:
                    current_line_index = annots_words[current_word_index]['index']
                except KeyError:
                    current_line_index = 0
                for n in notes_in_segment:
                    annot = annots_notes[n]
                    pitch = np.clip(offset + int(librosa.hz_to_midi(annot['freq'][0])), 0, 128)
                    text = []
                    for charac in annot['text']:
                        if charac == "`": charac = "'"
                        if charac in " abcdefghijklmnopqrstuvwxyz,.?!-'":
                            text.append((charac, pitch))
                    try:
                        if current_word_index != annot['index']:
                            current_word_index = annot['index']
                            if current_line_index != annots_words[annot['index']]:
                                current_line_index = annots_words[annot['index']]
                                text.insert(0, (" ", 128))
                            else:
                                text.insert(0, (" ", pitch))

                    except KeyError as e:
                        print(annot)
                    texts_with_pitch.extend(text)

                texts_plain = "".join([charac[0] for charac in texts_with_pitch])

                index = {
                'dali_id': dali_id, 
                'frames': [start_frame, end_frame],
                'text_with_pitch': texts_with_pitch,
                'text': texts_plain,
                'offset': offset,
                }
                indexes.append(index)
                #input(index)

            start_frame += segment_frames
    
    print(f"{len(indexes)} indexes are created in total.")
    print(f"Saving indexes to {idx_path}.")
    pickle.dump(indexes, open(idx_path, 'wb'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--workspace", type=str, default="/n/work1/deng/workspaces/", help="Directory of workspace.")
    parser.add_argument("--dataset_dir", type=str, default="/n/work1/deng/data/DALI", help="Directory of DALI dataset.")
    parser.add_argument("--config_yaml", type=str, default="./datapreprocess/configs/create_indexes.yaml", help="Path to configs.")
    parser.add_argument("--csv_path", type=str, default="./datapreprocess/DALI/offsets_and_results.csv")
    args = parser.parse_args()

    main(args)
