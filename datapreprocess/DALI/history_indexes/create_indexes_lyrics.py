import argparse, os, re, shutil
import numpy as np
import DALI as dali_code
import pickle
import h5py
import librosa
from tqdm import tqdm

from utils import read_yaml

def main(args):
    workspace = args.workspace
    dali_dir = args.dataset_dir
    gt_path = "/n/work1/deng/data/DALI/info/gt_v1.0_22_11_18.gz"
    config_yaml = args.config_yaml
    configs = read_yaml(config_yaml)
    
    n_fold = configs['n_fold']
    hdf5s_dir = os.path.join(workspace, "hdf5s", "DALI", f"config={os.path.split(config_yaml)[1]}")
    print(f"hdf5s_dir={hdf5s_dir}")
    dali_data = dali_code.get_the_DALI_dataset(dali_dir, gt_path)
    dali_ids = pickle.load(open("./datapreprocess/data_ids.pkl", 'rb')) 
    print(f"Creating indexes for {len(dali_ids)} downloaded tracks.")

    idx_dir = os.path.join(workspace, "indexes_lyrics", "DALI", f"config={os.path.split(config_yaml)[1]}")
    print(f"idx_dir={idx_dir}")
    os.makedirs(idx_dir, exist_ok=True)

    random_stat = np.random.RandomState(3756)
    random_stat.shuffle(dali_ids)

    fold_idx = np.linspace(0, len(dali_ids), n_fold * 4 + 1).astype(int)
    for i in range(n_fold):
        print(f"------------Fold{i}------------")
        fold_dir = os.path.join(idx_dir, f"fold_{i}")
        os.makedirs(fold_dir, exist_ok=True)
        train_ids = dali_ids[:fold_idx[i * 4]] + dali_ids[fold_idx[i * 4 + 1]:]
        test_ids = dali_ids[fold_idx[i * 4]:fold_idx[i * 4 + 1]]
        train_path = os.path.join(fold_dir, "train_idx.pkl")
        test_path = os.path.join(fold_dir, "test_idx.pkl")
        print("Creating indexes for train set.")
        create_indexes_for_list(dali_data, train_ids, train_path, hdf5s_dir, configs)
        print("Creating indexes for test set.")
        create_indexes_for_list(dali_data, test_ids, test_path, hdf5s_dir, configs)
    
    shutil.copy(config_yaml, os.path.join(idx_dir, "config.yaml"))

def create_indexes_for_list(dali_data, dali_ids, idx_path, hdf5s_dir, configs):
    indexes = []
    segment_tatums = configs['segment_tatums']
    sr = configs['sample_rate']
    hop_length = configs['stft_paras']['hop_length']

    for dali_id in tqdm(dali_ids, unit="file"):
        annots = dali_data[dali_id].annotations['annot']['lines']
        # (words, 2)
        word_frames = librosa.time_to_frames(
            np.array([annot['time'] for annot in annots]), 
            sr=sr, 
            hop_length=hop_length,
            )

        hdf5_path = os.path.join(hdf5s_dir, f"{dali_id}.h5")
        with h5py.File(hdf5_path, 'r') as hf:
            tatum_frames = hf['tatum_frames'][:]

        start_tatum = 0
        while True:
            end_tatum = start_tatum + segment_tatums
            if end_tatum >= len(tatum_frames):
                break

            start_frame = tatum_frames[start_tatum]
            end_frame = tatum_frames[end_tatum]
            words_in_segment = list(np.argwhere((word_frames[:, 0] >= start_frame) * (word_frames[:, 1] < end_frame)).flatten())
            if len(words_in_segment) > 0:
                texts = [annots[n]['text'] for n in words_in_segment]
                text = ' '.join(texts)
                text = text.replace("`", "'")
                text = re.sub(r"[^a-z 0-9 \s , \. ' \- \? !]", "", text)
                index = {
                'dali_id': dali_id, 
                'frames': [start_frame, end_frame],
                'tatum_ids': [start_tatum, end_tatum],
                # 'text': text,
                'words_in_segment': words_in_segment,
                }
                # print(index)
                indexes.append(index)

            start_tatum += segment_tatums
    
    print(f"{len(indexes)} indexes are created in total.")
    print(f"Saving indexes to {idx_path}.")
    pickle.dump(indexes, open(idx_path, 'wb'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--workspace", type=str, default="/n/work1/deng/workspaces/", help="Directory of workspace.")
    parser.add_argument("--dataset_dir", type=str, default="/n/work1/deng/data/DALI", help="Directory of DALI dataset.")
    parser.add_argument("--config_yaml", type=str, default="./datapreprocess/configs/create_hdf5s.yaml", help="Path to configs.")

    args = parser.parse_args()

    main(args)
