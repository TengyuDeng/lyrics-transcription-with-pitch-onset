import argparse
import os, shutil
from tqdm import tqdm
import h5py, numpy as np

import librosa

def main(args):

    dataset_dir = args.dataset_dir
    hdf5s_dir = args.hdf5s_dir

    # Paths
    audios_dir = os.path.join(dataset_dir, "mp3")
    wav_dir = os.path.join(dataset_dir, "wav")
    separated_dir = os.path.join(dataset_dir, "separated")
    os.makedirs(hdf5s_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(separated_dir, exist_ok=True)
    
    filenames = sorted(os.listdir(audios_dir))
    
    print("Creating hdf5s.")
    sr = 16000
    for filename in tqdm(filenames):
        wav_path = os.path.join(wav_dir, f"{filename[:filename.rfind('.')]}.wav")
        sep_path = os.path.join(separated_dir, f"{filename[:filename.rfind('.')]}.wav")
        waveform_mix, _ = librosa.load(wav_path, sr=sr)
        waveform_sep, _ = librosa.load(sep_path, sr=sr)
        hdf5_path = os.path.join(hdf5s_dir, f"{filename[:filename.rfind('.')]}.h5")
        
        with h5py.File(hdf5_path, 'w') as hf:
            hf.attrs.create("sample_rate", data=sr, dtype=np.int64)
            hf.create_dataset(name="waveform", data=waveform_mix, dtype=np.float32)
            hf.create_dataset(name="waveform_separated", data=waveform_sep, dtype=np.float32)

    # separate_single_audio(params[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/n/work1/deng/data/jamendolyrics",
        help="Directory of the jamendo dataset.",
    )

    parser.add_argument(
        "--hdf5s_dir",
        type=str,
        default="/n/work1/deng/workspaces/hdf5s/jamendo",
        help="Directory of the hdf5s files.",
    )

    # Parse arguments.
    args = parser.parse_args()

    # convert data into wav files.
    main(args)
