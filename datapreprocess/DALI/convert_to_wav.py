import argparse
import os, shutil
import time
from concurrent.futures import ProcessPoolExecutor

import librosa
import soundfile as sf
import numpy as np


def convert_to_wav(args):
    """
    Convert mp3 files to wav in order to speed up loading.
    Resample audio files first.

    Args:
        dataset_dir: str
        sample_rate
    """

    dataset_dir = args.dataset_dir
    resample_rate = args.resample_rate

    # Paths
    audios_dir = os.path.join(dataset_dir, "audios")
    completed_dir = os.path.join(audios_dir, "completed")
    wav_dir = os.path.join(dataset_dir, "wav", f"sr={resample_rate}")
    os.makedirs(completed_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)

    params = []

    filenames = sorted(os.listdir(audios_dir))
    filenames.remove("completed")

    for filename in filenames:
        
        audio_path = os.path.join(audios_dir, filename)
        completed_path = os.path.join(completed_dir, filename)
        wav_path = os.path.join(wav_dir, f"{filename[:filename.rfind('.')]}.wav")

        param = (
            audio_path,
            completed_path,
            wav_path,
            resample_rate,
        )
        params.append(param)

    start_time = time.time()

    with ProcessPoolExecutor() as pool:
        pool.map(convert_single_audio_to_wav, params)

    print("Time used: {:.3f} s".format(time.time() - start_time))
    
    for filename in sorted(os.listdir(completed_dir)):
        audio_path = os.path.join(audios_dir, filename)
        completed_path = os.path.join(completed_dir, filename)
        shutil.move(completed_path, audio_path)


def convert_single_audio_to_wav(param):
    (
        audio_path,
        completed_path,
        wav_path,
        resample_rate,
    ) = param

    # Load data files.
    waveform, sr = librosa.load(audio_path, sr=resample_rate, res_type="kaiser_fast")
    sf.write(wav_path, waveform, sr)
    print(f"{audio_path} is saved as {wav_path}.")
    shutil.move(audio_path, completed_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/n/work1/deng/data/DALI",
        help="Directory of the DALI dataset.",
    )
    parser.add_argument("--resample_rate", type=int, default=16000, help="Resample rate.")

    # Parse arguments.
    args = parser.parse_args()

    # convert data into wav files.
    convert_to_wav(args)
