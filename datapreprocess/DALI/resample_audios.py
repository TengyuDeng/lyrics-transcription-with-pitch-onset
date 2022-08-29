import argparse
import os, shutil, re
import time
from concurrent.futures import ProcessPoolExecutor

import librosa
import soundfile as sf


def resample_audios(args):
    """
    Resample separated audios to resample_rate.

    Args:
        dataset_dir: str
        sample_rate = 16000
    """

    dataset_dir = args.dataset_dir
    resample_rate = args.resample_rate

    # Paths
    wav_dir = os.path.join(dataset_dir, "wav")
    completed_dir = os.path.join(wav_dir, "completed")
    resampled_dir = os.path.join(wav_dir, f"sr={resample_rate}")
    os.makedirs(completed_dir, exist_ok=True)
    os.makedirs(resampled_dir, exist_ok=True)

    params = []

    filenames = sorted(os.listdir(wav_dir))

    for filename in filenames:
        if re.match(r".*\.wav", filename):
            audio_path = os.path.join(wav_dir, filename)
            completed_path = os.path.join(completed_dir, filename)
            resampled_path = os.path.join(resampled_dir, f"{filename[:filename.rfind('.')]}.wav")

            param = (
                audio_path,
                completed_path,
                resampled_path,
                resample_rate,
            )
            params.append(param)

    start_time = time.time()

    with ProcessPoolExecutor() as pool:
        pool.map(resample_single_audio, params)

    print("Time used: {:.3f} s".format(time.time() - start_time))
    
    for filename in sorted(os.listdir(completed_dir)):
        audio_path = os.path.join(wav_dir, filename)
        completed_path = os.path.join(completed_dir, filename)
        shutil.move(completed_path, audio_path)


def resample_single_audio(param):
    (
        audio_path,
        completed_path,
        resampled_path,
        resample_rate,
    ) = param

    # Load data files.
    waveform, sr = librosa.load(audio_path, sr=resample_rate)
    sf.write(resampled_path, waveform, sr)

    print(f"{audio_path} is resampled to sr={resample_rate} and saved as {resampled_path}.")
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
    resample_audios(args)
