import librosa
import h5py, pickle
from tqdm import tqdm
import pandas as pd

from utils import *
from datasets import get_dataloaders
from decoders import get_CTC_decoder, Mycodec
from models import get_model
from jiwer import wer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}.")

def transform(waveform, sr):
    try:
        mel_features = librosa.power_to_db(librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=512, hop_length=256, n_mels=80))
    except Exception as e:
        print(f"Error for transforming to melspectrogram!")
        print(e)
        return [0]
    return mel_features

# Load the model, change the directories here to use your own models

configs_zero = read_yaml("./configs/DALI_with_zero_pitch_onset_dropout0.2.yaml")
epoch=25
model_zero = {}
for model_name in configs_zero["model"]:
    model_zero[model_name] = get_model(num_classes_lyrics=34, **configs_zero["model"][model_name])

model_zero['lyrics'].load_state_dict(torch.load(f"./workspace/checkpoints/DALI/20220827/config=DALI_with_zero_pitch_onset_dropout0.2.yaml/lyrics/epoch={epoch}.pth"))
model_zero['lyrics'] = model_zero['lyrics'].to(DEVICE).eval()

configs_multi = read_yaml("./configs/DALI_multi_1_1_1_dropout0.2.yaml")
epoch=25
model_multi = {}
for model_name in configs_multi["model"]:
    model_multi[model_name] = get_model(num_classes_pitch=129, num_classes_lyrics=34, **configs_multi["model"][model_name])

model_multi['melody'].load_state_dict(torch.load(f"./workspace/checkpoints/DALI-multitask/20220827/config=DALI_multi_1_1_1_dropout0.2.yaml/melody/epoch={epoch}.pth"))
model_multi['lyrics'].load_state_dict(torch.load(f"./workspace/checkpoints/DALI-multitask/20220827/config=DALI_multi_1_1_1_dropout0.2.yaml/lyrics/epoch={epoch}.pth"))
for model_name in configs_multi["model"]:
    model_multi[model_name] = model_multi[model_name].to(DEVICE).eval()
codec = Mycodec(target_type="word")
decoder = get_CTC_decoder(blank_id=0, **configs_multi["decoder"])


hdf5s_dir = "./workspace/hdf5s/jamendo"
filenames = sorted(os.listdir(hdf5s_dir))
df = pd.DataFrame(columns=["track_name", "multitask_result", "zero_result"])


for filename in tqdm(filenames):
    trackname = filename[:filename.rfind('.')]
    hdf5_path = os.path.join(hdf5s_dir, filename)
    with h5py.File(hdf5_path, 'r') as hf:
        sr = hf.attrs['sample_rate']

        waveform_mix = hf['waveform'][:]
        waveform_sep = hf['waveform_separated'][:]
    
    length = min(len(waveform_mix), len(waveform_sep))
    input_features_sep = torch.tensor(transform(waveform_sep[:length], sr), dtype=torch.float)
    input_features_mix = torch.tensor(transform(waveform_mix[:length], sr), dtype=torch.float)
    input_features = torch.stack([input_features_sep, input_features_mix]).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output_melody = model_multi['melody'](input_features)
        pitches_prob = torch.softmax(output_melody[:, :-1, :], dim=1)
        onsets_prob = torch.sigmoid(output_melody[:, -1, :])
        zero_pitches = torch.zeros(pitches_prob.shape).to(DEVICE)
#         zero_pitches[:, -1, :] = 1
        zero_onsets = torch.zeros(onsets_prob.shape).to(DEVICE)
    
        output_lyrics = model_multi['lyrics'](input_features, pitches_prob, onsets_prob)
        lyrics_prob = torch.softmax(output_lyrics, dim=-1).transpose(0, 1)
    
        output_lyrics_zero = model_zero['lyrics'](input_features, zero_pitches, zero_onsets)
        lyrics_prob_zero = torch.softmax(output_lyrics_zero, dim=-1).transpose(0, 1)
    
    # Melody predict
    pitch_prob = pitches_prob[0].cpu().numpy()
    onset_prob = onsets_prob[0].cpu().numpy()
    pitch_pre = np.argmax(pitch_prob, axis=0)
    onset_pre = peakpicking(onset_prob)

    output_lengths = [lyrics_prob.shape[-2]]
    # Lyrics predict

    text_pre = decoder.decode(lyrics_prob.cpu().numpy(), output_lengths)[0]
    text_pre_zero = decoder.decode(lyrics_prob_zero.cpu().numpy(), output_lengths)[0]
    
    with open(f"/n/work1/deng/data/jamendolyrics/lyrics/{trackname}.raw.txt", 'r') as f:
        text_gt = f.read()
    text_gt = text_gt.replace("\n", " ")
    
    df = df.append({"track_name": trackname, "multitask_result": wer(text_gt, text_pre), "zero_result": wer(text_gt, text_pre_zero)}, ignore_index=True)


df.to_csv("./test/jamendo_results.csv")
print(f"WER with multitask model: {np.mean(df['multitask_result'])}\nWER with zero dummy model: {np.mean(df['zero_result'])}")