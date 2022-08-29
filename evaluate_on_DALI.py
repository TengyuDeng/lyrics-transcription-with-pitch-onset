#!/usr/bin/env python
# coding: utf-8

# In[11]:


import librosa.display
import librosa
import soundfile
import h5py, pickle
from tqdm import tqdm

from utils import *
from datasets import get_dataloaders
from decoders import get_CTC_decoder, Mycodec
from models import get_model
from jiwer import wer

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
from IPython.display import Audio

# DEVICE = "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def predict(data, model, decoder, down_sample, device, onset_type="pre", pitch_type="pre"):
    audio_features, pitches, onsets, targets, input_lengths, target_lengths = data
    output_lengths = downsample_length(input_lengths, down_sample).numpy()
    audio_features = audio_features.to(device)

    if onset_type == "pre" or pitch_type == "pre":
        with torch.no_grad():
            output_melody = model['melody'](audio_features)
            pitches_prob = torch.softmax(output_melody[:,:-1,:], dim=1)
            onsets_prob = torch.sigmoid(output_melody[:,-1,:])
    
    if onset_type == "pre":
        onset_input = onsets_prob
    elif onset_type == "zero":
        onset_input = torch.zeros(onsets.shape).to(device)
    else:
        onset_input = onsets.to(device)
    
    if pitch_type == "pre":
        pitch_input = pitches_prob
    elif pitch_type == "zero":
        pitch_input = torch.zeros(pitches.shape).to(device)
#     elif pitch_type == "zero_01":
#         pitch_input = torch.zeros(pitches.shape).to(device)
#         pitch_input[:,-1,:] = 1
    else:
        pitch_input = pitches.to(device)
        

    with torch.no_grad():
        output_lyrics = model['lyrics'](audio_features, pitch_input, onset_input)
        lyrics_prob = torch.softmax(output_lyrics, dim=-1).transpose(0, 1).cpu().numpy()
    
    return decoder.decode(lyrics_prob, output_lengths)


# Load the database

configs = read_yaml("./configs/DALI_with_pitch_onset_dropout0.2.yaml")
train_dataloaders, val_dataloaders, test_dataloaders = get_dataloaders(configs["dataloaders"])

codec = Mycodec(target_type="word")
decoder = get_CTC_decoder(blank_id=0, **configs["decoder"])


# Load the model, change the directories here to use your own models

# zero
epoch=25
configs_zero = read_yaml("./configs/DALI_with_zero_pitch_onset_dropout0.2.yaml")
model_zero = {}
for model_name in configs_zero["model"]:
    model_zero[model_name] = get_model(num_classes_pitch=129, num_classes_lyrics=len(codec.characs), **configs_zero["model"][model_name])

model_zero['lyrics'].load_state_dict(torch.load(f"./workspace/checkpoints/DALI/20220827/config=DALI_with_zero_pitch_onset_dropout0.2.yaml/lyrics/epoch={epoch}.pth"))
model_zero['lyrics'] = model_zero['lyrics'].to(DEVICE).eval()


# oracle
epoch=25
configs_oracle = read_yaml("./configs/DALI_with_pitch_onset_dropout0.2.yaml")
model_oracle = {}
for model_name in configs_oracle["model"]:
    model_oracle[model_name] = get_model(num_classes_pitch=129, num_classes_lyrics=len(codec.characs), **configs_oracle["model"][model_name])

model_oracle['lyrics'].load_state_dict(torch.load(f"./workspace/checkpoints/DALI/20220827/config=DALI_with_pitch_onset_dropout0.2.yaml/lyrics/epoch={epoch}.pth"))
model_oracle['lyrics'] = model_oracle['lyrics'].to(DEVICE).eval()


# multitask

epoch=25
configs_multi = read_yaml("./configs/DALI_multi_1_1_1_dropout0.2.yaml")
model_multi = {}
for model_name in configs_multi["model"]:
    model_multi[model_name] = get_model(num_classes_pitch=129, num_classes_lyrics=34, **configs_multi["model"][model_name])

model_multi['melody'].load_state_dict(torch.load(f"./workspace/checkpoints/DALI-multitask/20220827/config=DALI_multi_1_1_1_dropout0.2.yaml/melody/epoch={epoch}.pth"))
model_multi['lyrics'].load_state_dict(torch.load(f"./workspace/checkpoints/DALI-multitask/20220827/config=DALI_multi_1_1_1_dropout0.2.yaml/lyrics/epoch={epoch}.pth"))
for model_name in configs_multi["model"]:
    model_multi[model_name] = model_multi[model_name].to(DEVICE).eval()


results = []
down_sample = model_oracle['lyrics'].down_sample

# for data in tqdm(val_dataloaders["DALI_val"]):
for data in tqdm(test_dataloaders["DALI_test"]):

    predict_zero = predict(data, model_zero, 
                           decoder=decoder, down_sample=down_sample, device=DEVICE,
                           onset_type="zero", pitch_type="zero",
                          )
    predict_oracle = predict(data, model_oracle, 
                           decoder=decoder, down_sample=down_sample, device=DEVICE,
                           onset_type="gt", pitch_type="gt",
                          )
    
    predict_multi = predict(data, model_multi, 
                           decoder=decoder, down_sample=down_sample, device=DEVICE,
                           onset_type="pre", pitch_type="pre",
                          )
    _, _, _, targets, _, target_lengths = data
    texts = []
    targets = targets.numpy()
    target_lengths = target_lengths.numpy()
    start = 0
    batch_size = len(target_lengths)
    for i in range(batch_size):
        text = codec.decode(targets[start: start + target_lengths[i]])
        texts.append(text)
        start += target_lengths[i]
    
    for i in range(batch_size):
        try:
            results.append({
                "GT": texts[i],
                "zero": predict_zero[i],
                "zero WER": wer(texts[i], predict_zero[i]),
                "oracle": predict_oracle[i],
                "oracle WER": wer(texts[i], predict_oracle[i]),
                "multitask": predict_multi[i],
                "multitask WER": wer(texts[i], predict_multi[i]),
#                 "multitask new": predict_multi_new[i],
#                 "multitask new WER": wer(texts[i], predict_multi_new[i]),
            })
        except ValueError:
            print("empty text!")
#     input(results[-1])
    
WERs = [[result['zero WER'], result['oracle WER'], result['multitask WER']] for result in results]

res = np.mean(WERs, axis=0)
print(f"WER with zero dummy model: {res[0]}\nWER with oracle model: {res[1]}\nWER with multitask model: {res[2]}")

