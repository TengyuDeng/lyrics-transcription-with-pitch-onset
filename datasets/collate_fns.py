
import torch, numpy as np

def get_collate_fn(collate_fn_type="frame_based"):
    if collate_fn_type == "text_only":
        return collate_fn_text_only
    elif collate_fn_type == "frame_based":
        return collate_fn_frame_based
    elif collate_fn_type == "tatum_based":
        return collate_fn_tatum_based

def collate_fn_text_only(list_data):
    # Input:
    #     list_data: [
    #         (feature1, target1),
    #         (feature2, target2),
    #     ]
    # Output:
    #     (audio_features, targets, input_lengths, target_lengths)
    #     audio_features: (batch_size, feature_num, time)
    #     targets: (batch_size, max_length)
    
    # start_time = time.time()

    raw_features, raw_texts = tuple(zip(*list_data))

    # features
    new_features = []
    new_texts = []
    input_lengths = []
    text_lengths = []

    for n in range(len(list_data)):
        feature = raw_features[n]
        text = raw_texts[n]
        # feature (channel, time)
        if feature.ndim > 1:
            new_features.append(feature.transpose(0, -1))
            input_lengths.append(feature.shape[-1])
            new_texts.append(text)
            text_lengths.append(len(text))
    
    audio_features = torch.nn.utils.rnn.pad_sequence(new_features, batch_first=True).transpose(1, -1)
    texts = torch.cat(new_texts)
    input_lengths = torch.tensor(input_lengths, dtype=torch.int)
    text_lengths = torch.tensor(text_lengths, dtype=torch.int)

    return (audio_features, texts, input_lengths, text_lengths)


def collate_fn_frame_based(list_data):
    # Input:
    #     list_data: [
    #         (feature1, target1),
    #         (feature2, target2),
    #     ]
    # Output:
    #     (audio_features, targets, input_lengths, target_lengths)
    #     audio_features: (batch_size, feature_num, time)
    #     targets: (\sum{lengths})
    
    # start_time = time.time()

    raw_features, raw_pitches, raw_onsets, raw_texts = tuple(zip(*list_data))
    
    # features
    new_features = []
    new_pitches = []
    new_onsets = []
    new_texts = []
    input_lengths = []
    text_lengths = []
#     new_tatums = []

    for n in range(len(list_data)):
        feature = raw_features[n]
        pitch = raw_pitches[n]
        if pitch.ndim == 2:
            pitch = pitch.transpose(0, -1)
        onset = raw_onsets[n]
        text = raw_texts[n]
#         tatum = raw_tatums[n]
        # feature (channel, time)
        if feature.ndim > 1:
            new_features.append(feature.transpose(0, -1))
            input_lengths.append(len(pitch))
            new_pitches.append(pitch)
            new_onsets.append(onset)
            new_texts.append(text)
            text_lengths.append(len(text))
#             new_tatums.append(tatum)
    
    audio_features = torch.nn.utils.rnn.pad_sequence(new_features, batch_first=True).transpose(1, -1)
    # pitches = torch.cat(new_pitches)
    if new_pitches[0].ndim == 1:
        pitches = torch.nn.utils.rnn.pad_sequence(new_pitches, batch_first=True, padding_value=128)
    else:
        pitches = torch.nn.utils.rnn.pad_sequence(new_pitches, batch_first=True, padding_value=0).transpose(1, -1)
    onsets = torch.nn.utils.rnn.pad_sequence(new_onsets, batch_first=True, padding_value=0.)
    texts = torch.cat(new_texts)
    input_lengths = torch.tensor(input_lengths, dtype=torch.int)
    text_lengths = torch.tensor(text_lengths, dtype=torch.int)
#     tatum_frames = torch.stack(new_tatums)

    return (audio_features, pitches, onsets, texts, input_lengths, text_lengths)