import argparse
import torch
import os, pickle, re

from datasets import get_dataloaders
from models import get_model
from utils import *
from train_utils import *
from jiwer import wer, cer
from decoders import get_CTC_decoder, Mycodec

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}.")

@train_loop_decorator
def train_loop(data, model, loss_functions, optimizers, schedulers, loss_weights, batch_size_a, mode):
    for optimizer in optimizers:
        optimizer.zero_grad()
    # torch.cuda.empty_cache()
    audio_features, pitches, onsets, targets, input_lengths, target_lengths = data
    audio_features = audio_features.to(DEVICE)
    pitches = pitches.to(DEVICE)
    onsets = onsets.to(DEVICE)
    
    if mode == 2:
        pitch_features = torch.zeros_like(pitches).to(DEVICE)
        onset_features = torch.zeros_like(onsets).to(DEVICE)
    elif mode == 3:
        pitch_features = pitches.to(DEVICE)
        onset_features = onsets.to(DEVICE)
    else:
        # Forward melody
        output_melody = model['melody'](audio_features[:batch_size_a, ...])
        # output_melody: (batch_size, 129 + 1, length)
        loss_pitch = loss_functions['pitch'](output_melody[:, :-1, :], pitches[:batch_size_a, ...])
        loss_onset = loss_functions['onset'](output_melody[:, -1, :], onsets[:batch_size_a, ...])
        # Concat outputs of melody part and random LibriSpeech melody
        pitches_dali = torch.softmax(output_melody[:, :-1, :], dim=1)
        onsets_dali = torch.sigmoid(output_melody[:, -1, :])
        pitches_libri = pitches[batch_size_a:, ...]
        onsets_libri = onsets[batch_size_a:, ...]
        pitch_features = torch.cat([pitches_dali, pitches_libri], dim=0)
        onset_features = torch.cat([onsets_dali, onsets_libri], dim=0)
    # print(pitch_features)
    # print(onset_features)

    if mode != 1:
        # Forward lyrics
        lyrics_logits = model['lyrics'](audio_features, pitch_features, onset_features)
        # lyrics_logits: (length, batch_size, num_classes)
        lyrics_logprobs = torch.nn.functional.log_softmax(lyrics_logits, dim=-1)
        output_lengths = downsample_length(input_lengths, model['lyrics'].down_sample)
        # lyrics_logprobs: (max_length, batch_size, num_classes)
        # targets: (sum(target_lengths),)
        # output_lengths: (batch_size,)
        # target_lengths: (batch_size,)
        loss_lyrics = loss_functions['lyrics'](lyrics_logprobs, targets, output_lengths, target_lengths)
    
    if mode == 1:
        loss = loss_weights['pitch'] * loss_pitch + loss_weights['onset'] * loss_onset
    elif mode == 4:
        loss = loss_weights['lyrics'] * loss_lyrics + loss_weights['pitch'] * loss_pitch + loss_weights['onset'] * loss_onset
    else:
        loss = loss_lyrics

    # Backprop
    loss.backward()
    for optimizer in optimizers:
        optimizer.step()
    for scheduler in schedulers:
        scheduler.step()
    
    return loss.item()

@test_loop_decorator
def test_loop(data, model, dataloader_name, decoder, codec, mode):
    audio_features, pitches, onsets, targets, input_lengths, target_lengths = data
    audio_features = audio_features.to(DEVICE)
    with torch.no_grad():
        if mode == 2:
            pitch_features = torch.zeros_like(pitches).to(DEVICE)
            onset_features = torch.zeros_like(onsets).to(DEVICE)
        elif mode == 3:
            pitch_features = pitches.to(DEVICE)
            onset_features = onsets.to(DEVICE)
        else:
            output_melody = model['melody'](audio_features)
            pitch_features = torch.softmax(output_melody[:, :-1, :], dim=1)
            onset_features = torch.sigmoid(output_melody[:, -1, :])
        if mode != 1:
            lyrics_logits = model['lyrics'](audio_features, pitch_features, onset_features)
            lyrics_prob = torch.softmax(lyrics_logits, dim=-1).transpose(0, 1)
            output_lengths = downsample_length(input_lengths, model['lyrics'].down_sample).numpy()
    # print(pitch_features)
    # print(onset_features)
    # Melody evaluation
    pitches = np.argmax(pitches.numpy(), axis=1)
    onsets = peakpicking(onsets.numpy(), window_size=2, threshold=0.5)
    # onsets = onsets.numpy()
    pitches_pre = np.argmax(pitch_features.cpu().numpy(), axis=1)
    onsets_pre = peakpicking(onset_features.cpu().numpy(), window_size=2, threshold=0.5)
    # onsets_pre = (onsets_prob.cpu().numpy() > 0.5).astype(int)
    note_results = evaluate_notes(pitches, onsets, pitches_pre, onsets_pre, input_lengths, sr=16000, hop_length=256)
    frame_err = evaluate_frames_batch(pitches, pitches_pre, input_lengths)
    if mode == 1:
        return note_results[2], note_results[5], note_results[8], frame_err
    else:
        # Lyrics evaluation
        texts = codec.decode_batch(targets.numpy(), target_lengths.numpy())
        texts_pre = decoder.decode(lyrics_prob.cpu().numpy(), output_lengths)
        try:
            error_wer = wer(texts, texts_pre)
            error_cer = cer(texts, texts_pre)
        except ValueError:
            texts = [text + "." for text in texts]
            error_wer = wer(texts, texts_pre)
            error_cer = cer(texts, texts_pre)

        return note_results[2], note_results[5], note_results[8], frame_err, error_wer, error_cer

def main(args):
    mode_descriptions = {
     1: "pitch and onset estimation only",
     2: "zero dummy pitch and onset + lyrics transcription",
     3: "oracle pitch and onset + lyrics transcription",
     4: "pitch and onset estimation + lyrics transcription",
    }

    workspace = args.workspace
    mode = args.mode
    if mode not in [1, 2, 3, 4]:
        raise ValueError("mode must be in [1,2,3,4]")
    print(f"Running on mode {mode}: {mode_descriptions[mode]}")
    config_yaml = args.config_yaml
    # Read config.yaml
    configs = read_yaml(config_yaml)

    # Get directories
    checkpoints_dir, statistics_dir = get_dirs(workspace, configs['task'], config_yaml)

    # Get codec
    codec = Mycodec(target_type="word")

    # Construct dataloaders
    train_dataloaders, val_dataloaders, test_dataloaders = get_dataloaders(configs["dataloaders"])
    batch_size_a = train_dataloaders["mix"].batch_sampler.batch_size_a
    # Get model
    model = {}
    for model_name in configs["model"]:
        model[model_name] = get_model(num_classes_pitch=129, **configs["model"][model_name])
        model[model_name] = model[model_name].to(DEVICE)
        os.makedirs(os.path.join(checkpoints_dir, model_name), exist_ok=True)
    # if torch.cuda.DEVICE_count() > 1:
    #     print(f"Using {torch.cuda.DEVICE_count()} GPUs!")
    #     model = torch.nn.DataParallel(model)

    # Get decoder
    decoder = get_CTC_decoder(blank_id=0, **configs["decoder"])

    # Loss, optimizer, and scheduler
    training_configs = configs["training"]
    error_names = training_configs['error_names']
    if 'resume_checkpoint' in training_configs:
        for model_name in training_configs['resume_checkpoint']:
            model[model_name].load_state_dict(torch.load(training_configs['resume_checkpoint'][model_name]), strict=True)
    max_epoch = training_configs['max_epoch']
    learning_rate = float(training_configs['learning_rate'])
    warm_up_steps = training_configs['warm_up_steps']
    es_monitor = training_configs['early_stop_monitor']
    es_patience = training_configs['early_stop_patience']
    es_mode = training_configs['early_stop_mode']
    es_index = training_configs['early_stop_index']
    loss_weights = training_configs['loss_weights']
    reduce_lr_steps = training_configs['reduce_lr_steps'] if 'reduce_lr_steps' in training_configs else None
    epoch_0 = training_configs['continue_epoch'] if 'continue_epoch' in training_configs else 0
    # global half_window_length
    # global conv_weights
    # half_window_length = training_configs["half_window_length"]
    # conv_weights = get_conv_weight(half_length=half_window_length).to(DEVICE)
    
    loss_functions = {
    'pitch': CrossEntropyLossWithProb().to(DEVICE),
    'onset': torch.nn.BCEWithLogitsLoss().to(DEVICE),
    'lyrics': torch.nn.CTCLoss(blank=0, zero_infinity=True).to(DEVICE),
    }
    loss_functions = set_weights(loss_functions, training_configs, device=DEVICE)

    optimizers = [
    torch.optim.Adam(
        filter(lambda param : param.requires_grad, model[model_name].parameters()),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=True,
        ) for model_name in model
    ]
    
    lr_lambda = lambda step : get_lr_lambda(step, warm_up_steps=warm_up_steps, reduce_lr_steps=reduce_lr_steps)
    schedulers = [
    torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    for optimizer in optimizers
    ]
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)
    early_stop = EarlyStopping(mode=es_mode, patience=es_patience)
    
    stop_flag = False
    for epoch in range(epoch_0, epoch_0 + max_epoch + 1):
        print(f"Epoch {epoch+1}\n-------------------------------")
        for dataloader_name in val_dataloaders:
            statistics = test_loop(
                dataloader=val_dataloaders[dataloader_name],
                model=model,
                dataloader_name=dataloader_name,
                error_names=error_names,
                statistics_dir=statistics_dir,
                epoch=epoch,
                val=True,
                decoder=decoder,
                codec=codec,
                mode=mode,
                )
            if dataloader_name == es_monitor:
                if early_stop(statistics['errors'][es_index]):
                    best_epoch = epoch - early_stop.patience
                    stop_flag = True
        if stop_flag:
            break
        if epoch == epoch_0 + max_epoch:
            best_epoch = epoch
            break
        for dataloader_name in train_dataloaders:
            train_loop(
                dataloader=train_dataloaders[dataloader_name], 
                model=model,
                statistics_dir=statistics_dir,
                epoch=epoch + 1,
                loss_functions=loss_functions,
                optimizers=optimizers,
                schedulers=schedulers,
                loss_weights=loss_weights,
                batch_size_a=batch_size_a,
                mode=mode,
            )
        
        for model_name in model:
            save_checkpoints(model[model_name], os.path.join(checkpoints_dir, model_name), epoch + 1)

        # scheduler.step(error_wer)
    print("Testing with model on best validation error.")
    for model_name in model:
        load_checkpoints(model[model_name], os.path.join(checkpoints_dir, model_name), best_epoch)
    test_statistics = {}
    for dataloader_name in test_dataloaders:
        statistics_dict = test_loop(
            dataloader=test_dataloaders[dataloader_name],
            model=model,
            dataloader_name=dataloader_name,
            error_names=error_names,
            statistics_dir=None,
            epoch=None,
            val=True,
            decoder=decoder,
            codec=codec,
            mode=mode,
            )
        test_statistics[dataloader_name] = statistics_dict
    print(test_statistics)
    test_path = os.path.join(statistics_dir, "test_statistics.pkl")
    pickle.dump(test_statistics, open(test_path, 'wb'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, default=4, help="""
mode 1: pitch and onset estimation only
mode 2: zero dummy pitch and onset + lyrics transcription
mode 3: oracle pitch and onset + lyrics transcription
mode 4: pitch and onset estimation + lyrics transcription
"""
)
    parser.add_argument("--workspace", type=str, default="./workspace/", help="Directory of workspace.")
    parser.add_argument("--config_yaml", type=str, required=True, help="User defined config file.")
    args = parser.parse_args()
    
    main(args)
