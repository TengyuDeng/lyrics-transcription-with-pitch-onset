# End-to-End Lyrics Transcription Informed by Pitch and Onset Estimation

This is the working python scripts for paper:

T. Deng, E. Nakamura, and K. Yoshii, *"End-to-End Lyrics Transcription Informed by Pitch and Onset Estimation,"* in *Proc. of the 23rd International Society for Music Information Retrieval Conference,* Bengaluru, India, 2022.

Before running the codes, you need to obtain [DALI dataset](https://github.com/gabolsgabs/DALI) and download the audio files from YouTube. **This work may take time.**
Put the downloaded audio files in ./data/DALI/audios and run the scripts in ./dataprocess/DALI with the order `convert_to_wav.py` -> `separete_audios.py` -> `resample_audios.py` -> `create_hdf5s.py` -> `create_indexes.py`.

For the language model, please obtain the [DeepSpeech reposity](https://github.com/mozilla/DeepSpeech) and [LibriSpeech corpus](https://www.openslr.org/12).
Generate `librispeech.scorer` with steps like:
```
python3 generate_lm.py --input_txt librispeech-lm-norm.txt --output_dir . --top_k 200000 --kenlm_bins ./kenlm/build/bin/ --arpa_order 5 --max_arpa_memory "85%" --arpa_prune "0|0|1" --binary_a_bits 255 --binary_q_bits 8 --binary_type trie


./generate_scorer_package --alphabet ./chars.txt --lm ../lm.binary --vocab ./vocab-200000.txt 
  --package ../librispeech.scorer --default_alpha 0.931289039105002 --default_beta 1.1834137581510284

```

To train the model, run `train.py` with proper configs and mode.

<blockquote>
mode 1: pitch and onset estimation only<br>
mode 2: zero dummy pitch and onset + lyrics transcription<br>
mode 3: oracle pitch and onset + lyrics transcription<br>
mode 4: pitch and onset estimation + lyrics transcription
</blockquote>

 Here are some examples. Remember to train the model on LibriSpeech before training on DALI.
```
python3 train.py --mode=2 --config_yaml="./configs/LibriSpeech.yaml"

python3 train.py --mode=1 --config_yaml="./configs/DALI_pitch_onset_only.yaml"
python3 train.py --mode=2 --config_yaml="./configs/DALI_with_zero_pitch_onset.yaml"
python3 train.py --mode=3 --config_yaml="./configs/DALI_with_pitch_onset.yaml"
python3 train.py --mode=4 --config_yaml="./configs/DALI_multi_1_1_1.yaml"
```

Note that the data filtering result described in Section 4.1.1 in our paper can be find in `/resoures/offsets_and_results.csv`.
