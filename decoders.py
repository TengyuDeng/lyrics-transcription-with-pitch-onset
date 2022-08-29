import numpy as np
import torch
import re
from utils import *
from ds_ctcdecoder import Alphabet, Scorer, ctc_beam_search_decoder
from g2p_en import G2p

def get_CTC_decoder(blank_id, **configs):
    decoder_type = configs["name"]
    if decoder_type == "greedy":
        return greedy_CTC_decoder(blank_id, **configs)
    elif decoder_type == "ds_beam":
        return ds_beam_decoder(blank_id, **configs)
    else:
        raise ValueError("No such decoder!")

def annex(string):
    new_string = []
    i = 0
    while True:
        new_string.append(string[i])
        if i >= len(string) - 1:
            break
            
        j = i
        while string[j+1] == string[j]:
            j += 1
        i = j
        i += 1
    return new_string


class Mycodec:

    def __init__(self, target_type="word", min_pitch=0, max_pitch=128):
        self.target_type = target_type
        if target_type == "phoneme":
            self.g2p = G2p()
            self.characs =['|', ' '] + self.g2p.phonemes[4:] + [',', '.', '?', '!', '-', "'"]
        elif target_type == "with_pitch":
            self.characs = ['|'] + [(charac, pitch) 
            for charac in " abcdefghijklmnopqrstuvwxyz,.?!-'"
            for pitch in (list(range(min_pitch, max_pitch)) + [128])]
            self.min_pitch = min_pitch
            self.max_pitch = max_pitch
        else:
            self.characs = "| abcdefghijklmnopqrstuvwxyz,.?!-'"
        order_nums = list(range(len(self.characs)))
        self.codebook = dict(zip(self.characs, order_nums))

    def encode(self, s):
        codelist = []

        for char in s:
            try:
                codelist.append(self.codebook[char])
            except Exception as e:
                codelist.append(1)

        return codelist

    def decode(self, codes):
        if self.target_type == "word":
            string = ""
            for code in codes:
                string += self.characs[code]
        else:
            string = []
            for code in codes:
                string.append(self.characs[code])
        
        if not string:
            string += '.'
        return string

    def decode_batch(self, codes_batch, input_lengths):
        texts = []
        batch_size = len(input_lengths)
        if codes_batch.ndim == 1:
            start = 0
            for i in range(batch_size):
                text = self.decode(codes_batch[start: start + input_lengths[i]])
                texts.append(text)
                start += input_lengths[i]
        else:
            for i in range(batch_size):
                text = self.decode(codes_batch[i][: input_lengths[i]])
        
                texts.append(text)
        return texts

    def phonemize(self, text):
        return self.g2p(text)

class greedy_CTC_decoder:
    
    def __init__(self, blank_id, target_type="word", **args):
        
        self.target_type = target_type
        self.codec = Mycodec(target_type=target_type)
        self.blank_label = self.codec.characs[blank_id]
    
    def decode(self, logits, input_lengths):
        
        # logits: (batch_size, length, num_class)

        codes = torch.argmax(logits, dim=-1)
        predicts = []
        for n, segment in enumerate(codes):
            predict = self.codec.decode(segment[:input_lengths[n]])
            if self.target_type == "phoneme":
                predict = annex(predict)
                predict = predict.remove(self.blank_label)
            else:
                predict = re.sub(r'(.)\1+', r'\1', predict)
                predict = predict.replace(self.blank_label, '')
            predicts.append(predict)

        return predicts


class ds_beam_decoder:

    def __init__(self, blank_id, target_type="word", scorer_path=None, **args):

        if target_type == "phoneme":
            self.alphabet = Alphabet("/n/work1/deng/tools/lm/chars_phoneme.txt")
        else:
            self.alphabet = Alphabet("/n/work1/deng/tools/lm/chars.txt")
        if scorer_path is not None:
            self.scorer = Scorer(alphabet=self.alphabet, scorer_path=scorer_path, alpha=0.75, beta=1.85)
        else:
            self.scorer = None

    def decode(self, softmax_out, input_lengths):
        
        softmax_out = np.append(softmax_out[:, :, 1:], softmax_out[:, :, (0,)], axis=-1)
        predicts = []
        # softmax_out: (batch_size, length, num_class)
        
        for n in range(softmax_out.shape[0]):
            predict = ctc_beam_search_decoder(probs_seq=softmax_out[n, :input_lengths[n], :],
                                              alphabet=self.alphabet,
                                              scorer=self.scorer,
                                              beam_size=25,
                                             )[0][1]
            if len(predict) == 0: 
                predict = " "
            predicts.append(predict)

        return predicts
