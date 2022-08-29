from .CRNN_melody_frame import CRNN_melody_frame
from .CRNN_lyrics import CRNN_lyrics
from .CRNN_lyrics_onset import CRNN_lyrics_onset
from .CRNN_lyrics_pitch_onset import CRNN_lyrics_pitch_onset

model_list = {
    "CRNN_melody_frame": CRNN_melody_frame,
    "CRNN_lyrics": CRNN_lyrics,
    "CRNN_lyrics_onset": CRNN_lyrics_onset,
    "CRNN_lyrics_pitch_onset": CRNN_lyrics_pitch_onset,
}
def get_model(num_classes_pitch=129, num_classes_lyrics=34, **configs):
    
    model_type = configs["name"]

    harmonics_shift = False
    input_channels = 6 if harmonics_shift else 1
    Model = model_list[model_type]
    
    model = Model(
        num_classes_pitch=num_classes_pitch, 
        num_classes_lyrics=num_classes_lyrics,
        **configs,
        )

    return model