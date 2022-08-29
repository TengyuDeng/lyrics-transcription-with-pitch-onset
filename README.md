# End-to-End Lyrics Transcription Informed by Pitch and Onset Estimation

This is the working python scripts for ISMIR 2022 paper *End-to-End Lyrics Transcription Informed by Pitch and Onset Estimation*.

Before running the codes, you need to obtain **DALI** dataset from ... and download the audio files from YouTube.
Since the number of these files is rather large and they need to be downloaded separatedly from YouTube and converted to audio files, this work may take time.

After that, run scripts in dataprocess/ directory to create formated hdf5 data files. You can also directly load data without creating the hdf5 files, in that case, a custom dataset class is needed.