# Getting the data

The dataset is **not included** in this repository. Download it from the official source and follow its terms of use.

## MM-AVS

- Source: <https://github.com/xiyan-fu/MM-AVS>, which links to the authors' download.
- Paper: Fu, Wang & Yang, *Multi-modal Summarization for Video-containing Documents*, NAACL 2021.
- Size: about 27 GB, mostly video.

Unpack it so that the folders look like this:

```text
data/
├── data/
│   ├── daily mail/
│   │   └── 4000/  article.txt  highlight.txt  title.txt  transcript.txt  audio.mp3  video.mp4  0.png ...
│   ├── cnn_data/
│   │   └── 1/ ...
│   └── test_id_dm.txt          official Daily Mail test list
└── glove/
    └── glove.6B.100d.txt       https://nlp.stanford.edu/projects/glove/
```

`data/` is ignored by git. Everything the scripts build from it is written to `data/` as well: the text cache, the ResNet-50 features and the RoBERTa vectors.

## What is in the repository instead

- `outputs/splits.csv`: each story's split, with its near-duplicate group.
- `outputs/audio/whisper_transcripts.jsonl`: Whisper large-v3-turbo transcripts of the Daily Mail audio, with segment timestamps and confidence. Regenerate them with `src/transcribe_audio.py`.
- `webapp/samples/` and `webapp/explainer/`: four test stories used as app demos.
