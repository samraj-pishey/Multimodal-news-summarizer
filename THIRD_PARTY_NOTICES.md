# Third-party notices

This project uses the following data, models and libraries. Each keeps its own licence and terms.

## Data

| Item | Use | Terms |
|---|---|---|
| MM-AVS (Fu, Wang & Yang 2021): Daily Mail and CNN articles, highlights, videos | Training and evaluation; short excerpts and thumbnails of 4 test stories in the app demo | Research use, per the dataset's release; content remains the publishers' |
| GloVe 6B (Pennington et al. 2014) | Word vectors for M2SM | Public Domain Dedication and License (PDDL) |

## Pretrained models

| Model | Use | Licence |
|---|---|---|
| RoBERTa-base (`FacebookAI/roberta-base`) | Frozen sentence vectors | MIT |
| ResNet-50, ImageNet weights (torchvision) | Frame features, also exported to ONNX for the app | BSD-3-Clause (torchvision) |
| Whisper large-v3-turbo (`mlx-community/whisper-large-v3-turbo`) | Offline transcripts | MIT |
| Whisper tiny.en (`Xenova/whisper-tiny.en`) | In-browser transcription | MIT |
| BART-large-CNN (`facebook/bart-large-cnn`) | Abstractive baseline | MIT |
| DistilBART-CNN-6-6 (`Xenova/distilbart-cnn-6-6`) | In-browser abstractive summary | Apache-2.0 |

## Bundled in `webapp/vendor/` and fonts

| Library | Licence |
|---|---|
| ONNX Runtime Web | MIT |
| Transformers.js (Hugging Face) | Apache-2.0 |
| PDF.js (Mozilla) | Apache-2.0 |
| DM Sans, Playfair Display, IBM Plex Mono (in `webapp/fonts/`, `presentation/fonts/`) | SIL Open Font License 1.1 |
| Cantarell (in `presentation-rvu/fonts/`) | SIL Open Font License 1.1 |

The RV University logo and slide background in `presentation-rvu/assets/` belong to RV University and are used only for this course presentation.

## Methods

The M2SM and A2Summ architectures follow Fu et al. (2020, arXiv:2009.08018) and He et al. (CVPR 2023). The code in `src/` is this team's own implementation. VASNet (Fajtl et al. 2018), VSUMM (de Avila et al. 2011) and DR-DSN (Zhou et al. 2018) are re-implemented as baselines.
