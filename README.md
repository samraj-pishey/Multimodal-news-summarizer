<div align="center">

# Multimodal NLP Framework for Automated Summary Generation from Heterogeneous Sources

**Multimodal News Summarizer · RV University, School of Computer Science and Engineering**

A system that reads a news **article**, the **video** published with it and the **speech** in that video, then picks the key sentences and key frames. It is trained and evaluated on the **MM-AVS** Daily Mail and CNN dataset, and the whole thing runs in the browser.

[Live application](https://multimodal-news-summarizer-app.vercel.app) · [Presentation](https://multimodal-news-summarizer.vercel.app) · [Presentation (RVU template)](https://multimodal-news-summarizer-rvu.vercel.app) · [Source repository](https://github.com/samraj-pishey/Multimodal-news-summarizer)

</div>

---

## Project overview

Pick a sample story or upload your own article (PDF), video (MP4 or WebM) or both. The app transcribes the speech with Whisper, scores every sentence, returns the few sentences that matter and the frames worth seeing, and compares them with the journalist's highlights using ROUGE. If you ask for it, it also writes a short abstractive summary with DistilBART. All processing happens on the user's device, and files are never uploaded to a server.

The project builds and compares three approaches:

| Approach | Method |
|---|---|
| **M2SM replication** | Fu et al. 2020: BiLSTM encoders, bilinear and bi-hop attention over article, transcript and frames, late+ fusion, REINFORCE frame reward |
| **Hybrid (M2SM + A2Summ)** | Frozen RoBERTa sentence vectors, a sentence BiLSTM, a 2-layer transformer with one feed-forward expert per modality (text, video, speech), Whisper-timestamp alignment mask, focal and contrastive losses, late+ fusion |
| **Abstractive baseline** | BART-large-CNN, used without retraining, with greedy decoding chosen on validation |

> **Scope:** This is a research system for summarizing news stories. Its main summaries are **extractive**: they contain only sentences that already exist in the source, so they cannot invent facts. The optional AI summary is abstractive and is labelled as such in the app.

## At a glance

| Component | Configuration |
|---|---|
| Dataset | MM-AVS (Fu, Wang & Yang, NAACL 2021): 2,144 stories after cleaning |
| Splits | 1,359 train · 194 validation · 387 Daily Mail test · 204 CNN test |
| Text features | GloVe word vectors (M2SM) · frozen RoBERTa-base sentence vectors (hybrid) |
| Video features | ResNet-50, 1 frame per second, 2,048 numbers per frame |
| Speech | Whisper large-v3-turbo offline · Whisper tiny.en in the browser |
| Model selection | Validation only; each test set scored once |
| Significance | Paired bootstrap, 10,000 resamples |
| In-browser runtime | ONNX Runtime Web (M2SM, 8-bit ResNet-50) + transformers.js (Whisper, DistilBART) |
| Web application | Static site on Vercel |

## Contents

- [How the system works](#how-the-system-works)
- [Dataset and evaluation protocol](#dataset-and-evaluation-protocol)
- [Results](#results)
  - [Text summaries](#1-text-summaries-daily-mail-test)
  - [Hybrid ablations](#2-hybrid-ablations)
  - [Key frames](#3-key-frames)
  - [Video-only mode](#4-video-only-mode)
- [The web application](#the-web-application)
- [Repository layout](#repository-layout)
- [Run locally](#run-locally)
- [Limitations](#limitations)
- [Team](#team)
- [Dataset, citation and licence](#dataset-citation-and-licence)

---

## How the system works

```text
      ARTICLE (PDF / text)          VIDEO (MP4 / WebM)
              |                        |          |
              v                        v          v
     Sentence splitting          Frames, 1/s    Audio track
              |                        |          |
              v                        v          v
   GloVe / RoBERTa vectors       ResNet-50     Whisper speech-to-text
              |                  2,048-d        timed sentences
              |                        |          |
              +-----------+------------+----------+
                          |
                          v
          Encoder: BiLSTM (M2SM) or transformer with
          one expert per modality (hybrid), with attention
          from article sentences to speech and frames
                          |
                          v
                 Late+ fusion (β = 0.3)
                     /            \
                    v              v
          Sentence scores      Frame scores
                    |              |
                    v              v
          Top-k sentences      Key frames
          (extractive)         (VSUMM or model)
                    |
                    v
       ROUGE against the journalist's highlights
       + optional abstractive summary (BART)
```

Each article sentence is turned into numbers. Attention lets each sentence look at the speech and the video frames and take in whatever is relevant. Fusion then decides how much of that multimodal information to keep. The model gives every sentence a score, and the highest-scoring sentences form the summary. The hybrid model also uses Whisper's timestamps, so each stretch of speech attends only to the frames shown while it was being spoken.

If only an article is uploaded, the app uses a text-only model. If only a video is uploaded, Whisper's transcribed sentences take the article's place and VSUMM (k-means over frame features) picks the frames.

---

## Dataset and evaluation protocol

The project uses **MM-AVS**, a public research dataset. Each story contains the article, the journalist's highlights, the images, the video and its audio.

| Step | Result |
|---|---|
| Downloaded | 2,257 story folders (≈ 27 GB, mostly video) |
| Removed: no article text | 33 stories |
| Removed: empty or broken video | 80 stories |
| **Kept** | **2,144 stories** |
| Near-duplicate check | 8-word shingles, Jaccard > 0.3: 892 duplicate pairs found |
| Vocabulary | 19,531 words (at least 2 uses in training, with a GloVe vector) |
| Speech kept | Whisper segments with no_speech ≤ 0.6 and avg_logprob ≥ −1.0; a story keeps its speech if at least 5 words survive |

| Partition | Stories | Role |
|---|---:|---|
| Daily Mail train | 1,359 | Training |
| Daily Mail validation | 194 | Every choice: epoch, learning rate, summary length, decoding |
| Daily Mail test | 387 | Final score, computed once |
| CNN test | 204 | Extra out-of-domain check (CNN is test-only, as in the paper) |

The main results use the official split so that they can be compared with the published papers. A duplicate-free split (`outputs/splits.csv`) was also built to check for leakage. The dataset is **not redistributed** here; see [`docs/DATA.md`](docs/DATA.md).

### Reporting conventions

- ROUGE-1/2/L are F1 scores from `rouge_score`, using rougeLsum with stemming.
- Checkpoints and summary lengths are chosen on validation only. A2Summ chose on test and sized each summary by its number of gold labels. That sizing is reported separately in `results.json` and marked as such.
- Differences are called significant only when the 95% paired-bootstrap interval excludes zero.
- Results that do not reproduce the papers are reported as they are.

---

## Results

### 1. Text summaries (Daily Mail test)

387 stories. R-1 / R-2 / R-L, F1.

| Model | R-1 | R-2 | R-L |
|---|---:|---:|---:|
| Lead-3 (first three sentences) | 40.72 | 17.73 | 30.71 |
| Extraction oracle (upper bound) | 52.14 | 26.84 | 40.80 |
| M2SM replication, paper learning rate 1e-4 | 38.54 | 15.95 | 28.58 |
| M2SM replication, learning rate 0.01 (used in the app) | 40.10 | 17.81 | 30.88 |
| **Hybrid M2SM + A2Summ** | 40.05 | 17.61 | 30.66 |
| BART-large-CNN, abstractive (60 words on average) | 39.31 | 16.07 | 30.72 |
| *Paper: M2SM (Fu et al. 2020)* | *41.73* | *18.59* | *31.68* |
| *Paper: Lead-3* | *41.07* | *17.87* | *30.90* |

**No model significantly beats Lead-3.** News articles put their key facts first, and the highlights are written from the article. M2SM's published gain over Lead-3 did not reproduce under validation-only model selection.

### 2. Hybrid ablations

Each run removes one part of the full hybrid. The table shows the change in ROUGE-1 on the Daily Mail test set. A positive number means the model did better **without** that part.

| Removed part | R-1 | Change |
|---|---:|---:|
| None (full hybrid) | 40.05 | — |
| Speech | 41.25 | **+1.20** (significant) |
| Video | 40.61 | **+0.56** (significant) |
| Sentence BiLSTM | 40.49 | +0.44 |
| Speech-to-frame time alignment | 40.15 | +0.10 |
| Contrastive losses | 40.09 | +0.04 |
| Late+ fusion | 40.04 | −0.01 |

Adding speech and video **hurt** sentence selection. The video's speech and pictures usually cover only part of the story, so they add noise when the task is choosing article sentences.

### 3. Key frames

Cos score: the best cosine similarity between a chosen frame and the story's reference images (352 test stories with video).

| Method | Cos |
|---|---:|
| DR-DSN | 64.96 |
| Random | 68.67 |
| M2SM replication (paper learning rate) | 68.03 |
| Hybrid | 69.12 |
| Uniform spacing | 69.68 |
| Our frame selector (VASNet-style) | 69.61 |
| Our frame selector + attention to the article | 69.81 |
| **VSUMM (k-means)** | **70.46** |
| *Paper: M2SM* | *69.22* |

The unsupervised VSUMM baseline gives the best frames, so the app uses it whenever there is no trained frame head.

### 4. Video-only mode

When there is no article, Whisper's sentences take its place. Daily Mail test stories with usable speech: 254.

| Method | R-1 | R-2 | R-L |
|---|---:|---:|---:|
| First 5 speech sentences | 15.34 | 1.99 | 11.45 |
| Speech oracle (upper bound) | 16.10 | 4.99 | 11.90 |
| **Speech + video model** | **17.07** | **2.68** | **12.66** |

The trained model beats taking the first speech sentences by a significant margin. Scores are low because the highlights summarize the article, and the speech often doesn't cover it.

Each run's full results and per-story scores are in `outputs/m2sm/runs/`, `outputs/hybrid/runs/` and `outputs/abstractive/`.

---

## The web application

- **Sample stories:** four Daily Mail test stories, drawn at random before any output was seen. Each sample shows the model's picks next to the journalist's highlights, the ROUGE scores, sentence-score bars, key frames and the timed transcript.
- **Your own files:** article only, video only, or both. The video is transcribed in the browser, and only its first 10 minutes are used.
- **AI summary:** an on-demand abstractive summary from DistilBART (`Xenova/distilbart-cnn-6-6`, 8-bit), loaded the first time the button is pressed.
- **How it works:** an animated, nine-step walkthrough of the hybrid model on a real test story, covering tokens, vectors, attention, experts, alignment, fusion, scores and evaluation.

The first run downloads up to about 130 MB of models, which the browser then caches.

---

## Repository layout

```text
.
├── src/
│   ├── build_splits.py              duplicate-aware story splits
│   ├── m2sm_data.py                 loading, tokenizing, labels, ROUGE
│   ├── m2sm_features.py             ResNet-50 frame features (1 fps)
│   ├── transcribe_audio.py          Whisper large-v3-turbo transcripts
│   ├── m2sm_sentence_vectors.py     frozen RoBERTa sentence vectors
│   ├── m2sm_model.py                M2SM: BiLSTMs, bilinear / bi-hop attention, late+ fusion
│   ├── m2sm_train.py                M2SM training and evaluation
│   ├── m2sm_baselines.py            Lead-3, oracle, Random, Uniform, VSUMM, DR-DSN
│   ├── m2sm_frame_selector.py       supervised attention frame selector
│   ├── m2sm_speech_eval.py          video-only (speech) evaluation
│   ├── hybrid_data.py               hybrid inputs and time-alignment masks
│   ├── hybrid_model.py              transformer with modality experts, focal and contrastive losses
│   ├── hybrid_train.py              hybrid training and evaluation
│   ├── abstractive_eval.py          BART-large-CNN evaluation
│   ├── m2sm_export_*.py             ONNX export and sample data for the app
│   ├── explainer_export.py          data for the "How it works" walkthrough
│   └── *_queue.sh                   the exact run queues used for the experiments
├── outputs/
│   ├── splits.csv                   story splits
│   ├── audio/whisper_transcripts.jsonl
│   ├── m2sm/                        baselines, M2SM runs, logs
│   ├── hybrid/                      hybrid and ablation runs
│   └── abstractive/                 BART summaries and scores
├── webapp/                          the in-browser app (static site)
├── presentation/                    slide deck, with speaker notes on the N key
├── presentation-rvu/                the same deck in the RVU template
├── learning/                        beginner guide for the team
└── docs/DATA.md                     how to get the dataset
```

---

## Run locally

### Requirements

- Python 3.12
- About 30 GB of free disk space for MM-AVS
- An Apple-silicon Mac is needed only to regenerate the Whisper transcripts. The transcripts are already included in `outputs/audio/`.

### Install

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Reproduce the experiments

Download MM-AVS into `data/data/` as described in [`docs/DATA.md`](docs/DATA.md), and GloVe 6B 100d into `data/glove/glove.6B.100d.txt`. Then run:

```bash
.venv/bin/python src/build_splits.py
.venv/bin/python src/m2sm_features.py            # ResNet-50 frame features
.venv/bin/python src/m2sm_sentence_vectors.py    # RoBERTa sentence vectors
.venv/bin/python src/m2sm_baselines.py
sh src/m2sm_queue.sh && sh src/m2sm_lr_queue.sh  # M2SM replication runs
sh src/m2sm_speech_queue.sh                      # video-only runs
sh src/hybrid_queue.sh                           # hybrid + ablations
.venv/bin/python src/abstractive_eval.py --split test --beams 1
```

All training is resumable. On an 8 GB laptop's CPU, one hybrid run takes a few hours, so start long queues with `nohup` (and `caffeinate` on macOS).

### Run the web app

The browser models are not stored in git. Export them, then serve the folder:

```bash
.venv/bin/python src/m2sm_export_web.py --run m2sm_full_lr0.01
.venv/bin/python src/m2sm_export_modes.py
cd webapp && python3 -m http.server 8000
```

Open `http://localhost:8000`. Without the cross-origin isolation headers from `webapp/vercel.json`, ONNX Runtime runs on a single thread, which is slower but gives the same results.

---

## Limitations

- No model significantly beats the Lead-3 baseline on Daily Mail, and M2SM's published scores did not reproduce.
- Adding video and speech lowered sentence-selection scores, so the multimodal parts help mainly with frame choice and with stories that have no article.
- The main results use the official split, which contains near-duplicate stories across partitions.
- The frames are sampled at 1 per second instead of the paper's "one of five consecutive frames".
- The browser uses smaller models (Whisper tiny.en, DistilBART) than the offline evaluation (Whisper large-v3-turbo, BART-large-CNN). Its outputs can therefore differ from the reported numbers.
- The app's abstractive summary can state things the source doesn't say, and its extractive summaries are only as good as the sentence splitting of the uploaded PDF. Scanned PDFs are not supported.
- The model was trained only on English Daily Mail news.

---

## Team

| Name | USN |
|---|---|
| S Samraj Pishey | 1RUA24CSE0385 |
| S A Shashwath | 1RUA24CSE0382 |
| Rohan Olekar S G | 1RUA24CSE0372 |
| Ranjith Kumar P S | 1RUA24CSE0360 |

Guided by **Prof. Priyanka S Shivaramaiah**, Assistant Professor, School of Computer Science and Engineering, RV University.

---

## Dataset, citation and licence

### Dataset

MM-AVS is used for research and is **not redistributed** in this repository. The app's four sample stories and the walkthrough include short article excerpts, highlights and video thumbnails from the Daily Mail test set. They are there only to demonstrate the research.

### Papers this work builds on

```bibtex
@article{fu2020m2sm,
  title   = {Multi-modal Summarization for Video-containing Documents},
  author  = {Fu, Xiyan and Wang, Jun and Yang, Zhenglu},
  journal = {arXiv preprint arXiv:2009.08018},
  year    = {2020}
}

@inproceedings{fu2021mmavs,
  title     = {{MM-AVS}: A Full-Scale Dataset for Multi-modal Summarization},
  author    = {Fu, Xiyan and Wang, Jun and Yang, Zhenglu},
  booktitle = {Proceedings of NAACL-HLT},
  pages     = {5922--5926},
  year      = {2021}
}

@inproceedings{he2023a2summ,
  title     = {Align and Attend: Multimodal Summarization with Dual Contrastive Losses},
  author    = {He, Bo and Wang, Jun and Qiu, Jielin and Bui, Trung and Shrivastava, Abhinav and Wang, Zhaowen},
  booktitle = {Proceedings of CVPR},
  year      = {2023}
}
```

The pretrained models (RoBERTa, ResNet-50, Whisper, BART, DistilBART) and the bundled libraries are listed with their licences in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

### Licence

No open-source licence has been chosen for this code yet, so all rights are reserved by the authors. The third-party components keep their own licences.
