# Multimodal news summarization project: status since the M2SM rebuild

I'm a student working on an NLP course project titled "Multimodal News Summarization with Relevance-Filtered Heterogeneous Sources". It is summarization only: extractive, picking sentences, not generating articles. I'd like your advice on how to bring **audio** and **images** into the summary in a way that works on this dataset and hardware. Below is everything done since I restarted the project to replicate a published paper (from 2026-11-03).

## 1. What I set out to do

Replicate **M2SM** from Fu et al. 2020, "Multi-modal Summarization for Video-containing Documents" (arXiv 2009.08018), on the same **MM-AVS** dataset, within limited time and compute. I don't need a table-by-table match, but I must compare against the baselines honestly.

Paper targets, Daily Mail, ROUGE-1/2/L:

| Model | ROUGE-1 / 2 / L |
|---|---|
| Text only | 39.11 / 16.42 / 28.56 |
| + video | 40.86 / 17.48 / 30.23 |
| + transcripts | 41.26 / 17.95 / 30.98 |
| M2SM | 41.73 / 18.59 / 31.68 |
| Lead3 | 41.07 / 17.87 / 30.90 |

The paper also reports a frame-selection score of Cos 69.22.

## 2. Constraints

- **Hardware:** an 8 GB M-series MacBook, CPU only, and a budget of about ₹2,500. No GPU training. Long jobs run detached overnight.
- **No tuning on test:** everything is chosen on validation and scored on test once. The Daily Mail test set has now been looked at many times, so any new method must be fixed on validation before its test score is seen.
- **Extractive only:** summaries are selected sentences, never generated text.
- **No CLIP for now:** I asked not to use CLIP at this stage. It is open to revisit if you think it's the right tool.
- **Demo:** a static Vercel site that runs everything in the browser (ONNX Runtime Web, plus Whisper tiny.en through transformers.js). Hobby plan: 100 MB upload limit and no server-side model.

## 3. Dataset as used (MM-AVS)

- **Story folders:** each Daily Mail story folder has `article.txt`, `article_segment.txt` (one sentence per line), `highlight.txt` (the reference summary), `title.txt`, `img_caption.txt`, the article's images (`0.png`, …), `audio.mp3`, `video.mp4` and `transcript.txt`. The transcript is ASR output with timestamps, mostly unpunctuated, with `%HESITATION` tokens. CNN folders are similar, using `artitle_section.txt`.
- **Stories with usable video:** 2,144.
- **Daily Mail splits:** train 1,359 / val 194 / test 387, using the authors' official test IDs, with the rest split 7:1 into train and val. CNN is test only (204 stories), as in the paper.
- **Labels:** the authors' sentence labels for Daily Mail, and greedy ROUGE labels for CNN.
- **Transcripts:**
  - The dataset's own transcripts cover only part of Daily Mail: train 641 / val 86 / test 178.
  - I also have Whisper large-v3-turbo transcripts for 1,661 stories, with punctuation and timed segments. After filtering for confidence (`no_speech_prob` ≤ 0.6, `avg_logprob` ≥ −1.0, at least 5 words), the usable stories are train 827 / val 126 / test 254.
- **Video features:** ResNet-50 (ImageNet V1), 2048-d per frame, sampled at **1 frame per second**. The paper uses one of every five consecutive frames; I chose 1 fps for time.
- **Reference images:** these are the article's images. They are used **only** as ground truth for frame selection (the Cos score), across 352 test stories with images, and as training labels for my frame selector. They are **not** currently an input to any summarizer.
- **Text features:** GloVe 100d, vocabulary 19,531, at most 100 sentences, 60 words per sentence and 300 transcript words.
- **Overlap statistics:** Daily Mail transcripts score ROUGE-1/2/L of **20.56 / 5.10 / 9.64** against the article, but only **14.02 / 2.11 / 8.01** against the reference summary. The speech often covers a narrower moment than the article.

## 4. What was built (all in `src/`)

- **The M2SM model** (`m2sm_model.py`):
  - Word- and sentence-level BiLSTMs (64 hidden per direction) and a frame BiLSTM.
  - Bilinear attention (Eq. 4-7) and bi-hop attention, sentence → transcript words → frames (Eq. 8). Stories with no transcript fall back to direct single-step sentence → frame attention.
  - Early, tensor, late and late+ fusion (Eq. 9, β = 0.3).
  - Bi-stream training: BCE on sentences, plus REINFORCE on frames with diversity and representativeness rewards (Eq. 11-12, α ratio 3.33). Adagrad, early stopping, and summary size (2-5) chosen on validation.
- **Data and evaluation** (`m2sm_data.py`, `m2sm_train.py`):
  - ROUGE via `rouge_score` with stemming, summary-level ROUGE-Lsum. My lead3 is 40.72 / 17.73 / 30.71 against the paper's 41.07 / 17.87 / 30.90, so the measurement is close.
  - The training script is resumable and has a video-only mode (`--speech`) where Whisper sentences replace the article.
- **Baselines** (`m2sm_baselines.py`): lead3, the label oracle, and transcript-overlap statistics for text. For frames: Random, Uniform, VSUMM (k-means on ResNet features) and DR-DSN.
- **Frame selectors** (`m2sm_frame_selector.py`, my addition): a supervised VASNet-style self-attention scorer trained on reference-image similarity. An `--article` variant adds frame → sentence bilinear attention over averaged GloVe sentence vectors.
- **Exports and app:** `m2sm_export_web.py` and `m2sm_export_modes.py` export to ONNX, and `webapp/` holds the browser app.

**Known deviations from the paper:**
- 1 fps instead of the paper's one-of-five frames.
- The single-step fallback for stories with no transcript.
- Rewards computed on L2-normalised ResNet features.
- Average pooling, and a tensor-fusion projection of size 32.
- Batch size 1, patience 5, at most 30 epochs.
- GloVe 100d.
- REINFORCE details: 5 episodes, a moving-average baseline, and a rate penalty of 0.01 towards 0.5.
- The bi-hop key/value choice is my reading of the paper.

## 5. Results (Daily Mail test, 387 stories, ROUGE-1/2/L)

| Model | LR 1e-4 (paper's) | LR 1e-3 | LR 1e-2 |
|---|---|---|---|
| Text only | 39.85 / 17.73 / 30.71 | 39.83 / 17.73 / 30.69 | **41.23 / 18.03 / 31.05** |
| + video (early fusion) | 39.84 / 17.73 / 30.70 | 39.83 / 17.73 / 30.69 | 40.49 / 18.21 / 31.15 |
| + video + transcript | 39.86 / 17.73 / 30.72 | 39.83 / 17.73 / 30.69 | 40.86 / 17.76 / 30.71 |
| Full M2SM (late+, bi-stream) | 38.54 / 15.95 / 28.58 | 39.83 / 17.73 / 30.69 | 40.10 / 17.81 / 30.88 |

The extraction oracle scores 52.14 / 26.84 / 40.80. On CNN, lead3 scores 26.17 / 9.87 / 20.26 and text-only at LR 1e-2 scores 26.53 / 10.09 / 20.57.

**What the results show** (paired bootstrap, 10,000 resamples, 95% CI, single seed):

- **Opening-sentence copying:** at LR 1e-4 and 1e-3, models just pick the opening sentences. They copy them exactly on 99.5-100% of stories, except full M2SM at 1e-4, which copies on about 47% but scores lowest. Only LR 1e-2 learns, and those models still copy the opening on 53-67% of stories.
- **Adding modalities hurts at LR 1e-2**, compared with text only:

| Added to text only | ROUGE-1 change | ROUGE-2 and ROUGE-L |
|---|---|---|
| Video | −0.77, significant | Not significant |
| Video + transcript | −0.37, significant | Not significant |
| Full M2SM | −1.13, significant | Not significant |

  Even on the 178 stories that have a transcript, adding it doesn't help.
- **No model significantly beats lead3.** Text-only is +0.51 / +0.30 / +0.34, with every interval including zero. Full M2SM is significantly worse on ROUGE-1, at −0.62.
- **Comparison with the paper:** our text-only model is about 2 points *above* the paper's, and our full M2SM is 0.8-1.6 points *below* it. The paper's "multimodal helps" story is therefore not reproduced, partly because our text baseline is much stronger.

**Frame selection** (Cos, cosine between reference images and chosen frames, k = number of images, 352 test stories):

| Method | Cos |
|---|---|
| VSUMM | **70.46** |
| My attention selector | 69.81 |
| Uniform | 69.68 |
| My plain selector | 69.61 |
| Random | 68.67 |
| M2SM, LR 1e-3 | 68.29 |
| M2SM, LR 1e-4 | 68.03 |
| DR-DSN | 64.96 |
| M2SM, LR 1e-2 | 59.67 |

- **Selectors vs VSUMM:** both of my selectors are significantly below VSUMM and significantly above Random.
- **The attention does nothing useful:** on validation, swapping in a different story's article leaves 99.9% of picks unchanged. The attention distribution over sentences is almost uniform (normalised entropy 0.998). GloVe sentence vectors and ImageNet ResNet features don't share a space, so the attention has nothing to latch onto.
- **Narrow range:** the gap between Random and VSUMM is only about 1.8 points, and the score seems to reward spreading picks across the video more than choosing particular frames.

**Video-only mode** (Whisper sentences instead of the article; 254 test stories with usable speech):

- **Chosen model:** `speech_video_tuned`, fine-tuned from the video model, was fixed on validation before its test score was seen. It scores 17.07 / 2.68 / 12.66.
- **Against simple baselines:** it beats the first 5 speech sentences (15.34 / 1.99 / 11.45) by +1.73 / +0.69 / +1.21, all significant. The speech oracle is 16.10 / 4.99 / 11.90, so the ceiling is very low.
- **Against articles:** article models on the same stories score about 40-41 ROUGE-1, so they are about 23 ROUGE-1 points higher. Adding video frames to speech text doesn't help either: the video model is 0.53 ROUGE-1 below speech text alone, with ROUGE-2 and ROUGE-L unchanged.

## 6. Web app (built locally, not yet deployed)

- **Routes:** upload an article PDF, an MP4, or both.
  - Article + video uses full M2SM at LR 1e-2.
  - Article only uses text-only at LR 1e-2.
  - Video only uses the speech model, with VSUMM key frames.
  - A video with no usable speech shows key frames only.
- **Checks done:** all the exported models match PyTorch to within 1e-6. The JavaScript VSUMM gets Cos 70.23 against Python's 70.52 on validation. All four routes plus the empty-upload error were tested in Chrome.
- **Known gap:** the browser uses Whisper tiny.en, while training used large-v3-turbo. Tiny merged one story's speech into a single sentence.

## 7. Not done yet

- Extra random seeds.
- Human ratings, for which I'll collect informativeness and satisfaction on a 1-5 scale.
- A rerun on the split without near-duplicates.
- Testing the alternative bi-hop key choice.
- The report rewrite, deploying to Vercel, and committing to git.

## 8. Where audio and images stand now

- **Audio** only enters as ASR text, either the dataset's transcripts or Whisper. Nothing uses the audio signal itself, such as prosody, speaker changes or sound events. Speech text gives no gain over the article, and on its own it is a weak summary source.
- **Images:** the article's own images and their captions (`img_caption.txt`) are unused as inputs. They only serve as frame-selection ground truth. Video frames enter as ResNet features, which don't align with the text, so cross-modal attention stays flat.

## 9. What I'd like from you

1. **Approaches:** what approaches, ideally from published papers that could run on this dataset, would let audio and images actually contribute to an extractive text summary and to choosing an image or frame? They need to fit a CPU-only 8 GB laptop.
2. **Using the images as input:** how should I use the article images and captions as inputs without leaking the frame-selection ground truth?
3. **Proving a modality helps:** what evaluation would show that a modality helps, given that ROUGE references are written from the article?

Two papers I'm considering:
- **Li et al., EMNLP 2017**, "Multi-modal Summarization for Asynchronous Collection of Text, Image, Audio and Video" (aclanthology.org/D17-1114). It is extractive and uses submodular optimisation for salience, non-redundancy, readability and coverage. It brings in speech transcripts selectively, and images through text-image matching or captioning. It needs little training.
- **He et al., CVPR 2023**, "Align and Attend: Multimodal Summarization with Dual Contrastive Losses" (A2Summ, arXiv 2303.07284). It is a transformer with alignment-guided self-attention, using time correspondence between video and transcript, plus inter- and intra-sample contrastive losses. It reports results on Daily Mail and CNN multimodal summarization; I haven't confirmed whether that is the same MM-AVS data.
