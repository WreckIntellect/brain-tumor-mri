# Efficient Deep Learning for Brain Tumor MRI, plus two companion studies

Three self-contained, line-by-line commented notebooks. Each runs on a free Colab T4 GPU and
is written so that a reader with no deep learning background can follow every step.

| Notebook | Question it answers | Open |
|---|---|---|
| `brain_tumor_mri.ipynb` | Can a cheap decomposed-convolution network match a CNN on 4-class tumor MRI, and is the dataset's official split leaking? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WreckIntellect/brain-tumor-mri/blob/main/brain_tumor_mri.ipynb) |
| `clip_vs_coop_mri.ipynb` | Can CLIP, trained on internet photos, read an MRI zero-shot, and how far does learning 4 prompt vectors (CoOp) lift it? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WreckIntellect/brain-tumor-mri/blob/main/clip_vs_coop_mri.ipynb) |
| `eeg_cp_decomposition.ipynb` | Can 4 CP tensor components summarise 59-channel EEG well enough to decode imagined movement from a single trial? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WreckIntellect/brain-tumor-mri/blob/main/eeg_cp_decomposition.ipynb) |

## 1. Brain tumor MRI: CNN vs DCFormer-style network

Four-class classification (glioma, meningioma, pituitary, no tumor) on 3,264 MRI scans
(Bhuvaji et al., Kaggle 2020). PyTorch.

- **Leakage audit.** Perceptual hashing (dHash) groups duplicate scans. A duplicate-grouped,
  stratified 70/15/15 split guarantees no scan appears on two sides.
- **Baseline CNN:** four Conv-BN-ReLU-MaxPool blocks.
- **DCFormer-lite:** residual blocks of 1xk then kx1 depthwise convolutions plus pointwise mixing.
- **Metrics:** accuracy, macro one-vs-rest AUC, FLOPs (forward hooks), parameter count.
- **Grad-CAM** on test scans for both models.
- **3D study:** compute of a 3x3x3 convolution vs three axis-wise depthwise 1D convolutions.

## 2. CLIP zero-shot vs CoOp prompt learning

Same data, same clean split. CLIP ViT-B/32 is frozen throughout.

- **Zero-shot:** three prompt templates, image-text cosine similarity.
- **CoOp:** 4 learnable context vectors (2,048 parameters) prepended to the class names, trained
  with cross-entropy through the frozen text encoder; best epoch picked on validation.
- Reports accuracy and AUC for both, confusion matrices, and the nearest real tokens to the
  learned context.

## 3. CP tensor decomposition of 59-channel EEG

BCI Competition IV, data set 1 (Berlin BCI group): 59-channel motor imagery, 100 Hz
calibration data, real subjects a/b/f/g.

- 8-30 Hz band-pass, 0.5-3.5 s post-cue epochs, per-trial standardisation.
- Rank-4 CP (tensorly) on `trials x channels x time`. Spatial maps and time courses plotted.
- **Honest decoding:** CP fitted on training folds only, test trials projected by least squares
  onto the fixed spatial and temporal factors, LDA on the 4 loadings, 10-fold CV.
- Compared against a 59-feature log band-power LDA baseline. Rank sweep 1 to 8.

## Results

Each notebook writes a `results_*.json`. Fill these tables from your run.

| MRI model | Accuracy | AUC | MFLOPs | Params |
|---|---:|---:|---:|---:|
| Baseline CNN | | | | |
| DCFormer-lite | | | | |

| Vision-language | Accuracy | AUC |
|---|---:|---:|
| Chance | 25.0% | 0.500 |
| Zero-shot CLIP (best prompt) | | |
| CoOp, 4 context vectors | | |

| EEG decoder (mean over 4 subjects) | 10-fold CV accuracy |
|---|---:|
| Chance | 50.0% |
| CP rank-4 loadings + LDA | |
| Log band-power (59 features) + LDA | |

## Running

**Colab:** click a badge, set *Runtime > Change runtime type > T4 GPU*, run all cells.
- The MRI notebooks ask for your Kaggle username and API key (kaggle.com > Settings > API).
- The EEG notebook needs the `BCICIV_calib_ds1*.mat` (100 Hz) files from
  https://www.bbci.de/competition/iv/ (free sign-up form, link by email). Upload them to the
  Colab file panel or your Drive.

**Locally:**
```bash
pip install -r requirements.txt
DATA_DIR=/path/to/brain-tumor-classification-mri jupyter notebook brain_tumor_mri.ipynb
EEG_DIR=/path/to/mat/files jupyter notebook eeg_cp_decomposition.ipynb
```

## References

- Bhuvaji, S. et al. Brain Tumor Classification (MRI). Kaggle, 2020.
- Radford, A. et al. Learning Transferable Visual Models From Natural Language Supervision. ICML 2021. (CLIP)
- Zhou, K. et al. Learning to Prompt for Vision-Language Models. IJCV 2022. (CoOp)
- Blankertz, B. et al. BCI Competition IV, data set 1. Berlin BCI group, 2008.
- Kolda, T. and Bader, B. Tensor Decompositions and Applications. SIAM Review, 2009.

## Author

Bharati Gehani, MS Engineering Management, USC Viterbi School of Engineering.
