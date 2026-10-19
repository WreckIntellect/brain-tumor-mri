# Brain Tumor MRI + two side projects

Three notebooks, all run on a free Colab T4.

| Notebook | What it does | Open |
|---|---|---|
| `brain_tumor_mri.ipynb` | CNN vs DCFormer-lite on 4-class tumor MRI, plus a leakage check on the Kaggle split | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WreckIntellect/brain-tumor-mri/blob/main/brain_tumor_mri.ipynb) |
| `clip_vs_coop_mri.ipynb` | Zero-shot CLIP on MRI, then CoOp prompt learning on top | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WreckIntellect/brain-tumor-mri/blob/main/clip_vs_coop_mri.ipynb) |
| `eeg_cp_decomposition.ipynb` | CP tensor decomposition on 59-channel EEG, decode motor imagery from 4 loadings | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WreckIntellect/brain-tumor-mri/blob/main/eeg_cp_decomposition.ipynb) |

## 1. Brain tumor MRI

Kaggle dataset (sartajbhuvaji), 3,264 scans, 4 classes. PyTorch.

The official train/test split has a lot of duplicate images across both sides, so I fingerprint every scan with dHash and rebuild a 70/15/15 split by duplicate group.

Models: a 4-block CNN baseline and a DCFormer-style net (1xk + kx1 depthwise convs, pointwise mixing, residuals). Compared on accuracy, macro AUC, FLOPs and params. Grad-CAM on test scans for both. Last cell compares a 3x3x3 conv vs three 1D depthwise convs on a dummy 3D volume.

## 2. CLIP vs CoOp

Same data, same split. CLIP ViT-B/32 frozen.

Zero-shot with 3 prompt templates, then CoOp: 4 learnable context vectors (2,048 params) in front of the class names, trained through the frozen text encoder. Reports accuracy, AUC, confusion matrices, and the nearest real tokens to the learned context.

## 3. EEG CP decomposition

BCI Competition IV dataset 1, 59 channels, 100 Hz, subjects a/b/f/g.

8-30 Hz band-pass, 0.5-3.5 s epochs. Rank-4 CP with tensorly. For decoding, CP is fit on train folds only, test trials are projected onto the fixed factors, LDA on the 4 loadings, 10-fold CV. Compared against log band-power + LDA. Rank sweep at the end.

## Results

Each notebook writes a `results_*.json`.

| MRI model | Accuracy | AUC | MFLOPs | Params |
|---|---:|---:|---:|---:|
| Baseline CNN | | | | |
| DCFormer-lite | | | | |

| Vision-language | Accuracy | AUC |
|---|---:|---:|
| Chance | 25.0% | 0.500 |
| Zero-shot CLIP (best prompt) | | |
| CoOp | | |

| EEG (mean over 4 subjects) | 10-fold CV accuracy |
|---|---:|
| Chance | 50.0% |
| CP rank-4 + LDA | |
| Log band-power + LDA | |

## Running

Colab: open a badge, set runtime to T4 GPU, run all.
- MRI notebooks need a Kaggle username + API key (kaggle.com > Settings > API).
- EEG notebook needs the `BCICIV_calib_ds1*.mat` (100 Hz) files from https://www.bbci.de/competition/iv/. Upload to Colab or Drive.

Local:
```bash
pip install -r requirements.txt
DATA_DIR=/path/to/brain-tumor-classification-mri jupyter notebook brain_tumor_mri.ipynb
EEG_DIR=/path/to/mat/files jupyter notebook eeg_cp_decomposition.ipynb
```

## References

- Bhuvaji et al., Brain Tumor Classification (MRI), Kaggle 2020
- Radford et al., CLIP, ICML 2021
- Zhou et al., CoOp, IJCV 2022
- Blankertz et al., BCI Competition IV data set 1, 2008
- Kolda & Bader, Tensor Decompositions and Applications, SIAM Review 2009

## Author

Bharati Gehani, MS Engineering Management, USC Viterbi
