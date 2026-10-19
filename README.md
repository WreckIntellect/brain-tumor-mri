# Efficient Deep Learning for Brain Tumor MRI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WreckIntellect/brain-tumor-mri/blob/main/brain_tumor_mri.ipynb)

Four-class brain tumor classification (glioma, meningioma, pituitary, no tumor) on 3,264 MRI
scans, comparing a standard CNN against a **DCFormer-style network** built from decomposed 1D
depthwise convolutions. PyTorch, single notebook, runs on a free Colab T4 GPU in under an hour.

## What this project does

1. **Finds data leakage.** A large share of the dataset's official test images are copies of
   training images. Perceptual hashing (dHash) groups the duplicates, and a new
   duplicate-grouped, stratified 70/15/15 split guarantees no scan appears on two sides.
2. **Trains two models on the clean split.**
   - *Baseline CNN*: four Conv-BN-ReLU-MaxPool blocks.
   - *DCFormer-lite*: a stem plus four stages of residual blocks, each using a 1xk then kx1
     depthwise convolution followed by pointwise channel mixing.
3. **Measures accuracy, macro one-vs-rest AUC, and FLOPs** for both, so the trade-off between
   quality and compute is explicit.
4. **Runs Grad-CAM** on test scans to show which regions drive each model's prediction.
5. **Quantifies 3D savings**: on a 64^3 volume with 32 channels, replacing a 3x3x3 convolution
   with three axis-wise depthwise 1D convolutions plus a pointwise mix.

## Results

Results are written by the notebook to `results.json`. Fill this table from that file after
your run.

| Model          | Accuracy | AUC | MFLOPs | Params |
|----------------|---------:|----:|-------:|-------:|
| Baseline CNN   |          |     |        |        |
| DCFormer-lite  |          |     |        |        |

| Finding                                            | Value |
|----------------------------------------------------|------:|
| Official test images duplicated in training set    |       |
| CNN / DCFormer compute ratio                       |       |
| 3D compute saved by decomposition                  |       |

## Run it

**Colab (recommended):** click the badge above, set *Runtime > Change runtime type > T4 GPU*,
run all cells. The first data cell asks for your Kaggle username and API key
(kaggle.com > Settings > API > Create New Token).

**Locally:**
```bash
pip install -r requirements.txt
DATA_DIR=/path/to/brain-tumor-classification-mri jupyter notebook brain_tumor_mri.ipynb
```

## Repository layout

```
brain_tumor_mri.ipynb   the whole project, every cell commented
requirements.txt        Python dependencies
results.json            written by the notebook after a full run
```

## Dataset

Bhuvaji et al., *Brain Tumor Classification (MRI)*, Kaggle, 2020.
`sartajbhuvaji/brain-tumor-classification-mri`. 2,870 training and 394 test JPEG scans across
four classes.

## Method notes

- **Why group by duplicate before splitting?** Random splitting puts copies of the same scan on
  both sides, so test accuracy partly measures memorisation. Grouping removes that.
- **Why decomposed depthwise convolutions?** A kxk convolution over C channels costs about
  k^2 * C^2 multiply-adds per pixel. Depthwise drops the C^2 to C; decomposing kxk into 1xk and
  kx1 drops k^2 to 2k. Pointwise 1x1 convolutions restore channel mixing cheaply.
- **FLOP counting** uses forward hooks on every Conv and Linear layer and reports
  multiply-adds for a single 1x128x128 input.

## Author

Bharati Gehani, MS Engineering Management, USC Viterbi School of Engineering.
