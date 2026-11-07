# Task 1: Plant Image Classification

This repository contains the first chapter of a machine learning course report and the code used for the five-class plant image classification task.

## Contents

- `第一章_植物图像分类.txt`: report chapter in plain text.
- `plant_classifier.py`: training, validation, test-time augmentation, and submission generation code.
- `requirements.txt`: Python dependencies.
- `submission_template.csv`: submission format template.

The dataset is not included. Download or obtain it from the course/Kaggle competition and arrange it as shown below. The reported final Mean F1 score is 0.9021, supplied by the experiment author.

## Dataset layout

```text
dataset-for-task2/
  train/
    Black-grass/*.png
    Common wheat/*.png
    Loose Silky-bent/*.png
    Scentless Mayweed/*.png
    Sugar beet/*.png
  test/*.png
submission-for-task2.csv
```

## Setup

Python 3.9 or later is recommended. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

TensorFlow must be installed in a version compatible with the local Python and accelerator environment. For Kaggle, the preinstalled TensorFlow environment can be used.

## Run

Edit `TRAIN_PATH`, `TEST_PATH`, and `TEMPLATE_CSV` near the top of `plant_classifier.py` to point to the local dataset. Then run:

```bash
python plant_classifier.py
```

The script trains EfficientNetB2 with ImageNet weights, stratified validation split, image augmentation, early stopping and learning-rate reduction. It saves the best checkpoint and learning curves in `OUTPUT_DIR`, then produces a submission CSV ordered according to the template.

## Reproducibility notes

The random seed is 42. The model input size is 192 x 192, batch size is 32, maximum epochs is 100, and initial learning rate is 1.5e-4. Dataset files and pretrained ImageNet weights are external and are not bundled. Exact training curves and best epoch are not claimed in the report because the corresponding training logs were not provided.
