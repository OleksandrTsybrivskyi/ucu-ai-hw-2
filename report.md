---
title: Homework 2 -- Multiple Object Video Detection
author: Oleksandr Tsybrivskyi
course: Artificial Intelligence · UCU
date: 2026-11-16
---

## Introduction

In this homework we train a deep learning model capable of identifying objects on a video. We did try different approaches, including One-stage Anchor-free YOLOv8n and Query-based RT-DETR.

## Method

Lets break down steps done in this lab.

### Data preparation

The code was launched in a Kaggle Notebook, so there was no need to download dataset. Then we analyzed objects sizes on all the photos and created a plot.

![Objects sizes](images/objects-sizes-distribution.png)

It shows that many objects have sizes 10-30px. By default model would compress image to about 640x640px. Little objects would become difficult to identify so we increased image size setting in models to 1024px but it slowed performance. Also photos from the same sequence are correlated and could cause dataleakage if one gets to train set and other to validation set, so we divide dataset by sequences.

Next we mask ignored areas with (128,128,128) gray color, so the model doesnt get penalized for mis identification.

The masked images were created to /kaggle/working because input folder is read only

### One-stage Anchor-free YOLOv8n

Ultralitics is hardcoded to names images and labels, so for it to work we renamed `images_masked/` and `labels_masked/` folders back to `images/` and `labels/`. Also we tried to use 2 notebooks: first would prepare data and use *save and commit* button to share this data and the second would use it as input. But it did not allow to rename the folders in a second notebook, so we switched to 1 notebook variant. Later I understood that input and working folders can have folders with the same name inside, so we could just not create `images_masked/` and `labels_masked/` but create `images/` and `labels/` and there would be no need to rename them back, and 2 notebooks variant would work.

Also 2 GPUs couldnt do the job because of the bug with PyTorch, so only one was used for all the calculations in this homework.

YOLO means You only look once. It does look at image and analyze it only one time. Also it does not use predetermined sizes for boxes and guesses the box size by itself. so it is anchor free one stage. It is nano version so is camparably fast. Still on one GPU with 10 epochs and 1024 imgsize it took about one hour to train it.



Inline math works with single dollars: $f(x) = \sigma(Wx + b)$. Display math uses double dollars:

$$
\mathcal{L}(\theta) = -\frac{1}{N}\sum_{i=1}^{N} y_i \log \hat{y}_i
$$

Code blocks are highlighted:

```python
def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()
```

## Results

| Model    | Accuracy | F1   |
|----------|----------|------|
| Baseline | 0.71     | 0.68 |
| Ours     | 0.84     | 0.82 |

Images go into the `images/` folder:

<!-- ![Training loss](images/loss.png) -->

## Conclusion

> Summarize your findings here.
