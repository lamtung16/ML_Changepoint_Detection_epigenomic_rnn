# Changepoint Detection with Learned Penalty

## Overview

Changepoint detection problem in sequential data is to identify points in the data where the statistical properties change. This is particularly useful in various fields such as finance, medicine, and environmental monitoring.

Changepoint detection algorithms, such as OPART, PELT, and FPOP, can effectively locate these changepoints. However, to control the number of detected changepoints, a penalty parameter is used. A smaller penalty value leads to the detection of more changepoints, while a larger penalty results in fewer.

This study focuses on learning the penalty value for each sequence in changepoint detection algorithm.

## Problem Setting

The changepoint detection process consists of two main steps:

1. **Feature Extraction:** Extract features from sequences to prepare them for model training.
2. **Penalty Learning Model:** Learn the penalty lambda from the extracted features.

### Previous Methods to Learn Penalty from Labeled Sequences

The following traditional methods have been explored for learning the penalty from labeled sequences:

- **Linear Models:** Learning Sparse Penalties for Change-Point Detection using Max Margin Interval Regression (https://inria.hal.science/hal-00824075)
- **Tree-Based Models:** Maximum Margin Interval Trees (MMIT) (https://arxiv.org/abs/1710.04234)
- **Ensemble Trees:** Utilizing Accelerated Failure Time (AFT) in XGBoost (https://www.tandfonline.com/doi/full/10.1080/10618600.2022.2067548)

### Proposed Method

This project proposes an advanced learning model for penalty estimation:
- **Feature Extraction:** Implementing Recurrent Neural Network (RNN) architectures, including:
  - Original RNN
  - Long Short-Term Memory (LSTM)
  - Gated Recurrent Unit (GRU)
- **Penalty Learning Model:** Multi-Layer Perceptron (MLP)

### Experimented proposed model
- mlp: Utilizes the extracted features as input.
- rnn, lstm, gru: Employ sequences as input to derive features, with a linear model serving as the penalty learning model.

### Future experiments proposed model
RNN_MLP, LSTM_MLP, GRU_MLP: Use sequences as input to extract features, with an MLP model utilized for penalty learning.

## Results
![image1.png](figures\pngs\ATAC_JV_adipose.png)
![image1.png](figures\pngs\CTCF_TDH_ENCODE.png)
![image1.png](figures\pngs\H3K4me1_TDH_BP.png)
![image1.png](figures\pngs\H3K4me3_PGP_immune.png)
![image1.png](figures\pngs\H3K4me3_TDH_ENCODE.png)
![image1.png](figures\pngs\H3K4me3_TDH_immune.png)
![image1.png](figures\pngs\H3K4me3_TDH_other.png)
![image1.png](figures\pngs\H3K4me3_XJ_immune.png)
![image1.png](figures\pngs\H3K9me3_TDH_BP.png)
![image1.png](figures\pngs\H3K27ac_TDH_some.png)
![image1.png](figures\pngs\H3K27ac-H3K4me3_TDHAM_BP.png)
![image1.png](figures\pngs\H3K27me3_RL_cancer.png)
![image1.png](figures\pngs\H3K27me3_TDH_some.png)
![image1.png](figures\pngs\H3K36me3_AM_immune.png)
![image1.png](figures\pngs\H3K36me3_TDH_ENCODE.png)
![image1.png](figures\pngs\H3K36me3_TDH_immune.png)
![image1.png](figures\pngs\H3K36me3_TDH_other.png)