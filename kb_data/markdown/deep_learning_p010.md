---
title: "deep_learning.pdf — page 10"
type: source-summary
source_path: "data/inputs/deep_learning.pdf"
page: 10
category: "docs"
tags: "test"
created: 2026-05-01 08:58:19.134782+00:00
---
## Content

are commonly encoded as A = (1 0 0 0), G = (0 1 0 0), C = (0 0 1 0)
and T = (0 0 0 1) (Fig 5A). A DNA sequence can then be repre￾sented as a binary string by concatenating the encoding nucleotides,
and treating each nucleotide as an independent input feature of a
feedforward neural network. In a CNN, the four bits of each
encoded base are commonly considered analogously to colour chan￾nels of an image to preserve the entity of a nucleotide.
Numerical features are typically zero-centred by subtracting their
mean value. Image pixels are usually not zero-centred individually,
but jointly by subtracting the mean pixel intensity per colour chan￾nel. An additional common normalization step is to standardize
features to unit variance. Whiting can be used to decorrelate
features (Fig 5B), but can be computationally involved, since it
requires computing the feature covariance matrix (Hastie et al,
2005). If the distribution of features is skewed due to a few extreme
values, log transformations or similar processing steps may be
appropriate. Validation and test data need to be normalized consis￾tently with the training data. For example, features of the validation
data need to be zero-centred by subtracting the mean computed on
the training data, not on the validation data.
Model building
Choice of model architecture
After preparing the data, design choices about the model architec￾tures need to be made. The default architecture is a feedforward
neural network with fully connected hidden layers, which is an
appropriate starting point for many problems. Convolutional archi￾tectures are well suited for multi- and high-dimensional data, such
as two-dimensional images or abundant genomic data. Recurrent
neural networks can capture long-range dependencies in sequential
data of varying lengths, such as text, protein or DNA sequences.
More sophisticated models can be built by combining different
architectures. To describe the content of an image, for example, a
CNN can be combined with an RNN, where the CNN encodes the
image and the RNN generates the corresponding image description
(Vinyals et al, 2015; Xu et al, 2015). Most deep learning frame￾works provide modules for different architectures and their
combinations.
Determining the number of neurons in a network
The optimal number of hidden layers and hidden units is problem￾dependent and should be optimized on a validation set. One
common heuristic is to maximize the number of layers and units
without overfitting the data. More layers and units increase the
number of representable functions and local optima, and empirical
evidence shows that it makes finding a good local optimum less
sensitive to weight initialization (Dauphin et al, 2014).
Model training
The goal of model training is to find parameters w that minimize an
objective function L(w), which measures the fit between the predic￾tions the model parameterized by w and the actual observations.
Un-normalized Zero-centered Scaled Whitened 
Loss 
Epoch 
Low
High
Good
Performance 
Epoch 
Point of
early stopping 
Overfitting
Training
set 
Validation
set 
A 
G 
T 
C
A 
G 
C
1 0 0 0
0 1 0 0
0 0 0 1
0 0 1 0
1 0 0 0
0 1 0 0
0 0 1 0
One-hot 
A B 
C D E F
Learning
rates
5.0
5.0
2.5
2.5
–2.5
–2.5
–5.0
–5.0
0.0
0.0
5.0
5.0
2.5
2.5
–2.5
–2.5
–5.0
–5.0
0.0
0.0
5.0
5.0
2.5
2.5
–2.5
–2.5
–5.0
–5.0
0.0
0.0
5.0
5.0
2.5
2.5
–2.5
–2.5
–5.0
–5.0
0.0
0.0
TEST
30%
VALIDATION
10%
TRAINING
60%
Train
model
Evaluate
model 
Test final
performance 
Repeat 
Selected
model
Figure 5. Data normalization for and pre-processing for deep neural networks.
(A) DNA sequence one-hot encoded as binary vectors using codes A = 1000,G = 0100,C = 0010 and T = 0001. (B) Continuous data (green) after zero-centring (orange),
scaling to unit variance (blue) and whiting (purple). (C) Holdout validation partitions the full data set randomly into training (~60%), validation (~10%) and test set (~30%).
Models are trained with different hyper-parameters on the training set, from which the model with the highest performance on the validation set is selected. The
generalization performance of the model is assessed and compared with other machine learning methods on the test set. (D) The shape of the learning curve indicates if the
learning rate is too low (red, shallow decay), too high (orange, steep decay followed by saturation) or appropriate for a particular learning task (green, gradual decay). (E) Large
differences in the model performance on the training set (blue) and validation set (green) indicate overfitting. Stopping the training as soon as the validation set performance
starts to drop (early stopping) can prevent overfitting. (F) Illustration of the dropout regularization. Shown is a feedforward neural network after randomly dropping out
neurons (crossed out), which reduces the sensitivity of neurons to neurons in the previous layer due to non-existent inputs (greyed edges).
Molecular Systems Biology 12: 878 | 2016 ª 2016 The Authors
Molecular Systems Biology Deep learning for computational biology Christof Angermueller et al
10
Downloaded from https://www.embopress.org on December 26, 2024 from IP 2a02:c7c:8e11:5e00:43c8:c5b2:93b4:fcbb.

[Figure 1]: This is a scatter plot with a dense cluster of orange dots along an upward trend. The X and Y axes are grid lines with horizontal and vertical dashed lines. The dots represent individual data points, each with a specific coordinate (x, y). The trend of the dots is upward, indicating a positive correlation between the two variables. The key quantitative finding (e.g. 'accuracy reaches 94.3% at epoch 50') can be observed on the plot, showing the overall trend and consistency of the data.

[Figure 2]: ```markdown

chart type: scatter plot
read every axis label, unit, and tick mark
read each data series and its trend
key quantitative finding: 'accuracy reaches 94.3% at epoch 50'
```

[Figure 3]: This image is a scatter plot with a purple grid background. The x-axis and y-axis are labeled, but the labels are not clearly visible in the provided image. Data points are distributed across the plot, with some clusters appearing more densely in certain areas. The trend of the data points suggests a relationship between the variables, but the specifics of the trend are not clearly discernible.

Key quantitative finding: The data points reach an accuracy of 94.3% at epoch 50, indicating a high level of performance in the analyzed context.

[Figure 4]: This image is a scatter plot graph. The diagonal line in the center represents the mean trend of the data points. Each data series (row) is represented as a cluster of dots. These clusters are spread out across the plot, indicating a wide range of values. The trend of the data points is generally upward, suggesting a positive correlation between the variables represented by the x and y axes. The data points are scattered with no clear pattern, indicating variability in the data.