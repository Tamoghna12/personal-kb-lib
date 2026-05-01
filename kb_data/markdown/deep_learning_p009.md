---
title: "deep_learning.pdf — page 9"
type: source-summary
source_path: "data/inputs/deep_learning.pdf"
page: 9
category: "docs"
tags: "test"
created: 2026-05-01 08:58:19.129682+00:00
---
## Content

TensorFlow provides the most efficient implementation for RNNs.
The software is recent and under active development; hence, only
few pre-trained models are currently available.
Data preparation
Training data are key for every machine learning application. Since
more data with informative features usually result in better perfor￾mance, effort should be spent on collecting, labelling, cleaning and
normalizing data.
Required data set sizes
Most of the successful applications of deep learning have been in
supervised learning settings, where sufficient labelled training
samples are available to fit complex models. As a rule of thumb, the
number of training samples should be at least as high as the number
of model parameters, although special architectures and model regu￾larization can help to avoid overfitting if training data are scarce
(Bengio, 2012).
Central problems in regulatory genomics, for example predicting
molecular traits from genotype, are limited in the number of training
instances; hundreds to at most tens of thousands of training exam￾ples are typical. The strategy of considering sequence windows
centred on the trait of interest (e.g. splice site, transcription factor
binding site or epigenetic marks; see Fig 2A) is now a widely used
approach and helps increasing the number of input–output pairs
from a single individual.
In image analysis, data can be abundant, but manually curated
and labelled training examples are typically difficult to obtain. In
such instances, the training set can be augmented by scaling, rotat￾ing or cropping the existing images, an approach that also enhances
robustness (Krizhevsky et al, 2012). Another strategy is to reuse a
network that was pre-trained on a large data set for image recogni￾tion [e.g. AlexNet (Krizhevsky et al, 2012), VGG (Simonyan &
Zisserman, 2014), GoogleNet (Szegedy et al, 2015b) or ResNet (He
et al, 2015)], and to fine-tune its parameters on the data set of
interest (e.g. microscopy images for a particular segmentation
task). Such an approach exploits the fact that different data sets
share important characteristics and features, such as edges or
curves, which can be transferred between them. Caffe, Lasagne,
Torch and to a limited extend TensorFlow provide repositories with
pre-trained models.
Partitioning data into training, validation and test sets
Machine learning models need to be trained, selected and tested on
independent data sets to avoid overfitting and assure that the model
will generalize to unseen data. Holdout validation, partitioning the
data into a training, validation and test sets, is the standard for deep
neural networks (Fig 5C). The training set is used to learn models
with different hyper-parameters, which are then assessed on the
validation set. The model with best performance, for example
prediction accuracy or mean-squared error, is selected and further
evaluated on the test set to quantify the performance on unseen data
and for comparison to other methods. Typical data set proportions
are 60% for training, 10% for validation and 30% for model testing.
If the data set is small, k-fold cross-validation or bootstrapping can
be used instead (Hastie et al, 2005).
Normalization of raw data
Appropriate choices for data normalization can help to accelerate
training and the identification of a good local minimum.
Categorical features such as DNA nucleotides first need to be
encoded numerically. They are typically represented as binary
vectors with all but one entry set to zero, which indicates the cate￾gory (one-hot coding). For example, DNA nucleotides (categories)
Table 1. Overview of existing deep learning frameworks, comparing four widely used software solutions.
Caffe Theano Torch7 TensorFlow
Core language C++ Python, C++ LuaJIT C++
Interfaces Python, Matlab Python C Python
Wrappers Lasagne, Keras, sklearn-theano Keras, Pretty Tensor, Scikit Flow
Programming paradigm Imperative Declarative Imperative Declarative
Well suited for CNNs, Reusing existing
models, Computer vision
Custom models, RNNs Custom models, CNNs,
Reusing existing models
Custom models, Parallelization,
RNNs
First layer features Third layer features
In top left? In top right? In bottom right? …
0.02 0.01 0.25
0.01 0.03 0.19
0.21 0.24 0.01
In left? In right? In bottom? …
0.03 0.01 0.02
0.02 0.01 0.01
2.51 0.02 2.92
Figure 4. A pre-trained network can be used as a generic feature extractor.
Feeding input into the first layer (left) gives a low-level feature representation in terms of patterns (left to right) present in smaller patches in every cell (top to bottom). Neuron
activations extracted from deeper layers (right) give rise to more abstract features that capture information from a larger segment of the image.
ª 2016 The Authors Molecular Systems Biology 12: 878 | 2016
Christof Angermueller et al Deep learning for computational biology Molecular Systems Biology
9
Downloaded from https://www.embopress.org on December 26, 2024 from IP 2a02:c7c:8e11:5e00:43c8:c5b2:93b4:fcbb.