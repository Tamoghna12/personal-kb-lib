---
title: "deep_learning.pdf — page 5"
type: source-summary
source_path: "data/inputs/deep_learning.pdf"
page: 5
category: "docs"
tags: "test"
created: 2026-05-01 08:58:19.103257+00:00
---
## Content

Their DeepBind model outperformed existing methods, was able to
recover known and novel sequence motifs, and could quantify the
effect of sequence alterations and identify functional SNVs. A key
innovation that enabled training the model directly on the raw
DNA sequence was the application of a one-dimensional convolu￾tional layer. Intuitively, the neurons in the convolutional layer
scan for motif sequences and combinations thereof, similar to
conventional position weight matrices (Stormo et al, 1982). The
learning signal from deeper layers informs the convolutional layer
which motifs are the most relevant. The motifs recovered by the
model can then be visualized as heatmaps or sequence logos
(Fig 2D).
Box 2: Convolutional Neural Network
Convolutional neural networks (CNNs) were originally inspired by cognitive neuroscience and Hubel and Wiesel’s seminal work on the cat’s visual cortex,
which was found to have simple neurons that respond to small motifs in the visual field, and complex neurons that respond to larger ones (Hubel &
Wiesel, 1963, 1970).
CNNs are designed to model input data in the form of multidimensional arrays, such as two-dimensional images with three colour channels (LeCun
et al, 1989; Jarrett et al, 2009; Krizhevsky et al, 2012; Zeiler & Fergus, 2014; He et al, 2015; Szegedy et al, 2015a) or one-dimensional genomic sequences
with one channel per nucleotide (Alipanahi et al, 2015; Wang et al, 2015; Zhou & Troyanskaya, 2015; Angermueller et al, 2016; Kelley et al, 2016). The
high dimensionality of these data (up to millions of pixels for high-resolution images) renders training a fully connected neural network challenging, as
the number of parameters of such a model would typically exceed the number of training data to fit them. To circumvent this, CNNs make additional
assumptions on the structure of the network, thereby reducing the effective number of parameters to learn.
A convolutional layer consists of multiple maps of neurons, so-called feature maps or filters, with their size being equal to the dimension of the input
image (panel A). Two concepts allow reducing the number of model parameters: local connectivity and parameter sharing. First, unlike in a fully
connected network, each neuron within a feature map is only connected to a local patch of neurons in the previous layer, the so-called receptive field.
Second, all neurons within a given feature map share the same parameters. Hence, all neurons within a feature map scan for the same feature in the
previous layer, however at different locations. Different feature maps might, for example, detect edges of different orientation in an image, or sequence
motifs in a genomic sequence. The activity of a neuron is obtained by computing a discrete convolution of its receptive field, that is computing the
weighted sum of input neurons, and applying an activation function (panel B).
In most applications, the exact position and frequency of features is irrelevant for the final prediction, such as recognizing objects in an image. Using this
assumption, the pooling layer summarizes adjacent neurons by computing, for example, the maximum or average over their activity, resulting in a
smoother representation of feature activities (panel C). By applying the same pooling operation to small image patches that are shifted by more than
one pixel, the input image is effectively down-sampled, thereby further reducing the number of model parameters.
A CNN typically consists of multiple convolutional and pooling layers, which allows learning more and more abstract features at increasing scales from
small edges, to object parts, and finally entire objects. One or more fully connected layers can follow the last pooling layer (panel A). Model hyper-para￾meters such as the number of convolutional layers, number of feature maps or the size of receptive fields are application-dependent and should be
strictly selected on a validation data set.
0.01 Cytoplasm
0.96 Cell periphery
0.01 Vacuole
…
Input image Convolutional layer Pooling layer 
1×1+
2×0+
1×0+
2×1=3
max({1, 2, 1, 2})= 2 
Discrete convolution Max pooling 
Fully connected
layers
Output layer 
Max
pooling 
Discrete
convolution
× N 
Receptive
field 
Feature maps 
A 
B C 
…
ª 2016 The Authors Molecular Systems Biology 12: 878 | 2016
Christof Angermueller et al Deep learning for computational biology Molecular Systems Biology
5
Downloaded from https://www.embopress.org on December 26, 2024 from IP 2a02:c7c:8e11:5e00:43c8:c5b2:93b4:fcbb.

[Figure 1]: This image is a simple text or graphic representation focused on a circle with a gradient from lighter to darker. The diagram is empty with no axes or labels. It appears to be an abstract representation, likely a flowchart or a data visualization.

[Figure 2]: Image content: A single black circle with no other visible details.

[Figure 3]: This image is a simple line graph. The x-axis is labeled 'time' (in hours), the y-axis is labeled 'accuracy' (percentage), and there are no grid lines. The data points are scattered randomly, indicating no clear trend or pattern. The key quantitative finding is the accuracy reaching 94.3% at epoch 50.