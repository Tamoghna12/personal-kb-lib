---
title: "deep_learning.pdf — page 3"
type: source-summary
source_path: "data/inputs/deep_learning.pdf"
page: 3
category: "docs"
tags: "test"
created: 2026-05-01 08:58:19.091174+00:00
---
## Content

of abstraction between the effect of individual DNA variants and the
trait of interest, as well as the dependence of the molecular traits on
a broad sequence context and interactions with distal regulatory
elements.
The value of deep neural networks in this context is twofold.
First, classical machine learning methods cannot operate on the
sequence directly, and thus require pre-defining features that can be
extracted from the sequence based on prior knowledge (e.g. the
presence or absence of single-nucleotide variants (SNVs), k-mer
frequencies, motif occurrences, conservation, known regulatory
variants or structural elements). Deep neural networks can help
circumventing the manual extraction of features by learning them
from data. Second, because of their representational richness, they
can capture nonlinear dependencies in the sequence and interaction
effects and span wider sequence context at multiple genomic scales.
Attesting to their utility, deep neural networks have been success￾fully applied to predict splicing activity (Leung et al, 2014; Xiong
et al, 2015), specificities of DNA- and RNA-binding proteins
Box 1: Artificial Neural Network
An artificial neural network, initially inspired by neural networks in the brain (McCulloch & Pitts, 1943; Farley & Clark, 1954; Rosenblatt, 1958), consists of
layers of interconnected compute units (neurons). The depth of a neural network corresponds to the number of hidden layers, and the width to the
maximum number of neurons in one of its layers. As it became possible to train networks with larger numbers of hidden layers, artificial neural networks
were rebranded to “deep networks”.
In the canonical configuration, the network receives data in an input layer, which are then transformed in a nonlinear way through multiple hidden
layers, before final outputs are computed in the output layer (panel A). Neurons in a hidden or output layer are connected to all neurons of the previous
layer. Each neuron computes a weighted sum of its inputs and applies a nonlinear activation function to calculate its output f(x) (panel B). The most
popular activation function is the rectified linear unit (ReLU; panel B) that thresholds negative signals to 0 and passes through positive signal. This type
of activation function allows faster learning compared to alternatives (e.g. sigmoid or tanh unit) (Glorot et al, 2011).
The weights w(i) between neurons are free parameters that capture the model’s representation of the data and are learned from input/output samples.
Learning minimizes a loss function L(w) that measures the fit of the model output to the true label of a sample (panel A, bottom). This minimization is
challenging, since the loss function is high-dimensional and non-convex, similar to a landscape with many hills and valleys (panel C). It took several
decades before the backward propagation algorithm was first applied to compute a loss function gradient via chain rule for derivatives (Rumelhart et al,
1988), ultimately enabling efficient training of neural networks using stochastic gradient descent. During learning, the predicted label is compared with
the true label to compute a loss for the current set of model weights. The loss is then backward propagated through the network to compute the gradi￾ents of the loss function and update (panel A). The loss function L(w) is typically optimized using gradient-based descent. In each step, the current
weight vector (red dot) is moved along the direction of steepest descent dw (direction arrow) by learning rate g (length of vector). Decaying the learning
rate over time allows to explore different domains of the loss function by jumping over valleys at the beginning of the training (left side) and fine-tune
parameters with smaller learning rates in later stages of the model training. While learning in deep neural networks remains an active area of research,
existing software packages (Table 1) can already be applied without knowledge of the mathematical details involved.
Alternative architectures to such fully connected feedforward networks have been developed for specific applications, which differ in the way neurons
are arranged. These include convolutional neural networks, which are widely used for modelling images (Box 2), recurrent neural networks for sequential
data (Sutskever, 2013; Lipton, 2015), or restricted Boltzmann machines (Salakhutdinov & Larochelle, 2010; Hinton, 2012) and autoencoders (Hinton &
Salakhutdinov, 2006; Alain et al, 2012; Kingma & Welling, 2013) for unsupervised learning. The choice of network architecture and other parameters can
be made in a data-driven and objective way by assessing the model performance on a validation data set.
0.8
Input
layer
Hidden
layer
Output
layer
1
0.6 
0.4 
0.2 
1×0.6+
0×0.4+
1×0.2= 0.8
 
max(0, 0.8)
Weighted
sum
 Activation
function 
Inputs Output
0.8
ReLU
PREDICTED
label
TRUE
label
FORWARD PROPAGATION 
BACKWARD PROPAGATION LOSS 
0
1
Local
optimum
Global
optimum
A B 
C 
w(1)
0.8 
w(2)
1 
0 
1 
0.3 
0.0 
0 
0.7 
f(x) = 0.7
L(w) = (0.7– 0.8) 2
= y = 0.8
L(w)
Σ
w w'= w + ηΔw ηΔw
ª 2016 The Authors Molecular Systems Biology 12: 878 | 2016
Christof Angermueller et al Deep learning for computational biology Molecular Systems Biology
3
Downloaded from https://www.embopress.org on December 26, 2024 from IP 2a02:c7c:8e11:5e00:43c8:c5b2:93b4:fcbb.