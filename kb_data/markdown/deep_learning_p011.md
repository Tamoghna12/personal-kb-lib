---
title: "deep_learning.pdf — page 11"
type: source-summary
source_path: "data/inputs/deep_learning.pdf"
page: 11
category: "docs"
tags: "test"
created: 2026-05-01 08:58:19.140405+00:00
---
## Content

The most common objective functions are the cross-entropy for clas￾sification and mean-squared error for regression. Minimizing L(w)
is challenging since it is high-dimensional and non-convex (Fig 5C);
see also Box 1 and Fig 2.
Stochastic gradient descent
Stochastic gradient descent is widely used to train deep models.
Starting from an initial set of parameters w0, the gradient dw of L
with respect to w is computed for a random batch of only few, for
example 128, training samples. dw points to the direction of steepest
descent, towards which w is updated with step size eta, the learning
rate (Fig 1C). At each step, the parameters are updated into the
direction of steepest descent until a minimum is reached, analo￾gously to a ball running down a hill to a valley (Bengio, 2012). The
training performance strongly depends on parameter initialization,
learning rate and batch size.
Parameter initialization
In general, model parameters should be initialized randomly to
avoid local optima determined by a fixed initialization. Starting
points for model parameters can be sampled independently from a
normal distribution with small variance, or more commonly from a
normal distribution with its variance scaled inversely by the number
of hidden units in the input layer (Glorot & Bengio, 2010; He et al,
2015).
Learning rate and batch size
The learning rate and batch size of stochastic gradient descent need
to be chosen with care, since they can strongly impact training
speed and model performance. Different learning rates are usually
explored on a logarithmic scale such as 0.1, 0.01 or 0.001, with 0.01
as the recommended default value (Bengio, 2012). A batch size of
128 training samples is suitable for most applications. The batch
size can be increased to speed up training or decreased to reduce
memory usage, which can be important for training complex models
on memory-limited GPUs. The optimum learning rate and batch size
are connected, with larger batch sizes typically requiring smaller
learning rates.
Learning rate decay
The learning rate can be gradually reduced during training, which is
based on the idea that larger steps may be helpful in early training
stages in order to overcome possible local optima, whereas smaller
step sizes allow exploring narrow parameter regions of the loss
function in advanced stages of training. Common approaches
include to linearly reduce the learning rate by a constant factor such
as 0.5 after the validation loss stops improving, or exponentially
after every training iteration or epoch (Bengio, 2012; Gawehn et al,
2016).
Momentum
Vanilla stochastic gradient descent can be extended by “momen￾tum”, which usually improves training (Sutskever et al, 2013).
Instead of updating the current parameter vector wt at time t by the
gradient vector dwt+1 directly, a fraction of the previous update is
added to the current one. With momentum rate v, weights are
updated by a momentum vector mt+1 = m  mt - 2 dWt+1. This
approach can help to take larger steps in directions where gradients
point consistently, and therefore speed up the convergence. The
momentum rate v can be set between [0, 1], and a typical value
is 0.9. Nesterov momentum (Nesterov, 1983, 2013) is a special
form of the same concept, which sometimes provides additional
advantages.
Per-parameter adaptive learning rate methods
To reduce the sensitivity to the specific choice of the learning rate,
adaptive learning rate methods, such as RMSprop, Adagrad
(Srivastava et al, 2014) and Adam (Kingma & Ba, 2014), have been
developed in order to appropriately adapt the learning rate per
parameter during training. The most recent method, Adam, combi￾nes the strengths of previous methods RMSprop and Adagrad and is
generally recommended for many applications.
Batch normalization
Batch normalization (Ioffe & Szegedy, 2015) is a recently described
approach to reduce the dependency of training to the parameter
initialization, speed up training and reduce overfitting. It is easy to
implement, has marginal additional compute costs and has hence
become common practice. Batch normalization zero centres and
normalizes data not only at the input layer, but also at hidden layers
before the activation function. This approach allows using higher
learning rates and hence also accelerates training.
Analysing the learning curve
To validate the learning process, the loss should be monitored as a
function of the number of training epochs, that is the number of
times the full training set has been traversed (Fig 5D). If the learn￾ing curve decreases slowly, the learning rate may be too small and
should be increased. If the loss decreases steeply at the beginning
but saturates quickly, the learning rate may be too high. Extreme
learning rates can result in an increasing or fluctuating learning
curve (Bengio, 2012).
Monitoring training and validation performance
In parallel with the training loss, it is recommended to monitor the
target performance such as the accuracy for both the training and
validation set during training (Fig 5E). A low or decreasing valida￾tion performance relative to the training performance indicates over￾fitting (Bengio, 2012).
Avoiding overfitting
Deep neural networks are notoriously difficult to train, and overfit￾ting to data is a major challenge, since they are nonlinear and have
many parameters. Overfitting results from a too complex model
relative to the size of the training set, and can thus be reduced by
decreasing the model complexity, for example the number of hidden
layers and units, or by increasing the size of the training set, for
example via data augmentation. The following training guidelines
can help to avoid overfitting.
Dropout (Srivastava et al, 2014) is the most common regulariza￾tion technique and often one of the key ingredients to train deep
models. Here, the activation of some neurons is randomly set to
zero (“dropped out”) during training in each forward pass, which
intuitively results in an ensemble of different networks whose
ª 2016 The Authors Molecular Systems Biology 12: 878 | 2016
Christof Angermueller et al Deep learning for computational biology Molecular Systems Biology
11
Downloaded from https://www.embopress.org on December 26, 2024 from IP 2a02:c7c:8e11:5e00:43c8:c5b2:93b4:fcbb.