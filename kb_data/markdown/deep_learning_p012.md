---
title: "deep_learning.pdf — page 12"
type: source-summary
source_path: "data/inputs/deep_learning.pdf"
page: 12
category: "docs"
tags: "test"
created: 2026-05-01 08:58:19.146640+00:00
---
## Content

predictions are averaged (Fig 5E). The dropout rate corresponds to
the probability that a neuron is dropped out, where 0.5 is a sensible
default value. In addition to dropping out hidden units, input units
can be dropped, however usually at a lower rate. Dropout is often
combined with regularizing the magnitude or parameter values by
the L2 norm, and less commonly the L1 norm.
Another popular regularization method is “early stopping”. Here,
training is stopped as soon as the validation performance starts to
saturate or deteriorate, and the parameters with the best perfor￾mance on the validation set chosen.
Layerwise pre-training (Bengio et al, 2007; Salakhutdinov &
Hinton, 2012) should be considered if the model overfits despite the
mentioned regularization techniques. Instead of training the entire
network at once, layers are first pre-trained unsupervised using
autoencoders or restricted Boltzmann machines. Afterwards, the
entire network is fine-tuned using the actual supervised learning
objective.
Hyper-parameter optimization
Table 2 summarizes recommendations and starting points for the
most common hyper-parameters, excluding architecture-dependent
hyper-parameters such as the size and number of filters of a
CNN. Since the best hyper-parameter configuration is data- and
application-dependent, models with different configurations
should be trained and their performance be evaluated on a valida￾tion set. As the number of configurations grows exponentially
with the number of hyper-parameters, trying all of them is impos￾sible in practice (Bengio, 2012). It is therefore recommended to
optimize the most important hyper-parameters such as the learn￾ing rate, batch size or length of convolutional filters indepen￾dently via line search, which is exploring different values while
keeping all other hyper-parameters constant. The refined hyper￾parameter space can then be further explored by random
sampling, and settings with the best performance on the
validation set are chosen. Frameworks such as Spearmint (Snoek
et al, 2012), Hyperopt (Bergstra & Cox, 2013) or SMAC (Hutter
et al, 2011) allow to automatically explore the hyper-parameter
space using Bayesian optimization. However, although conceptu￾ally more powerful, they are at present more difficult to apply
and parallelize than random sampling.
Training on GPUs
Training neural networks is more time-consuming compared to
shallow models and can take hours, days or even weeks, depending
on the size of training set and model architecture. Training on GPUs
can considerably reduce the training time (commonly by tenfold or
more) and is therefore crucial for evaluating multiple models effi￾ciently. The reason for this speedup is that learning deep networks
requires large numbers of matrix multiplications, which can be
parallelized efficiently on GPUs. All state-of-the-art deep learning
frameworks provide support to train models on either CPUs or GPUs
without requiring any knowledge about GPU programming. On
desktop machines, the local GPU card can often be used if the
framework supports the specific brand. Alternatively, commercial
providers provide GPU cloud compute clusters.
Pitfalls
No single method is universally applicable, and the choice of
whether and how to use deep learning approaches will be problem￾specific. Conventional analysis approaches will remain valid and
have advantages when data are scarce or if the aim is to assess
statistical significance, which is currently difficult using deep learn￾ing methods. Another limitation of deep learning is the increased
training complexity, which applies both to model design and the
required compute environment.
Conclusion
Deep learning methods are a powerful complement to classical
machine learning tools and other analysis strategies. Already, these
approaches have found use in a number of applications in computa￾tional biology, including regulatory genomics and image analysis.
The first publicly available software frameworks have helped to
reduce the overhead of model development and provided a rich,
accessible toolbox to practitioners. We expect that continued
improvement of software infrastructure will make deep learning
applicable to a growing range of biological problems.
Acknowledgements
OS and CA were funded by the European Molecular Biology Laboratory. TP was
supported by the European Regional Development Fund through the BioMedIT
project, and Estonian Research Council (IUT34-4). LP was supported by the
Wellcome Trust and Estonian Research Council (IUT34-4). OS was supported
by the European Research Council (agreement N635290).
Conflict of interest
The authors declare that they have no conflict of interest.
Table 2. Central parameters of a neural network and recommended
settings.
Name Range Default value
Learning rate 0.1, 0.01, 0.001,
0.0001
0.01
Batch size 64, 128, 256 128
Momentum rate 0.8, 0.9, 0.95 0.9
Weight initialization Normal, Uniform,
Glorot uniform
Glorot uniform
Per-parameter adaptive
learning rate methods
RMSprop, Adagrad,
Adadelta, Adam
Adam
Batch normalization Yes, no Yes
Learning rate decay None, linear,
exponential
Linear (rate 0.5)
Activation function Sigmoid, Tanh, ReLU,
Softmax
ReLU
Dropout rate 0.1, 0.25, 0.5, 0.75 0.5
L1,L2 regularization 0, 0.01, 0.001
Molecular Systems Biology 12: 878 | 2016 ª 2016 The Authors
Molecular Systems Biology Deep learning for computational biology Christof Angermueller et al
12
Downloaded from https://www.embopress.org on December 26, 2024 from IP 2a02:c7c:8e11:5e00:43c8:c5b2:93b4:fcbb.