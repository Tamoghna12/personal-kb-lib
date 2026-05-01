---
title: "deep_learning.pdf — page 8"
type: source-summary
source_path: "data/inputs/deep_learning.pdf"
page: 8
category: "docs"
tags: "test"
created: 2026-05-01 08:58:19.124313+00:00
---
## Content

thereby detecting corners and angles; deeper layer neurons activate
for specific object parts (e.g. noses, eyes); and the deepest layers
detect whole objects (e.g. faces, cars). It is complicated to hand￾engineer features that look specifically for noses, eyes or faces, but
neural networks can learn these features solely from input–output
examples.
Hiding important image parts
To understand which image parts are important for determining the
value of each feature, Zeiler and Fergus (2014) occluded images
with smaller grey boxes. The parts that are most influential will
drastically change the feature value when occluded. In a similar
vein, Simonyan et al (2013) and Springenberg et al (2014) visual￾ized which individual pixels make the most difference in the feature,
and Bach, Binder and colleagues developed pixel relevance for indi￾vidual classification decisions in a more general framework (Bach
et al, 2015). This information can also be used for object localiza￾tion or segmentation, as the sensitive image pixels usually correctly
correspond to the true object. Kraus et al (2015) used this idea to
effectively localize cells in large microscopy images.
Visualizing similar inputs in two dimensions
Visualizing the CNN representations can help gauge what inputs get
mapped to similar feature vectors, and hence understand what the
model has learned. Donahue et al (2013) projected CNN features
into two dimensions to show that each subsequent layer transforms
data to be more and more separable by a linear classifier. In general,
different CNN visualization methods show that higher layer features
are more specific to the learning task, while low-level features tend
to capture general aspects of images, such as edges and corners.
Off-the-shelf tools and practical considerations
Deep learning frameworks
Deep learning frameworks have been developed to easily build
neural networks from existing modules on a high level. The most
popular ones are Caffe (Jia et al, 2014), Theano (Bastien et al, 2012),
Torch7 (Collobert et al, 2011) and TensorFlow (Abadi et al, 2016;
Rampasek & Goldenberg, 2016) (Table 1), which differ in modular￾ity, ease of use and the way models are defined and trained.
Caffe (Jia et al, 2014) is developed by the Berkeley Vision and
Learning Center and is written in C++. The network architecture is
specified in a configuration file and models can be trained and used
via command line, without writing code at all. Additionally, Python
and MATLAB interfaces are available. Caffe offers one of the most
efficient implementations for CNNs and provides multiple pre￾trained models for image recognition, making it well suited for
computer vision tasks. As a downside, custom models need to be
implemented in C++, which can be difficult. Additionally, Caffe is
not optimized for recurrent architectures.
Theano (Bastien et al, 2012; Team et al, 2016) is developed and
maintained by the University of Montreal and written in Python and
C++. Model definitions follow a declarative instead of an imperative
programing paradigm, which means that the user specifies what
needs to be done, not in which order. A neural network is declared
as a computational graph, which is then compiled to native code
and executed. This design allows Theano to optimize computational
steps and to automatically derive gradients—one of its main
strengths. Consequently, Theano is well suited for building custom
models and offers particularly efficient implementations for RNNs.
Software wrappers such as Keras (https://github.com/fchollet/
keras) or Lasagne (https://github.com/Lasagne/Lasagne) provide
additional abstraction and allow building networks from existing
components, and reusing pre-trained networks. The major draw￾back of Theano is frequently long compile times when building
larger models.
Torch7 (Collobert et al, 2011) was initially developed at the
University of New York and is based on the scripting language
LuaJIT. Networks can be easily built by stacking existing modules
and are not compiled, hence making it more suited for fast prototyp￾ing than Theano. Torch7 offers an efficient CNN implementation and
access to a range of pre-trained models. A possible downside is the
need of the user to be familiar with the LuaJIT scripting language.
Also, LuaJIT is less suited for building custom recurrent networks.
TensorFlow (Abadi et al, 2016) is the most recent deep learning
framework developed by Google. The software is written in C++ and
offers interfaces to Python. Similar to Theano, a neural network is
declared as a computational graph, which is optimized during
compilation. However, the shorter compile time makes it more
suited for prototyping. A key strength of TensorFlow is native
support for parallelization across different devices, including CPUs
and GPUs, and using multiple compute nodes on a cluster. The
accompanying tool TensorBoard allows to conveniently visualize
networks in a web browser and to monitor training progress, for
example learning curves or parameter updates. At present, Fully connected
Conv
Pool
Conv
Pool
Conv
Pool
…
0.01 Vacuole
0.01 Cytoplasm
0.96 Cell periphery
Figure 3. Convolution and pooling operators are stacked, thereby creating a deep network for image analysis.
In standard applications, convolution layers are followed by a pooling layer (Box 2). In this example, the lowest level convolutional units operate on 3 × 3 patches, but deeper
ones use and capture information from larger regions. These convolutional pattern-matching layers are followed by one or multiple fully connected layers to learn
which features are most informative for classification. For each layer with learnable weights, three example images that maximize some neuron output are shown.
Molecular Systems Biology 12: 878 | 2016 ª 2016 The Authors
Molecular Systems Biology Deep learning for computational biology Christof Angermueller et al
8
Downloaded from https://www.embopress.org on December 26, 2024 from IP 2a02:c7c:8e11:5e00:43c8:c5b2:93b4:fcbb.

[Figure 1]: This image is a plain yellow circle with no additional text or graphics.