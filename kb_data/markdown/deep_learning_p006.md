---
title: "deep_learning.pdf — page 6"
type: source-summary
source_path: "data/inputs/deep_learning.pdf"
page: 6
category: "docs"
tags: "test"
created: 2026-05-01 08:58:19.109757+00:00
---
## Content

In silico prediction of mutation effects
An important application of deep neural networks trained on the
raw DNA sequence is to predict the effect of mutations in silico.
Such model-based assessments of the effect of sequence changes
complement methods based on QTL mapping, and can in particular
help to uncover regulatory effects of rare SNVs or to fine-map likely
causal genes. An intuitive approach for visualizing such predicted
regulatory effects is mutation maps (Alipanahi et al, 2015), whereby
the effect of all possible mutations for a given input sequence is
represented in a matrix view (Fig 2E). The authors could further
reliably identify deleterious SNVs by training an additional neural
network with predicted binding scores for a wild-type and mutant
sequence (Fig 2C).
Joint prediction of multiple traits and further extensions
Following their initial successes, convolutional architectures have
been extended and applied to a range of tasks in regulatory geno￾mics. For example, Zhou and Troyanskaya (2015) considered these
architectures to predict chromatin marks from DNA sequence. The
authors observed that the size of the input sequence window is a
major determinant of model performance, where larger windows
(now up to 1 kb) coupled with multiple convolutional layers
enabled capturing sequence features at different genomic length
scales. A second innovation was to use neural network architectures
with multiple output variables (so-called multitask neural networks)
to predict multiple chromatin states in parallel. Multitask architec￾tures allow learning shared features between outputs, thereby
improving generalization performance, and markedly reducing the
computational cost of model training compared to learning indepen￾dent models for each trait (Dahl et al, 2014).
In a similar vein, Kelley et al (2016) developed the open-source
deep learning framework Basset, to predict DNase I hypersensitivity
across multiple cell types and to quantify the effect of SNVs on chro￾matin accessibility. Again, the model improved prediction perfor￾mance compared to conventional methods and was able to retrieve
both known and novel sequence motifs that are associated with
DNase I hypersensitivity. A related architecture has also been
considered by Angermueller et al to predict DNA methylation states
in single-cell bisulphite sequencing studies (Angermueller et al,
2016). This approach combined convolutional architectures to
detect informative DNA sequence motifs with additional features
derived from neighbouring CpG sites, thereby accounting for methy￾lation context. Most recently, Koh, Pierson and Kundaje applied
CNNs to de-noise genomewide chromatin immunoprecipitation
followed by sequencing data in order to obtain a more accurate
prevalence estimate for different chromatin marks (Koh et al, 2016).
At present, CNNs are among the most widely used architectures
to extract features from fixed-size DNA sequence windows. However,
alternative architectures could also be considered. For example,
recurrent neural networks (RNNs) are suited to model sequential
data (Lipton, 2015) and have been applied for modelling natural
language and speech (Hinton et al, 2012; Graves et al, 2013;
Sutskever et al, 2014; Che et al, 2015; Deng & Togneri, 2015; Xiong
et al, 2016), protein sequences (Agathocleous et al, 2010; Sønderby
& Winther, 2014), clinical medical data (Che et al, 2015; Lipton et al,
2015) and to a limited extent DNA sequences (Xu et al, 2007; Lee
et al, 2015). RNNs are appealing for applications in regulatory geno￾mics, because they allow modelling sequences of variable length, and
to capture long-range interactions within the sequence and across
multiple outputs. However, at present, RNNs are more difficult to
train than CNNs, and additional work is needed to better understand
the settings where one should be preferred over the other.
Complementary to supervised methods, unsupervised deep learn￾ing architectures learn low-dimensional feature representations from
high-dimensional unlabelled data, similarly to classical principal
component analysis or factor analysis, but using a nonlinear model.
Examples of such approaches are stacked autoencoders (Vincent
et al, 2010), restricted Boltzmann machines and deep belief networks
(Hinton et al, 2006). The learnt features can be used to visualize data
or as input for classical supervised learning tasks. For example,
sparse autoencoders have been applied to classify cancer cases using
gene expression profiles (Fakoor et al, 2013) or to predict protein
backbones (Lyons et al, 2014). Restricted Boltzmann machines can
also be used for unsupervised pre-training of deep networks to subse￾quently train supervised models of protein secondary structures
(Spencer et al, 2015), disordered protein regions (Eickholt & Cheng,
2013) or amino acid contacts (Eickholt & Cheng, 2012). Skip-gram
neural networks have been applied to learn low-dimensional repre￾sentations of protein sequences and improve protein classification
(Asgari & Mofrad, 2015). In general, unsupervised models are a
powerful approach if large quantities of unlabelled data are available
to pre-train complex models. Once trained, these models can help to
improve performance on classification tasks, for which smaller
numbers of labelled examples are typically available.
Deep learning for biological image analysis
Historically, perhaps the most important successes of deep neural
networks have been in image analysis. Deep architectures trained
on millions of photographs can famously detect objects in pictures
better than humans do (He et al, 2015). All current state-of-the-art
models in image classification, object detection, image retrieval and
semantic segmentation make use of neural networks.
The convolutional neural network (Box 2) is the most common
network architecture for image analysis. Briefly, a CNN performs
pattern matching (convolution) and aggregation (pooling) opera￾tions (Box 2). At a pixel level, the convolution operation scans the
image with a given pattern and calculates the strength of the match
for every position. Pooling determines the presence of the pattern in
a region, for example by calculating the maximum pattern match in
smaller patches (max-pooling), thereby aggregating region informa￾tion into a single number. The successive application of convolution
and pooling operations is at the core of most network architectures
used in image analysis (Box 2).
First applications in computational biology—pixel￾level classification
The early applications of deep networks for biological images
focused on pixel-level tasks, with additional models building on the
network outputs. For example, Ning et al (2005) applied
Molecular Systems Biology 12: 878 | 2016 ª 2016 The Authors
Molecular Systems Biology Deep learning for computational biology Christof Angermueller et al
6
Downloaded from https://www.embopress.org on December 26, 2024 from IP 2a02:c7c:8e11:5e00:43c8:c5b2:93b4:fcbb.