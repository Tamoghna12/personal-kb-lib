---
title: "deep_learning.pdf — page 1"
type: source-summary
source_path: "data/inputs/deep_learning.pdf"
page: 1
category: "docs"
tags: "test"
created: 2026-05-01 08:58:16.502197+00:00
---
## Content

Review
Deep learning for computational biology
Christof Angermueller1,†
, Tanel Pärnamaa2,3,†
, Leopold Parts2,3,* & Oliver Stegle1,**
Abstract
Technological advances in genomics and imaging have led to an
explosion of molecular and cellular profiling data from large
numbers of samples. This rapid increase in biological data dimen￾sion and acquisition rate is challenging conventional analysis
strategies. Modern machine learning methods, such as deep learn￾ing, promise to leverage very large data sets for finding hidden
structure within them, and for making accurate predictions. In this
review, we discuss applications of this new breed of analysis
approaches in regulatory genomics and cellular imaging. We
provide background of what deep learning is, and the settings in
which it can be successfully applied to derive biological insights. In
addition to presenting specific applications and providing tips for
practical use, we also highlight possible pitfalls and limitations to
guide computational biologists when and how to make the most
use of this new technology.
Keywords cellular imaging; computational biology; deep learning; machine
learning; regulatory genomics
DOI 10.15252/msb.20156651 | Received 11 April 2016 | Revised 2 June 2016 |
Accepted 6 June 2016
Mol Syst Biol. (2016) 12: 878
Introduction
Machine learning methods are general-purpose approaches to learn
functional relationships from data without the need to define them a
priori (Hastie et al, 2005; Murphy, 2012; Michalski et al, 2013). In
computational biology, their appeal is the ability to derive predictive
models without a need for strong assumptions about underlying
mechanisms, which are frequently unknown or insufficiently
defined. As a case in point, the most accurate prediction of gene
expression levels is currently made from a broad set of epigenetic
features using sparse linear models (Karlic et al, 2010; Cheng et al,
2011) or random forests (Li et al, 2015); how the selected features
determine the transcript levels remains an active research topic.
Predictions in genomics (Libbrecht & Noble, 2015; Ma¨rtens et al,
2016), proteomics (Swan et al, 2013), metabolomics (Kell, 2005) or
sensitivity to compounds (Eduati et al, 2015) all rely on machine
learning approaches as a key ingredient.
Most of these applications can be described within the canonical
machine learning workflow, which involves four steps: data clean￾ing and pre-processing, feature extraction, model fitting and evalua￾tion (Fig 1A). It is customary to denote one data sample, including
all covariates and features as input x (usually a vector of numbers),
and label it with its response variable or output value y (usually a
single number) when available.
A supervised machine learning model aims to learn a function
f(x) = y from a list of training pairs (x1,y1), (x2,y2), ... for which data
are recorded (Fig 1B). One typical application in biology is to predict
the viability of a cancer cell line when exposed to a chosen drug
(Menden et al, 2013; Eduati et al, 2015). The input features (x) would
capture somatic sequence variants of the cell line, chemical make-up
of the drug and its concentration, which together with the measured
viability (output label y) can be used to train a support vector
machine, a random forest classifier or a related method (functional
relationship f). Given a new cell line (unlabelled data sample x*) in
the future, the learnt function predicts its survival (output label y*) by
calculating f(x*), even if f resembles more of a black box, and its inner
workings of why particular mutation combinations influence cell
growth are not easily interpreted. Both regression (where y is a real
number) and classification (where y is a categorical class label) can be
viewed in this way. As a counterpart, unsupervised machine learning
approaches aim to discover patterns from the data samples x them￾selves, without the need for output labels y. Methods such as cluster￾ing, principal component analysis and outlier detection are typical
examples of unsupervised models applied to biological data.
The inputs x, calculated from the raw data, represent what the
model “sees about the world”, and their choice is highly problem￾specific (Fig 1C). Deriving most informative features is essential for
performance, but the process can be labour-intensive and requires
domain knowledge. This bottleneck is especially limiting for high￾dimensional data; even computational feature selection methods do
not scale to assess the utility of the vast number of possible input
combinations. A major recent advance in machine learning is
automating this critical step by learning a suitable representation of
the data with deep artificial neural networks (Bengio et al, 2013;
LeCun et al, 2015; Schmidhuber, 2015) (Fig 1D). Briefly, a deep
neural network takes the raw data at the lowest (input) layer and
transforms them into increasingly abstract feature representations by
successively combining outputs from the preceding layer in a data￾driven manner, encapsulating highly complicated functions in the
1 European Molecular Biology Laboratory, European Bioinformatics Institute, Wellcome Trust Genome Campus, Hinxton, Cambridge, UK
2 Department of Computer Science, University of Tartu, Tartu, Estonia
3 Wellcome Trust Sanger Institute, Wellcome Trust Genome Campus, Hinxton, Cambridge, UK
*Corresponding author. Tel: +44 1223 834 244; E-mail: leopold.parts@sanger.ac.uk
**Corresponding author. Tel: +44 1223 494 101; E-mail: oliver.stegle@ebi.ac.uk †
These authors contributed equally to this work
ª 2016 The Authors. Published under the terms of the CC BY 4.0 license Molecular Systems Biology 12: 878 | 2016 1
Downloaded from https://www.embopress.org on December 26, 2024 from IP 2a02:c7c:8e11:5e00:43c8:c5b2:93b4:fcbb.

[Figure 1]: Check for updates