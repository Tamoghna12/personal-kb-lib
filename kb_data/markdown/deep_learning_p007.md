---
title: "deep_learning.pdf — page 7"
type: source-summary
source_path: "data/inputs/deep_learning.pdf"
page: 7
category: "docs"
tags: "test"
created: 2026-05-01 08:58:19.117445+00:00
---
## Content

convolutional neural networks in a study that predicted abnormal
development in C. elegans embryo images. They trained a CNN on
40 × 40 pixel patches to classify the centre pixel to cell wall, cyto￾plasm, nucleus membrane, nucleus or outside medium, using three
convolutional and pooling layers, followed by a fully connected
output layer. The model predictions were then fed into an energy￾based model for further analysis. CNNs have outperformed standard
methods, for example Markov random fields and conditional
random fields (Li, 2009) in such raw data analysis tasks, for exam￾ple restoring noisy neural circuitry images (Jain et al, 2007).
Adding layers allows moving from clearing up pixel noise to
modelling more abstract image features. Ciresan et al (2013) used
five convolutional and pooling layers, followed by two fully
connected layers, to find mitosis in breast histology images. This
model won the mitosis detection challenge at the International
Conference of Pattern Recognition 2012, outperforming competitors
by a substantial margin. The same approach was also used to
segment neuronal structures in electron microscopy images, classi￾fying each pixel as membrane or non-membrane (Ciresan et al,
2012). In these applications, while the CNNs were trained in an
end-to-end manner, additional post-processing was required to
obtain class probabilities from the outputs for new images.
Successive pooling operations lose information on localization,
as only summaries are retained from larger and larger regions. To
avoid this, skip links can be added to carry information from early,
fine-grained layers forward to deeper ones. The currently best￾performing pixel-level classification method for neuronal structures
(U-Net; Ronneberger et al, 2015) employs an architecture in which
neurons take inputs from lower layers to localize high-resolution
features, as well as to overcome the arbitrary choice of context size.
Analysis of whole cells, cell populations and tissues
In many cases, pixel-level predictions are not required. For example,
Xu et al directly classified colon histopathology images into cancer￾ous and non-cancerous, finding that supervised feature learning
with deep networks was superior to using handcrafted features (Xu
et al, 2014). Pa¨rnamaa and Parts used CNNs to classify pre￾segmented image patches of individual yeast cells carrying a fluores￾cent protein to different subcellular localization patterns (Pa¨rnamaa
& Parts, 2016). Again, deep networks outperformed methods based
on traditional features. Further, Kraus et al combined the segmenta￾tion and classification tasks into a single architecture that can be
learned end-to-end and applied the model to full resolution yeast
microscopy images (Kraus et al, 2015). This approach allowed clas￾sifying entire images without performing segmentation as a pre￾processing step. CNNs have even been applied to count bacterial
colonies in agar plates (Ferrari et al, 2015). Since the early de￾noising applications on the pixel level, the field has been moving
towards end-to-end image analysis pipelines that make use of large
bioimage data sets, and the representational power of CNNs.
Reusing trained models
Training convolutional neural networks requires large data sets.
While biological data acquisition can be expensive, this does not
mean that deep neural networks cannot be used when millions of
images are not available. Regardless of image source, lower levels
of the network tend to capture similar signal (edges, blobs) that are
not specific to the training data and the application, but instead
recur in perceptual tasks in general. Thus, convolutional neural
networks can reuse pictures from a similar domain to help with
learning, or even be pre-trained on other data, thereby requiring
fewer images to fine-tune the model for the task of interest. Indeed,
Donahue et al (2013) and Razavian et al (2014) showed that
features learned from millions of images to classify objects, can
successfully be used in image retrieval, detection or classification on
new domains where only hundreds of images are labelled. The
effectiveness of such an approach depends on the similarity between
the training data and the new domain (Yosinski et al, 2014).
The concept of transferring model parameters has also been
successful in bioimage analysis. For example, Zhang et al (2015)
showed that features learned from natural images can be transferred
to biological data, improving the prediction of Drosophila melanoga￾ster developmental stages from in situ hybridization images. The
model was first pre-trained on data from the ImageNet
(Russakovsky et al, 2015), an open corpus of more than one million
diverse images, to extract rich features at different scales. Xie et al
(2015) further used synthetic images to train a CNN for automatic
cell counting in microscopy images. We expect that network reposi￾tories that host pre-trained models will emerge for biological image
analysis; such efforts already exist for general image processing
tasks (see learning section below). These trained models could be
downloaded and used as feature extractors (Fig 3), or further fine￾tuned and adapted to a particular task on small-scale data.
Interpreting and visualizing convolutional networks
Convolutional neural networks have been successful across many
domains. In interpreting their performance, it is useful to under￾stand the features they capture.
Visualizing input weights
One way to understand what a particular neuron represents is to
look for inputs that maximally activate it. Under some mathematical
constraints, these patterns are proportional to the incoming weights
(see also Box 1). Krizhevsky et al visualized weights in the first
convolutional layer (Krizhevsky et al, 2012) and found that these
maximally activating patterns correspond to colour blobs, edges at
different orientations and Gabor-like filters (Fig 4). Gabor filters are
widely used pre-defined features in image analysis; neural networks
rediscover them in a data-driven way as a useful component of the
image model. Higher layer weights can be visualized as well, but as
the inputs are not pixels, their weights are more difficult to interpret.
Finding images that maximize neuron activity
To understand the deeper layers in terms of input pixels, Girshick
et al (2014) retrieved and Simonyan et al (2013) generated images
that maximize the output of individual neurons (Fig 4). While this
approach yields no explicit representation, it can provide an over￾view of the type of features that differentiate images with large
neuron activity from all others. Such visualizations tend to show
that second-layer features combine edges from the first layer,
ª 2016 The Authors Molecular Systems Biology 12: 878 | 2016
Christof Angermueller et al Deep learning for computational biology Molecular Systems Biology
7
Downloaded from https://www.embopress.org on December 26, 2024 from IP 2a02:c7c:8e11:5e00:43c8:c5b2:93b4:fcbb.