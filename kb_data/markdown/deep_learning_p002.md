---
title: "deep_learning.pdf — page 2"
type: source-summary
source_path: "data/inputs/deep_learning.pdf"
page: 2
category: "docs"
tags: "test"
created: 2026-05-01 08:58:19.082415+00:00
---
## Content

process (Box 1). Deep learning is now one of the most active fields in
machine learning and has been shown to improve performance
in image and speech recognition (Hinton et al, 2012; Krizhevsky et al,
2012; Graves et al, 2013; Zeiler & Fergus, 2014; Deng & Togneri, 2015),
natural language understanding (Bahdanau et al, 2014; Sutskever
et al, 2014; Lipton, 2015; Xiong et al, 2016), and most recently, in
computational biology (Eickholt & Cheng, 2013; Dahl et al, 2014;
Leung et al, 2014; Sønderby & Winther, 2014; Alipanahi et al, 2015;
Wang et al, 2015; Zhou & Troyanskaya, 2015; Kelley et al, 2016).
The potential of deep learning in high-throughput biology is
clear: in principle, it allows to better exploit the availability of
increasingly large and high-dimensional data sets (e.g. from DNA
sequencing, RNA measurements, flow cytometry or automated
microscopy) by training complex networks with multiple layers that
capture their internal structure (Fig 1C and D). The learned
networks discover high-level features, improve performance over
traditional models, increase interpretability and provide additional
understanding about the structure of the biological data.
In this review, we discuss recent and forthcoming applications of
deep learning, with a focus on applications in regulatory genomics
and biological image analysis. The goal of this review was not to
provide comprehensive background on all technical details, which
can be found in the more specialized literature (Bengio, 2012;
Bengio et al, 2013; Deng, 2014; Schmidhuber, 2015; Goodfellow
et al, 2016). Instead, we aimed to provide practical pointers and the
necessary background to get started with deep architectures, review
current software solutions and give recommendations for applying
them to data. The applications we cover are deliberately broad to
illustrate differences and commonalities between approaches;
reviews focusing on specific domains can be found elsewhere (Park
& Kellis, 2015; Gawehn et al, 2016; Leung et al, 2016; Mamoshina
et al, 2016). Finally, we discuss both the potential and possible
pitfalls of deep learning and contrast these methods to traditional
machine learning and classical statistical analysis approaches.
Deep learning for regulatory genomics
Conventional approaches for regulatory genomics relate sequence
variation to changes in molecular traits. One approach is to leverage
variation between genetically diverse individuals to map quantitative
trait loci (QTL). This principle has been applied to identify regulatory
variants that affect gene expression levels (Montgomery et al, 2010;
Pickrell et al, 2010), DNA methylation (Gibbs et al, 2010; Bell et al,
2011), histone marks (Grubert et al, 2015; Waszak et al, 2015) and
proteome variation (Vincent et al, 2010; Albert et al, 2014; Parts et al,
2014; Battle et al, 2015) (Fig 2A). Better statistical methods have
helped to increase the power to detect regulatory QTL (Kang et al,
2008; Stegle et al, 2010; Parts et al, 2011; Rakitsch & Stegle, 2016);
however, any mapping approach is intrinsically limited to variation that
is present in the training population. Thus, studying the effects of rare
mutations in particular requires data sets with very large sample size.
An alternative is to train models that use variation between
regions within a genome (Fig 2A). Splitting the sequence into
windows centred on the trait of interest gives rise to tens of thou￾sands of training examples for most molecular traits even when
using a single individual. Even with large data sets, predicting molec￾ular traits from DNA sequence is challenging due to multiple layers
x y 
Clean data Features Model Results 
A 
D 
Feature
extraction 
Discriminative features 
Raw data 
Label 
C 
Intron
Exon
Feature
extraction Training Evaluation 
Supervised Unsupervised 
x
• Linear regression
• Logistic regression
• Random Forest
• SVM
• … 
• PCA
• Factor analysis
• Clustering
• Outlier detection
• …
 
B 
A C G T C 
G C G T A 
G T C C G 
T T A G T 
C G T A G 
G A G A A 
GA
T
CCG
T
CA
GC
G
A
TC
G
A
TC
G
A
T
C
GA
T
C
GT
C
G
T
C
G
TA
TC
G
A
T
CG
TCG
A
G
AT
CCAC
ACC
GGA
TG
T
G
TC
G
A
T C
AT
C
A
GC
ATGT
CCG
ATC
G
T
G
AC
GTCAAT
CT
G
CG
T
G
A
TC
G
ACG
A
CA
GCA
GCG
C
GT
C
G
A
TC
T
C
G
A
T
C
C
GA
C
G
AT
GA
T
CCAT
CGA
C
CCGA
CTG
T
CA
C
G
T
CAT
GCA
T
GC
G
AT
C
GA
TG
CCGA
GA
TG
CGA
CCA
T
GCA
T
GCA
T
G
GA
T
CC
G
A
TCG
A
TGT
AAT
GCA
GC
GC
GA
T
CA
TGA
TC
GA
TAT
G
CG
T
A
CG
T
A
CG
TCG
AC
A
TCG
A
T
G
A
TC
T
AATG
A
TC
T
ACG
T
A
Layer 1 G
A
TC
G
ACG
A
CA
GCA
GCG
C
GT
C
G
A
TC
T
C
G
A
T
CG
T
A
CG
T
A
CG
TCG
AC
A
TCG
A
T
G
A
TC
T
AATG
A
TC
T
ACG
T
A
Raw data 
Pre￾processing 
Raw data 
Layer 2 TSS Intron Exon
Figure 1. Machine learning and representation learning.
(A) The classical machine learning workflow can be broken down into four steps: data pre-processing, feature extraction, model learning and model evaluation. (B) Supervised
machine learning methods relate input features x to an output label y, whereas unsupervised method learns factors about x without observed labels. (C) Raw input data are
often high-dimensional and related to the corresponding label in a complicated way, which is challenging for many classical machine learning algorithms (left plot).
Alternatively, higher-level features extracted using a deep model may be able to better discriminate between classes (right plot). (D) Deep networks use a hierarchical
structure to learn increasingly abstract feature representations from the raw data.
Molecular Systems Biology 12: 878 | 2016 ª 2016 The Authors
Molecular Systems Biology Deep learning for computational biology Christof Angermueller et al
2
Downloaded from https://www.embopress.org on December 26, 2024 from IP 2a02:c7c:8e11:5e00:43c8:c5b2:93b4:fcbb.