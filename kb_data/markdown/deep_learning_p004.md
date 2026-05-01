---
title: "deep_learning.pdf — page 4"
type: source-summary
source_path: "data/inputs/deep_learning.pdf"
page: 4
category: "docs"
tags: "test"
created: 2026-05-01 08:58:19.097395+00:00
---
## Content

(Alipanahi et al, 2015) or epigenetic marks and to study the effect of
DNA sequence alterations (Zhou & Troyanskaya, 2015; Kelley et al,
2016).
Early applications of neural networks in
regulatory genomics
The first successful applications of neural networks in regulatory
genomics replaced a classical machine learning approach with a
deep model, without changing the input features. For example,
Xiong et al (2015) considered a fully connected feedforward neural
network to predict the splicing activity of individual exons. The
model was trained using more than 1,000 pre-defined features
extracted from the candidate exon and adjacent introns. Despite the
relatively low number of 10,700 training samples in combination
with the model complexity, this method achieved substantially
higher prediction accuracy of splicing activity compared to simpler
approaches, and in particular was able to identify rare mutations
implicated in splicing misregulation.
Convolutional designs
More recent work using convolutional neural networks (CNNs)
allowed direct training on the DNA sequence, without the need to
define features (Alipanahi et al, 2015; Zhou & Troyanskaya, 2015;
Angermueller et al, 2016; Kelley et al, 2016). The CNN architecture
allows to greatly reduce the number of model parameters compared
to a fully connected network by applying convolutional operations
to only small regions of the input space and by sharing parameters
between regions. The key advantage resulting from this approach is
the ability to directly train the model on larger sequence windows
(Box 2; Fig 2B).
Alipanahi et al (2015) considered convolutional network archi￾tectures to predict specificities of DNA- and RNA-binding proteins.
C
G
TC
G
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
Pooling +
Convolution 
Convolution 
Individual 1
C GCCTGCCG… G C C T G C C G … 
 …ATGCTATA … A T G C T A T A G GCCTGCCG… G C C T G C C G … 
 …ATGCTATA … A T G C T A T A C GCCTGCCG… G C C T G C C G … 
 …ATGCTATA … A T G C T A T A C GCCTGCCG… 
Between￾individual
variation
ANOVA
eQTL 
Within-individual
variation
Locus 1 Locus 2 Locus 3
Fully
connected
layers 
Output
layer
2nd convolution
layer 
1st convolution
layer 
Input
sequence 
Variant
score
G
A
TC
G
A
C
G
A
CA
GCA
GC
G
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
Active 
Normal 0.9 
Deleterious 
A B 
C 
G
A
TC
G
A
CG
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
D 
E 
Wild type
Individual 2
Individual 3 
Mutant
Wild type
response 
A T G G C G C G C T G
T C A C C G C G C C T
C T T C C G C C G T A
G G T G G G G G T T T
Sequence
alignment
Motif
Mutant
response 
 …ATGCTATA G GCCTGCCG… 
 …ATGCTATA C GCCTGCCG… 
ChIP-seq
peak
G
A
TC
G
ACG
ACA
GCA
GCG
C
GT
CG
A
TC
T
C
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
TG
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
G
A
TC
G
ACG
ACA
GCA
GCG
C
GT
CG
A
TC
T
C
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
TG
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
…
…
…
CGC 
 …ATGCTATA
T
A
T
A
C
G
C
C
T
A
T
A
G
G
C
C
T
A
T
A
C
G
C
C
CGC 
CGC WT
0.9 
>
>
>
>
CGC 
A
C
G
T
Figure 2. Principles of using neural networks for predicting molecular traits from DNA sequence.
(A) DNA sequence and the molecular response variable along the genome for three individuals. Conventional approaches in regulatory genomics consider variations between
individuals, whereas deep learning allows exploiting intra-individual variations by tiling the genome into sequence DNA windows centred on individual traits, resulting
in large training data sets from a single sample. (B) One-dimensional convolutional neural network for predicting a molecular trait from the raw DNA sequence in a window.
Filters of the first convolutional layer (example shown on the edge) scan for motifs in the input sequence. Subsequent pooling reduces the input dimension, and
additional convolutional layers can model interactions between motifs in the previous layer. (C) Response variable predicted by the neural network shown in
(B) for a wild-type and mutant sequence is used as input to an additional neural network that predicts a variant score and allows to discriminate normal from deleterious
variants. (D) Visualization of a convolutional filter by aligning genetic sequences that maximally activate the filter and creating a sequence motif. (E) Mutation map of
a sequence window. Rows correspond to the four possible base pair substitutions, columns to sequence positions. The predicted impact of any sequence change is colour-coded.
Letters on top denote the wild-type sequence with the height of each nucleotide denoting the maximum effect across mutations (figure panel adapted from Alipanahi et al, 2015).
Molecular Systems Biology 12: 878 | 2016 ª 2016 The Authors
Molecular Systems Biology Deep learning for computational biology Christof Angermueller et al
4
Downloaded from https://www.embopress.org on December 26, 2024 from IP 2a02:c7c:8e11:5e00:43c8:c5b2:93b4:fcbb.

[Figure 1]: WT CGTAAAGCCCTTGATAAACCCCTTCCCTGGA

- The chart type: a flowchart or diagram.
- Read every axis label, unit, and tick mark.
- Describe each data series and its trend.
- State the key quantitative finding (e.g. 'accuracy reaches 94.3% at epoch 50').
- Describe the key components and describe every connection and direction of flow.
- Name all components and describe every connection and direction of flow.

Transcribe exactly what is visible in the image.