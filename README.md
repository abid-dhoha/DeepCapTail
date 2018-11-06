# INTRODUCTION
DeepCapTail is deep learning package that predicts capsid and tail proteins of phage genomes. Basically, the input for this package is a protein sequence, and the output is the propabality that the given sequence is capsid or tail protein. This package is entirely written in Python.

# SYSTEM REQUIREMENTS
This project should run on all Unix platform, although it was only tested on Ubuntu 16.04.4 LTS

# SOFTWARE REQUIREMENTS
DeepCapTail requires the following packages:
1. Python - version 3.6 or later
2. Biopython - version 1.7 or later
3. keras - version 2.1.6 or later
# TO PREDICT SEQUENCES
1. Clone this repository
2. % cd DeepCapTail/
3. python predict_sequence.py --p_fasta protein.fasta --capsid_tail capsid --p_output prediction.csv

where 
'protein.fasta': fasta file of the sequences that we want to predict.
'capsid ' to predict capsid probability and 'tail' to predict tail probability.
'prediction.csv': csv file that will include the prediction of probabilty of the sequence being capsid or tail.

# HELP
% python predict_sequence.py --help

# OUTPUT FILE
The output file is a csv file that has two columns. The first columns is the id of the protein sequence and the second column is the probabilty of the sequence being capsid or tail.
