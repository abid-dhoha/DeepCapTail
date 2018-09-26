"""
We code in this module helper functions
"""


def is_fasta_file(p_fasta):
    """
    check if the path is a valid fasta file
    :param p_fasta:
    :return:
    """
    from Bio.SeqIO import parse

    fasta = parse(open(p_fasta, 'r'), 'fasta')
    return any(fasta)
