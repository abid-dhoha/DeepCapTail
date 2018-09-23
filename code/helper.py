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


def build_kmer_df_learn(lp_fasta, l_label=None):
    from Bio.SeqIO import parse
    from itertools import chain
    from pandas import DataFrame, Series, concat

    l_kmer_size = [1, 2, 3]
    l_letter = ['M', 'F', 'L', 'I', 'V', 'P', 'T', 'A', 'Y', 'H', 'Q', 'N', 'K', 'D', 'E', 'C', 'R', 'S', 'W', 'G']
    l_kmer = list(chain(*[generate_kmer(kmer_size, l_letter, l_letter) for kmer_size in l_kmer_size]))
    l_l_kmer_freq = []
    l_seq_id = []
    for p_fasta in lp_fasta:
        for record in parse(p_fasta, 'fasta'):
            l_seq_id.append(record.id)
            d_record = {}
            seq = str(record.seq)
            len_seq = len(seq)
            for i in range(len_seq):
                for kmer in [seq[i:i + kmer_size] for kmer_size in l_kmer_size if i <= len_seq - kmer_size]:
                    d_record[kmer] = 1 if kmer not in d_record.keys() else d_record[kmer] + 1
            l_kmer_freq = []
            for kmer in l_kmer:
                l_kmer_freq.append(d_record[kmer] if kmer in d_record.keys() else 0)
            l_l_kmer_freq.append(l_kmer_freq)

    df_data = DataFrame(l_l_kmer_freq, columns=l_kmer, index=l_seq_id)

    if l_label:
        s_label = Series(name='label')
        for p_fasta, label in zip(lp_fasta, l_label):
            l_seq_id = [record.id for record in parse(p_fasta, 'fasta')]
            s_label = concat([s_label, Series(label, name='label', index=l_seq_id)])
            return df_data, s_label
    else:
        return df_data


def generate_kmer(kmer_size, l_kmer, l_letter):
    if all(len(kmer) == kmer_size for kmer in l_kmer):
        return l_kmer
    return generate_kmer(kmer_size, [''.join([kmer, letter]) for letter in l_letter for kmer in l_kmer], l_letter)
