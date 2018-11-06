"""
use the capsid and tail models to predict aa sequences
"""


def predict_sequence(capsid_tail, df_data, p_model_json, p_model_h5):
    from keras.models import model_from_json
    from pandas import DataFrame

    # load the model
    with open(p_model_json, 'r') as f_json:
        model_json = f_json.read()
    model = model_from_json(model_json)
    model.load_weights(p_model_h5)

    # do the prediction with the loaded model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    prediction = model.predict_proba(df_data)

    return DataFrame(
        {'_'.join([capsid_tail, 'predictions']): [item[0] for item in prediction]}
        , index=df_data.index.values
    )


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


def main():
    from argparse import ArgumentParser
    #from code.build_kmer_df_learn import build_kmer_df_learn
    from os.path import exists
    from os import makedirs
    from ntpath import split

    parser = ArgumentParser()
    parser.add_argument('--p_fasta', '-p_fasta', help='path of fasta file of amino acid sequences that we want to'
                                                      ' predict')
    parser.add_argument('--capsid_tail', '-capsid_tail', help='"capsid" for capsid prediction and "tail" for tail '
                                                              'prediction')
    parser.add_argument('--p_output', '-p_output', help='path of the output file of the predictions')

    args = parser.parse_args()

    p_model_json = 'models/capsid.json' if args.capsid_tail == 'capsid' else 'models/tail.json'
    p_model_h5 = 'models/capsid.h5' if args.capsid_tail == 'capsid' else 'model/tail.h5'

    df_data = build_kmer_df_learn(lp_fasta=[args.p_fasta])

    df_prediction = predict_sequence(args.capsid_tail, df_data, p_model_json, p_model_h5)

    p_directory, p_file = split(args.p_output)
    if not exists(p_directory):
        makedirs(p_directory)
    df_prediction.to_csv(args.p_output, index_label='ids')


if __name__ == '__main__':
    main()
