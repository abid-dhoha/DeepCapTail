"""
use the capsid and tail models to predict aa sequences
"""


def predict_sequence(df_data, p_model, p_output):
    from sklearn.externals.joblib import load
    from os.path import join

    capsid_model = load(join(p_model, 'capsid.model'))
    tail_model = load(join(p_model, 'tail.model'))

    capsid_prediction = capsid_model.predict_proba(df_data)
    tail_prediction = tail_model.predict_proba(df_data)


def main():
    from argparse import ArgumentParser
    from code.build_kmer_df_learn import build_kmer_df_learn

    parser = ArgumentParser()
    parser.add_argument('--p_fasta', '-p_fasta', help='path of fasta file of amino acid sequences to predict')
    parser.add('--p_model', '-p_model', help='path of the directory of the capsid and tail models')
    parser.add('p_output', 'p_output', help='path of the output directory')

    args = parser.parse_args()

    df_data = build_kmer_df_learn(lp_fasta=[args.p_fasta])

    predict_sequence(df_data, args.p_model, args.p_output)


if __name__ == '__main__':
    main()
