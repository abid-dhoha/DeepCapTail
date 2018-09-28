"""
This model codes for the capsid and tail model: train and save modules
"""


def train_model(df_data, s_label):
    from code.capsid_tail_deep_models import capsid_model, tail_model
    from numpy import size

    capsid_model = capsid_model(size(df_data, 1))
    capsid_model.fit(df_data, s_label, epochs=150, batch_size=10, verbose=1)

    tail_model = tail_model(size(df_data, 1))
    tail_model.fit(df_data, s_label, epochs=150, batch_size=10, verbose=1)

    return capsid_model, tail_model


def main():
    from code.build_kmer_df_learn import build_kmer_df_learn
    from os.path import exists, join
    from os import makedirs
    from argparse import ArgumentParser
    from sklearn.externals.joblib import dump

    parser = ArgumentParser()
    parser.add('--p_lp_fasta', '-p_lp_fasta', help='path of the file that has the list of paths of fasta files '
                                                   'that you intend to train')
    parser.add('--p_l_label', '-p_l_label', help='path of the file that has the list of labels')
    parser.add('--p_output', '-p_output', help='path of the output path where the constructed models will be stored')

    args = parser.parse_args()
    # build df_learn, which the learning matrix
    df_data, s_label = build_kmer_df_learn(
        lp_fasta=[p_fasta for p_fasta in open(args.p_lp_fasta).read().splitlines()]
        , l_label=[int(label) for label in open(args.p_l_label).read().splitlines()]
    )
    # train capsid and tail models
    capsid_model, tail_model = train_model(df_data, s_label)
    # create the output directory if doesn't exist
    if exists(args.p_output):
        makedirs(args.p_ouput)
    # save the capsid and tail models
    dump(capsid_model, join(args.p_output, 'capsid.model'))
    dump(tail_model, join(args.p_output, 'tail.model'))


if __name__ == '__name__':
    main()
