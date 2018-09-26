"""
This model codes for the capsid and tail model: train and save modules
"""


def train_save_model(data, label):
    from code.capsid_tail_models import capsid_model, tail_model
    from numpy import size

    capsid_model = capsid_model(size(data, 1))
    capsid_model.fit(data, label, epochs=150, batch_size=10, verbose=1)

    tail_model = tail_model(size(data, 1))
    tail_model.fit(data, label, epochs=150, batch_size=10, verbose=1)


def main():
    from code.build_kmer_df_learn import build_kmer_df_learn
    from os.path import exists
    from os import makedirs
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add('--p_lp_fasta', '-p_lp_fasta', help='path of the file that has the list of paths of fasta files '
                                                   'that you intend to train')
    parser.add('--p_l_label', '-p_l_label', help='path of the file that has the list of labels')
    parser.add('--p_output', '-p_output', help='path of the output path where the constructed models will be stored')

    args = parser.parse_args()

    data, train = build_kmer_df_learn(
        lp_fasta=[p_fasta for p_fasta in open(args.p_lp_fasta).read().splitlines()]
        , l_label=[int(label) for label in open(args.p_l_label).read().splitlines()]
    )

    # check this one to save your model.
    # https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
    if exists(args.p_output):
        makedirs(args.p_ouput)


if __name__ == '__name__':
    main()
