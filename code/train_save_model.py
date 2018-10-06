"""
This model codes for the capsid and tail model: train and save modules
"""


def train_model(df_data, s_label, model_name):
    from code.capsid_tail_deep_models import capsid_model, tail_model
    from numpy import size

    if model_name == 'capsid':
        model = capsid_model(size(df_data, 1))
        model.fit(df_data, s_label, epochs=150, batch_size=10, verbose=0)
    else:
        model = tail_model(size(df_data, 1))
        model.fit(df_data, s_label, epochs=150, batch_size=10, verbose=0)

    return model


def main():
    from code.build_kmer_df_learn import build_kmer_df_learn
    from os.path import exists, join
    from os import makedirs
    from argparse import ArgumentParser
    from sklearn.externals.joblib import dump
    import pickle

    parser = ArgumentParser()
    parser.add_argument('--p_lp_fasta', '-p_lp_fasta', help='path of the file that has the list of paths of fasta'
                                                            ' files that you intend to train')
    parser.add_argument('--p_l_label', '-p_l_label', help='path of the file that has the list of labels')
    parser.add_argument('--model_name', '-model_name', help='capsid or tail which is the name of the model')
    parser.add_argument('--p_output', '-p_output', help='path of the output path where the constructed models '
                                                        'will be stored')

    args = parser.parse_args()
    # build df_learn, which the learning matrix
    df_data, s_label = build_kmer_df_learn(
        lp_fasta=[p_fasta for p_fasta in open(args.p_lp_fasta).read().splitlines()]
        , l_label=[int(label) for label in open(args.p_l_label).read().splitlines()]
    )
    # train capsid and tail models
    model = train_model(df_data, s_label, args.model_name)
    # create the output directory if doesn't exist
    #if exists(args.p_output):
    #    makedirs(args.p_ouput)
    # save the capsid and tail models
    # pickle.dump(model, open(args.p_output, 'wb'))

    # serialize model to JSON
    model_json = model.to_json()
    with open(args.p_output, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights('.'.join([args.p_output, 'h5']))
    print("Saved model to disk")
    # https://machinelearningmastery.com/save-load-keras-deep-learning-models/


if __name__ == '__main__':
    main()
