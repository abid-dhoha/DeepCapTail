"""
This model codes for the capsid and tail model: train and save modules
"""


def train_model(df_data, s_label, model_name):
    # from code.capsid_tail_deep_models import capsid_model, tail_model
    from numpy import size

    if model_name == 'capsid':
        model = capsid_model(size(df_data, 1))
        model.fit(df_data, s_label, epochs=150, batch_size=10, verbose=1)
    else:
        model = tail_model(size(df_data, 1))
        model.fit(df_data, s_label, epochs=150, batch_size=10, verbose=1)

    return model


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


def capsid_model(nb_feature):
    """
    capsid model has the structure 400,200,100,50 nodes
    :param nb_feature:
    :return:
    """
    from keras.models import Sequential
    from keras.layers import Dense


    # construct the model
    model = Sequential()
    model.add(Dense(400, input_dim=nb_feature, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile and save the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def tail_model(nb_feature):
    """
    tail model has the structure 600,300,150,60 nodes
    :param nb_feature:
    :return:
    """
    from keras.models import Sequential
    from keras.layers import Dense


    # construct the model
    model = Sequential()
    model.add(Dense(600, input_dim=nb_feature, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile and save the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def main():
    # from code.build_kmer_df_learn import build_kmer_df_learn
    from os.path import exists
    from os import makedirs
    from argparse import ArgumentParser
    from ntpath import split

    parser = ArgumentParser()
    parser.add_argument('--p_lp_fasta', '-p_lp_fasta', help='path of the file that has the list of paths of fasta'
                                                            ' files that you intend to train')
    parser.add_argument('--p_l_label', '-p_l_label', help='path of the file that has the list of labels')
    parser.add_argument('--model_name', '-model_name', help='capsid or tail which is the name of the model')
    parser.add_argument('--p_output', '-p_output', help='path of the output file where the built models will be stored')

    args = parser.parse_args()
    # build df_learn, which the learning matrix
    df_data, s_label = build_kmer_df_learn(
        lp_fasta=[p_fasta for p_fasta in open(args.p_lp_fasta).read().splitlines()]
        , l_label=[int(label) for label in open(args.p_l_label).read().splitlines()]
    )

    # train capsid and tail models
    model = train_model(df_data, s_label, args.model_name)

    # create the output directory if doesn't exist
    p_directory, p_file = split(args.p_output)
    if not exists(p_directory):
        makedirs(p_directory)

    # serialize the model to JSON
    model_json = model.to_json()
    with open('.'.join([args.p_output, 'json']), "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights('.'.join([args.p_output, 'h5']))
    print("Model is saved.")
    # code from https://machinelearningmastery.com/save-load-keras-deep-learning-models/


if __name__ == '__main__':
    main()
