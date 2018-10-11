"""
use the capsid and tail models to predict aa sequences
"""


def predict_sequence(capsid_tail, df_data, p_model_json, p_model_h5):
    from os.path import join
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

    return DataFrame({capsid_tail: [item[0] for item in prediction]})


def main():
    from argparse import ArgumentParser
    from code.build_kmer_df_learn import build_kmer_df_learn
    from os.path import exists
    from os import makedirs
    from ntpath import split

    parser = ArgumentParser()
    parser.add_argument('--p_fasta', '-p_fasta', help='path of fasta file of amino acid sequences to predict')
    parser.add_argument('--capsid_tail', '-model_name', help='capsid or tail to predict either capsid or tail')
    parser.add('p_output', 'p_output', help='path of the output file')

    args = parser.parse_args()

    p_model_json = 'models/capsid.json' if args.capsid_tail == 'capsid' else 'models/tail.json'
    p_model_h5 = 'models/tail.h5' if args.capsid_tail == 'capsid' else 'model/tail.h5'

    df_data = build_kmer_df_learn(lp_fasta=[args.p_fasta])

    df_prediction = predict_sequence(args.capsid_tail, df_data, p_model_json, p_model_h5)

    p_directory, p_file = split(args.p_output)
    if not exists(p_directory):
        makedirs(p_directory)
    df_prediction.to_csv(args.p_output)


if __name__ == '__main__':
    main()
