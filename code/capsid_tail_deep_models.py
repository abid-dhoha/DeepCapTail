"""
We encode here the capsid and tail models. In other words, we define the number of layers,
number of nodes, and activation functions.
"""


from keras.models import Sequential
from keras.layers import Dense


def capsid_model(nb_feature):
    """
    capsid model has the structure 400,200,100,50 nodes
    :param nb_feature:
    :return:
    """
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
