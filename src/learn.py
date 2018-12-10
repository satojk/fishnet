import sys
import gzip
import ubjson
import pickle
import numpy as np
import pprint
pp = pprint.PrettyPrinter()
from time import sleep
from copy import deepcopy
from functools import partial

from parser.reader import Game
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from keras.models import Sequential
from keras.layers import Dense

# _PGN_PATH_AND_FILENAME = './data/games.pgn'
# _CACHE_FEATURE_PATH = './data/games_cache.pkl'

_PGN_PATH_AND_FILENAME = './data/fics_2017_HvC.pgn'
_CACHE_FEATURE_PATH = './data/fics_cache.pkl'

##########################################################
# Utils
##########################################################

def get_next_game(f):
    pgn = f.readline()
    while True:
        if "EOF" in pgn: return None
        new_line = f.readline()
        pgn += new_line
        if new_line[:2] == "1.": break
    return pgn


def serialize_data(src, data):
    with gzip.open(src, "wb") as f:
        ubjson.dump(data, f)


def deserialize_data(src):
    with gzip.open(src, "rb") as f:
        return ubjson.load(f)

##########################################################
# Extractors
##########################################################

def performPCA(X, variance):
    pca = PCA(variance)
    print("Reducing dimensions with PCA")
    print("OG: ", X.shape)
    components = pca.fit_transform(X)
    print("Reduced: ", components.shape)
    return components

##########################################################
# Extractors
##########################################################

def extract_pieces(game):
    game.go_to_move(len(game.moves))
    a, b = game.num_pieces()
    out = np.zeros(256)
    out[(a-1)*16 + (b-1)] = 1
    return out

def extract_num_pieces_over_n_moves(game, n=50):
    n_moves_vector = []
    for i in range(1, n+1):
        try:
            game.go_to_move(i)
            next_num_pieces = game.num_pieces()[i%2]
        except Exception:
            next_num_pieces = 0
        n_moves_vector.append(next_num_pieces)
    return np.array(n_moves_vector)


def generate_coefficient_matrix_for_extract_pieces(clf):
    '''
    Generate a 16 x 16 coefficient matrix from a 256-long vector of
    coefficients (from a trained classifier)
    Probably should only be used for the above extractor `extract_pieces`
    '''
    coeff_matrix = np.zeros((16, 16))
    for a in range(16):
        for b in range(16):
            coeff_matrix[a][b] = round(clf.coef_[0][a*16 + b], 2)
    return coeff_matrix

def extract_coords_n_moves(game, n=15):
    return game.vectorize_moves(n)


def extract_sparse_vector(game):
    full_game_vector = np.array([])
    next_state = game.board_state(0)
    for i in range(1, 36):
        try: next_state = game.board_state(i)
        except Exception: pass
        full_game_vector = np.concatenate((full_game_vector, next_state))
    return full_game_vector


def extract_sparse_vector_n_moves_even(game, n=14):
    n_moves_vector = np.array([])
    next_state = game.board_state(0)
    for i in range(1, n+1):
        try: next_state = game.board_state(i)
        except Exception: pass
        if i % 2 == 0:
            n_moves_vector = np.concatenate((n_moves_vector, next_state))
    return n_moves_vector


def extract_sparse_vector_endgame(game):
    return game.board_state(len(game.moves))


def extract_controlled_squares(game, n=50):
    n_moves_vector = []
    for i in range(1, n+1):
        try: next_controlled = game.controlled_squares(i)
        except Exception: next_controlled = 0
        n_moves_vector.append(next_controlled)
    #print(n_moves_vector)
    return np.array(n_moves_vector)


def generate_coefficient_matrix_for_extract_controlled(clf):
    '''
    Probably should only be used for above extractor
    `extract_controlled_squares`
    '''
    return list(map(lambda x: round(x, 2), clf.coef_[0]))


def extract_average_controlled_squares(game):
    n_moves_vector_even = []
    n_moves_vector_odd = []
    i = 1
    while True:
        try:
            controlled_squares = game.controlled_squares(i)
            if i % 2:
                n_moves_vector_odd.append(controlled_squares)
            else:
                n_moves_vector_even.append(controlled_squares)
        except Exception:
            break
        i += 1
    return (sum(n_moves_vector_even) / len(n_moves_vector_even),
            sum(n_moves_vector_odd) / len(n_moves_vector_odd))


def extract_control_spread(game, n=50):
    n_moves_vector = []
    for i in range(1, n+1):
        try: next_control_spread = game.controlled_squares_spread(i)
        except Exception: next_control_spread = 0
        n_moves_vector.append(next_control_spread)
    #print(n_moves_vector)
    return np.array(n_moves_vector)

##########################################################
# Generic pipeline
##########################################################

def extract_features(extractor):
    out = [[],[]]
    with open(_PGN_PATH_AND_FILENAME, 'r') as f:
        i = 1
        while True:
            if i % 1000 == 0: print("{} games processed".format(i))
            pgn = get_next_game(f)
            if pgn == None: break
            game = Game(pgn)
            out[0].append(extractor(game))
            out[1].append(game.ai_player())
            i += 1
    return out


def get_features(extractor):
    try:
        print("Attempting to get cached features...")
        data = deserialize_data(_CACHE_FEATURE_PATH)
        print("SUCCESS")
    except:
        print("FAILED to get cached features...")
        print("Extracting features...")
        X, y = extract_features(extractor)
        X_reduced = performPCA(np.array(X), variance=0.95)
        data =  list(X_reduced), y
        print("Caching features...")
        serialize_data(_CACHE_FEATURE_PATH, data)

    print("Transforming to NumPy arrays")
    return map(np.array, data)


def dataset_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=1)
    X_test, X_val, y_test, y_val = train_test_split(
            X_test, y_test, test_size=0.5, random_state=1)
    return X_train, X_test, X_val, y_train, y_test, y_val


def train(models, X_train, y_train, X_test, y_test):
    print("Training...")
    output = {}
    for name, clf in models:
        print("Training {}: ".format(name))
        clf.fit(X_train, y_train)
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        output[name] = (train_score, test_score)
        print("Train: ", train_score)
        print("Test: ", test_score)
    return output


def extract_and_train(models, extractor):
    X, y = get_features(extractor)
    X_train, X_test, X_val, y_train, y_test, y_val = dataset_split(X, y)
    return train(models, X_train, y_train, X_test, y_test)

def extract_and_train_over_many_n(extractor):
    outputs = {}
    for i in range(2, 37, 2):
        print('\n\nApplying extractor with n = {}'.format(i))
        partial_extractor = partial(extractor, n=i)
        out = extract_and_train(partial_extractor)
        outputs[i] = deepcopy(out)
    with open('outputs.pkl', 'wb') as pkl:
        pickle.dump(outputs, pkl)


##########################################################
# Main Pipeline
##########################################################

def scikit_training():
    C = 1.0  # SVM regularization parameter
    models = [["Log-Reg", LogisticRegression(solver='liblinear')]]
    models.append(["SVM RBF C=10 gamma=0.04", svm.SVC(kernel='rbf', gamma=0.04, C=10)])

    for C in [0.1, 0.5, 1.0, 2.0, 10.0]:
        models.append(["SVM Linear {}".format(C), svm.SVC(kernel='linear', C=C)])
        models.append(["SVM RBF {}".format(C), svm.SVC(kernel='rbf', gamma=0.7, C=C)])
        models.append(["SVM Poly {}".format(C), svm.SVC(kernel='poly', gamma='auto', degree=3, C=C)])

    for C in [0.1, 0.5, 1.0, 2.0, 10.0]:
        for deg in np.arange(0.001, 0.05, 0.005):
            models.append(["SVM RBF C={} gamma={}".format(C, deg), svm.SVC(kernel='rbf', gamma=deg, C=C)])

    output = extract_and_train(models, extract_sparse_vector_n_moves_even)
    pp.pprint(output)

def keras_training():
    X, y = get_features(extract_sparse_vector_n_moves_even)
    X_train, X_test, X_val, y_train, y_test, y_val = dataset_split(X, y)
    _, n = X.shape

    outputs = {}

    model = Sequential()
    model.add(Dense(units=200, activation='relu', input_dim=n))
    model.add(Dense(units=40, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy']) 

    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
    loss, acc = model.evaluate(X_test, y_test, batch_size=128)
    outputs[(first_units, second_units)] = acc

    pp.pprint(outputs)

def main():
    keras_training()

if __name__ == '__main__':
    main()
