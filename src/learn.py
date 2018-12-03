import sys
import gzip
import ubjson
import numpy as np
from time import sleep

from parser.reader import Game
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# _PGN_PATH_AND_FILENAME = './data/games.pgn'
_PGN_PATH_AND_FILENAME = './data/fics_2017_HvC.pgn'
_CACHE_FEATURE_PATH = './data/fics_sparse_vector_n_14.pkl'

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

def extract_pieces(game):
    game.go_to_move(len(game.moves))
    a, b = game.num_pieces()
    out = np.zeros(256)
    out[(a-1)*16 + (b-1)] = 1
    return out

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
        data = extract_features(extractor)
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
    print(X.shape)
    X = X[:5000]
    y = y[:5000]
    X_train, X_test, X_val, y_train, y_test, y_val = dataset_split(X, y)
    return train(models, X_train, y_train, X_test, y_test)


##########################################################
# Main
##########################################################

C = 1.0  # SVM regularization parameter
models = [["Log-Reg", LogisticRegression(solver='liblinear')]]

for C in [0.1, 0.5, 1.0, 2.0, 10.0]:
    models.append(["SVM Linear {}".format(C), svm.SVC(kernel='linear', C=C)])
    models.append(["SVM RBF {}".format(C), svm.SVC(kernel='rbf', gamma=0.7, C=C)])
    models.append(["SVM Poly {}".format(C), svm.SVC(kernel='poly', gamma='auto', degree=3, C=C)])

def main():
    # train_n_moves(models, games)
    output = extract_and_train(models, extract_sparse_vector_n_moves_even)
    print(output)

if __name__ == '__main__':
    main()
