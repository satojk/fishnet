import sys
import pickle
import numpy as np
from time import sleep
from parser.reader import Game
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

_PGN_PATH_AND_FILENAME = './data/games.pgn'

def load_games():
    with open(_PGN_PATH_AND_FILENAME, 'r') as f:
        large_pgn = f.read()
    pgns = large_pgn.split('\n\n\n')[:-1]
    return [Game(pgn) for pgn in pgns]


def extract_pieces(game):
    game.go_to_turn(len(game.moves))
    return game.num_pieces()


def extract_coords(game):
    return game.vectorize_moves()


def extract_sparse_vector(game):
    full_game_vector = np.array([])
    next_state = game.board_state(0)
    for i in range(1, 36):
        try:
            next_state = game.board_state(i)
        except Exception:
            pass
        full_game_vector = np.concatenate((full_game_vector, next_state))
    return full_game_vector


def extract_sparse_vector_endgame(game):
    return game.board_state(len(game.moves))


def extract_features(games, extractor):
    out = [[],[]]
    for game in games:
        out[0].append(extractor(game))
        out[1].append(game.ai_player())
    return map(np.array, out)


def dataset_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=1)
    X_test, X_val, y_test, y_val = train_test_split(
            X_test, y_test, test_size=0.5, random_state=1)
    return X_train, X_test, X_val, y_train, y_test, y_val


def main():
    outputs = {}
    games = load_games()
    for n in range(2, 41, 2):
        def extract_sparse_vector_n_moves_even(game):
            n_moves_vector = np.array([])
            next_state = game.board_state(0)
            for i in range(1, n+1):
                try:
                    next_state = game.board_state(i)
                except Exception:
                    pass
                if i % 2 == 0:
                    n_moves_vector = np.concatenate((n_moves_vector, next_state))
            return n_moves_vector
        iter_outputs = {}
        print('Running simulations for n = {}'.format(n))
        print('Extraction is extract_sparse_vector_n_moves_even')
        X, y = extract_features(games, extract_sparse_vector_n_moves_even)
        X_train, X_test, X_val, y_train, y_test, y_val = dataset_split(X, y)

        C = 1.0  # SVM regularization parameter
        models = [
                ["Log-Reg: ",    LogisticRegression(solver='liblinear')],
                ["SVM Linear: ", svm.SVC(kernel='linear', C=C)],
                ["SVM RBF: ",    svm.SVC(kernel='rbf', gamma=0.7, C=C)],
                ["SVM Poly:",  svm.SVC(kernel='poly', gamma='auto', degree=3, C=C)],
                ]

        for name, clf in models:
            clf.fit(X_train, y_train)
            train_score = clf.score(X_train, y_train)
            test_score = clf.score(X_test, y_test)
            iter_outputs[name.strip(': ')] = (train_score, test_score)
            print(name)
            print("Train: ", train_score)
            print("Test: ", test_score)

        outputs[n] = iter_outputs

        print('\n\n')
        sys.stdout.flush()
    with open('outputs.pkl', 'wb') as pkl:
        pickle.dump(outputs, pkl)

if __name__ == '__main__':
    main()
