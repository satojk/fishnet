import numpy as np
from time import sleep
from parser.reader import Game
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

_PGN_PATH_AND_FILENAME = './data/games.pgn'

def load_games():
    with open(_PGN_PATH_AND_FILENAME, 'r') as f:
        large_pgn = f.read()
    pgns = large_pgn.split('\n\n\n')[:-1]
    return [Game(pgn) for pgn in pgns]


def extract_features(games):
    out = [[],[]]
    for game in games:
        game.goToTurn(len(game.moves))
        out[0].append(game.num_pieces())
        out[1].append(game.ai_player())
    return map(np.array, out)


def extract_coords(games):
    out = [[],[]]
    for game in games:
        out[0].append(game.vectorize_moves())
        out[1].append(game.ai_player())
    return map(np.array, out)


def dataset_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=1)
    X_test, X_val, y_test, y_val = train_test_split(
            X_test, y_test, test_size=0.5, random_state=1)
    return X_train, X_test, X_val, y_train, y_test, y_val


def main():
    games = load_games()
    X, y = extract_coords(games)
    print(X)
    X_train, X_test, X_val, y_train, y_test, y_val = dataset_split(X, y)
    model = LogisticRegression().fit(X_train, y_train)
    print(model.score(X_test, y_test))


if __name__ == '__main__':
    main()

# g = Game(data)
# print(g.aiPlayer())
