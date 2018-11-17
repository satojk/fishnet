import sys
import pickle
import requests
import itertools
from time import sleep


_BASE_URL = 'https://lichess.org/game/export/XXXXX?evals=0&clocks=0'

# GAME_IDS is a dictionary of GAME_ID -> SCRAPING_SESSION pairs
_GAME_IDS_FILENAME = 'game_ids.pkl'

_RECONSTRUCTED_GAME_IDS_FILENAME = 'reconstructed_game_ids.pkl'

# PGN is a large pgn file containing a bunch of games
_PGN_FILENAME = 'games.pgn'

def load_game_ids():
    '''
    Load _GAME_IDS_FILENAME file
    Return it, or an empty dictionary if it is not found
    '''
    try:
        with open(_GAME_IDS_FILENAME, 'rb') as pkl:
            return pickle.load(pkl)
    except FileNotFoundError:
        return {}


def update_reconstructed_game_ids(new_value):
    '''
    Write a _RECONSTRUCTED_GAME_IDS_FILENAME file with new_value value
    '''
    with open(_RECONSTRUCTED_GAME_IDS_FILENAME, 'wb') as pkl:
        pickle.dump(new_value, pkl)


def reconstruct_game_id(game_id):
    remaining_chars = 8 - len(game_id)
    if remaining_chars == 0:
        return game_id
    characters = ['b', 'l', 'a', 'c', 'k', '_']
    reconstruction_templates = [template for template
                                in itertools.permutations(characters, 1 + remaining_chars) 
                                if '_' in template]
    for template in reconstruction_templates:
        new_game_id = ''.join(template).replace('_', game_id)
        r = requests.get(_BASE_URL.replace('XXXXX', new_game_id))
        lines = r.text.split('\n')
        if (lines[4] == '[White "lichess AI level 5"]' or
            lines[5] == '[Black "lichess AI level 5"]'):
            return new_game_id
    return None



def main():
    game_ids = load_game_ids()
    total_game_ids = len(game_ids)
    print('Loaded {} game_ids. Starting request sequence...'.format(
          total_game_ids))
    sys.stdout.flush()
    reconstructed_game_ids = {}
    for ix, game_id in enumerate(game_ids):
        print('Working with game {} of {}'.format(ix+1, total_game_ids))
        sys.stdout.flush()
        reconstructed_game_id = reconstruct_game_id(game_id)
        reconstructed_game_ids[reconstructed_game_id] = game_ids[game_id]
        if len(game_id) == 8:
            print('No reconstruction was needed for game {}'.format(game_id))
            sys.stdout.flush()
        elif reconstructed_game_id:
            print('\033[92mReconstructed game {} into game {}!\033[0m'.format(
                  game_id, reconstructed_game_id))
            sys.stdout.flush()
        else:
            print('\033[93mCould not reconstruct from game_id {}!!!\033[0m'.format(
                  game_id))
            sys.stdout.flush()
        update_reconstructed_game_ids(reconstructed_game_ids)


if __name__ == '__main__':
    main()
