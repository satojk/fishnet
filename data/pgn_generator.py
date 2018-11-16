import pickle
import requests
from time import sleep

_BASE_URL = 'https://lichess.org/game/export/XXXXX?evals=0&clocks=0'

# GAME_IDS is a dictionary of GAME_ID -> SCRAPING_SESSION pairs
_GAME_IDS_FILENAME = 'game_ids.pkl'

# CONVERTED_GAME_IDS is a set of GAME_IDS which we already have the pgns for
_CONVERTED_GAME_IDS_FILENAME = 'converted_game_ids.pkl'

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


def load_converted_game_ids():
    '''
    Load _CONVERTED_GAME_IDS_FILENAME file
    Return it, or an empty set if it is not found
    '''
    try:
        with open(_CONVERTED_GAME_IDS_FILENAME, 'rb') as pkl:
            return pickle.load(pkl)
    except FileNotFoundError:
        return set()


def update_converted_game_ids(new_value):
    '''
    Write a _CONVERTED_GAME_IDS_FILENAME file with new_value value
    '''
# I will call this every time I generate a new pgn. This is super inefficient, 
# but my main concern here is bounding my space efficiency (I don't want to 
# have a bunch of pgns in memory for batch writing, and I want 
# converted_game_ids to constantly reflect what pgns I already have)
    with open(_CONVERTED_GAME_IDS_FILENAME, 'wb') as pkl:
        pickle.dump(new_value, pkl)


def write_new_pgn(pgn):
    '''
    Append a new pgn to the end of _PGN_FILENAME file
    '''
    with open(_PGN_FILENAME, 'a') as out_fp:
        out_fp.write(pgn)


def get_pgn_file(game_id):
    '''
    Send request for the pgn for game_id
    Return text of response
    '''
    r = requests.get(_BASE_URL.replace('XXXXX', game_id))
    return r.text


def main():
    game_ids = load_game_ids()
    total_game_ids = len(game_ids)
    print('Loaded {} game_ids. Starting request sequence...'.format(
          total_game_ids))
    converted_game_ids = load_converted_game_ids()
    for ix, game_id in enumerate(game_ids):
        print('Working with game {} of {}'.format(ix+1, total_game_ids))
        if game_id in converted_game_ids:
            print('Skipping game_id {}...'.format(game_id))
            continue
        new_pgn = get_pgn_file(game_id)
        if new_pgn:
            print('Writing pgn for game_id {}...'.format(game_id))
            converted_game_ids.add(game_id)
            update_converted_game_ids(converted_game_ids)
            write_new_pgn(new_pgn)
        else:
            print('No text returned for game_id {}. Investigate?'.format(
                  game_id))
        sleep(3)


if __name__ == '__main__':
    main()
