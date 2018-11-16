import pickle
import requests
from time import sleep
from datetime import datetime
from bs4 import BeautifulSoup


__BASE_URL = 'https://lichess.org/games/search?page=XXXXX&hasAi=1&aiLevelMin=5&aiLevelMax=5&perf=3&sort.field=d&sort.order=desc&_=YYYYY'

_BASE_PAGE_NO = 2
_BASE_PAGE_ID = 1541890166213
_NUM_PAGES_TO_SCRAPE = 10
_VERBOSE = True

# I'm pickling all requests I've made as well as the game ids. Game ids are
# useful because we'll later fetch all of their pgns. page requests are not
# necessarily very useful, but I'm keeping track of them to later study a bit
# how to not spam lichess with unnecessary requests.

# PAGE REQUESTS is a list of 4-tuples of the form (SCRAPING_SESSION,
# DATETIME, FIRST_REQUEST, LAST_REQUEST). SCRAPING_SESSION is an
# autoincrementing int denoting how many times i've ran this script already
# (should correspond to ix in the list), DATETIME is the datetime for the
# beginning of the scraping session, first request is the first request made in
# the scraping session (the arg to load_page), and last request is the last
# request made in the scraping session. all requests between the first and last
# should be simply incrementing/decrementing the id and pageno.

_PAGE_REQUESTS_FILENAME = 'page_requests.pkl'

# GAME IDS is a dict of {GAME_ID: ORIGINAL_SCRAPING_SESSION}
_GAME_IDS_FILENAME = 'game_ids.pkl'

# SCRAPING SESSION is an int. It records how many scraping sessions have
# happened
_SCRAPING_SESSION_FILENAME = 'scraping_session.pkl'

###############################################################################
### THE FOLLOWING 4 FUNCTIONS ARE SIMPLY FOR PICKLING / UNPICKLING INFO     ###
### UNLESS YOU'RE ME, YOU PROBABLY DON'T HAVE TO WORRY ABOUT THIS           ###
###############################################################################


def update_page_requests(scraping_session, first_request, last_request,
                         verbose=False):
    '''
    Load _PAGE_REQUESTS_FILENAME file
    Append (scraping_session, cur_datetime, first_request, last_request)
    Write updated _PAGE_REQUESTS_FILENAME file
    '''
    page_requests = []
    new_page_request = (scraping_session,
                        datetime.utcnow(),
                        first_request,
                        last_request)
    try:
        with open(_PAGE_REQUESTS_FILENAME, 'rb') as pkl:
            page_requests = pickle.load(pkl)
            if verbose:
                print('Last page request elem: {}'.format(page_requests[-1]))
    except FileNotFoundError:
        if verbose:
            print('No file {} found. Generating new pickle...'.format(
                  _PAGE_REQUESTS_FILENAME))
    page_requests.append(new_page_request)
    with open(_PAGE_REQUESTS_FILENAME, 'wb') as pkl:
        if verbose:
            print('New last page request elem: {}'.format(new_page_request))
        pickle.dump(page_requests, pkl)


def update_game_ids(new_game_ids, verbose=False):
    '''
    Load _GAME_IDS_FILENAME file
    Add newly scraped game ids to it
    Write updated _GAME_IDS_FILENAME
    Verbose: print old and new number of game ids
    '''
    try:
        with open(_GAME_IDS_FILENAME, 'rb') as pkl:
            old_game_ids = pickle.load(pkl)
            if verbose:
                print('Previously there were {} saved game ids'.format(
                      len(old_game_ids)))
            for old_key in old_game_ids.keys():
                new_game_ids[old_key] = old_game_ids[old_key]
    except FileNotFoundError:
        if verbose:
            print('No file {} found. Generating new pickle...'.format(
                  _GAME_IDS_FILENAME))
    with open(_GAME_IDS_FILENAME, 'wb') as pkl:
        if verbose:
            print('After updating, there are {} saved game ids'.format(
                  len(new_game_ids)))
        pickle.dump(new_game_ids, pkl)


def load_scraping_session(verbose=False):
    '''
    Load and return _SCRAPING_SESSION_FILENAME
    '''
    scraping_session = 0
    try:
        with open(_SCRAPING_SESSION_FILENAME, 'rb') as pkl:
            scraping_session = pickle.load(pkl)
    except FileNotFoundError:
        pass
    if verbose:
        print('Current scraping session: {}'.format(scraping_session+1))
    return scraping_session


def update_scraping_session(new_scraping_session, verbose=False):
    '''
    Write updated _SCRAPING_SESSION_FILENAME
    '''
    if verbose:
        print('New value for scraping session: {}'.format(
              new_scraping_session))
    with open(_SCRAPING_SESSION_FILENAME, 'wb') as pkl:
        pickle.dump(new_scraping_session, pkl)


###############################################################################
### THE ABOVE 4 FUNCTIONS ARE SIMPLY FOR PICKLING / UNPICKLING INFO         ###
### UNLESS YOU'RE ME, YOU PROBABLY DON'T HAVE TO WORRY ABOUT THIS           ###
###############################################################################


def load_page(url):
    '''
    GET url
    Return webpage as a txt
    '''
    r = requests.get(url)
    return r.text


def extract_game_ids_from_page(scraping_session, page, game_ids):
    '''
    Parse page text into a BeautifulSoup
    Add new values to game_ids of form game_id: scraping_session taken from page
    '''
    soup = BeautifulSoup(page, 'html.parser')
    game_row_elements = soup.find_all('div',
                                      class_='game_row paginated_element')
    for game_row_element in game_row_elements:
        next_game_id = game_row_element.find('a')['href'].strip('/black')
        print('Next game id: {}'.format(next_game_id))
        game_ids[next_game_id] = scraping_session


def main():
    scraping_session = load_scraping_session()
    page_no = _BASE_PAGE_NO
    page_id = _BASE_PAGE_ID
    game_ids = {}
    first_request = None
    last_request = None
    for i in range(_NUM_PAGES_TO_SCRAPE):
        print('Scraping for the {}th time'.format(i))
        url = __BASE_URL.replace('XXXXX', str(page_no))
        url = url.replace('YYYYY', str(page_id))
        if _VERBOSE:
            print('Scraping page {}'.format(url))
        if not i:
            first_request = url
        last_request = url
        page = load_page(url)
        extract_game_ids_from_page(scraping_session, page, game_ids)
        page_no += 1
        page_id += 1
        sleep(7)
    update_page_requests(scraping_session, first_request, last_request)
    update_game_ids(game_ids)
    update_scraping_session(scraping_session+1)


if __name__ == '__main__':
    main()
