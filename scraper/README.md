Two scripts in this directory: scraper.py and pgn_generator.py

The first one hits Lichess's search endpoint with our filters (Stockfish Level 
5, etc) and populates game_ids.pkl.

The second one takes game_ids.pkl and hits Lichess's export endpoint and 
generates a games.pgn file, containing the pgn for each game_id in 
game_ids.pkl.

Plan:

`example_requests` give a few examples of the GET requests sent when scrolling
down the search result page. We can get expected responses without providing
any headers, it seems (not even cookies).

An example response is in `example_page`

The response gives a full web page. This web page has `div`s of class "game_row
paginated_element", each of which has, as a top level internal element, an
anchor tag with href=[GAME CODE].

Example [GAME CODE]s: /KeaUF5g0, /FTPmiuah, /Ma74vB2j, etc.
Note that, sometimes, a `/black` might follow the [GAME CODE] (eg
/KeaUF5g0/black). You probably want to strip that. This happens when black won
the game, I think.

Sending a GET request to https://lichess.org/game/export[GAME
CODE]?evals=0&clocks=0 initiates the download of a png file with the game's
info.

This png file is all we want. An example file is in this directory as
`example_game.pgn`. There are 15 rows containing a piece of information within
square brackets, where each piece of information follows the format [KEY]
[WHITESPACE] "[VALUE]" (eg: [Date "2018.11.02"] is a row), followed by a blank
line, followed by a line containing the algebraic notation of the game (this is
the 18th line in the example file), followed by two blank lines.

