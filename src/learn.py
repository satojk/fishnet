from parser.reader import Game

with open('./data/samples/example_game.pgn', 'r') as f:
    data = f.read()

g = Game(data)
print(g.aiPlayer())
