import reader

with open("../../data/samples/example_game.pgn") as my_file:
    data = my_file.read()

game1 = reader.Game(data)
print(game1.board_state(0))
