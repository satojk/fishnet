import reader

with open("./scraper/samples/example_game.pgn") as my_file:
    data = my_file.read()

game1 = reader.Game(data)
print(game1.ai_player())
