clean = open("../data/fics_games_2017.pgn", "w+")

with open("../data/fics_games_2017_raw.pgn") as raw:
    for i, line in enumerate(raw):
        if (i % 1000): print(i)
        if "\r\n" in line:
            clean.write(line.replace("\r\n", "\n"))
        else:
            clean.write("\n\n")
