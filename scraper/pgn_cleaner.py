# clean = open("../data/fics_2017_HvC.pgn", "w+")

# with open("../data/fics_2017_HvC_raw.pgn") as raw:
#     for i, line in enumerate(raw):
#         if (i % 1000): print(i)
#         if "\r\n" in line:
#             clean.write(line.replace("\r\n", "\n"))
#         else:
#             clean.write("\n\n")
#     clean.write("\n\n")

def writeEOF(src):
    with open(src, "a") as f:
        f.write("EOF\n")

writeEOF("../data/fics_2017_HvC.pgn")
writeEOF("../data/games.pgn")
