# fishnet
Discriminates between Human and AI chess players

ML Techniques
1. Vectorized form [x1, x2, y1, y2]
2. Use log-reg for basic baseline
3. Try out SVMs with cool kernel's
4. If performance sucks, think of more creative vector structures to describe
   the game
5. Use time-series with RNN

Data Processing
1. Just Stockfish via Lichess (worry about generalization to generic
   "computer")
2. If easy, expand to OnlineChess which has several computers (remember to
   segregate computers in test and train sets)
