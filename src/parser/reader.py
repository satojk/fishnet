import io
import chess
import chess.pgn


class Game:
    def __init__(self, pgnText):
        self.pgnText = pgnText
        self.moves = [m for m in chess.pgn.read_game(io.StringIO(self.pgnText)).main_line()]
        self.board = chess.Board()

    def aiPlayer(self):
        lines = self.pgnText.split('\n')
        if lines[4] == "[White \"lichess AI level 5\"]":
            return 0
        if lines[5] == "[Black \"lichess AI level 5\"]":
            return 1
        raise Exception("No AI players detected")

    def turnNum(self):
        return len(self.board.move_stack)
    
    def goToTurn(self, turn):
        if self.turnNum() == turn:
            return
        if turn < 0 or turn > len(self.moves):
            raise Exception("turn out of bounds in goToTurn")
        if self.turnNum() < turn :
            for i in range(self.turnNum(), turn):
                self.board.push(self.moves[i])
        else:
            for _ in range(self.turnNum() - turn):
                self.board.pop()
 
    def numPieces(self):
        pieces = self.board.piece_map()
        whiteNum = 0
        for location, piece in pieces.items():
            if piece.color == chess.WHITE:
                whiteNum += 1
        return (whiteNum, len(pieces.keys()) - whiteNum)

    def vectorizeGrid(self):



# with open('./data/samples/example_game.pgn', 'r') as myFile :
#     data = myFile.read()

# game1 = Game(data)
# game1.goToTurn(len(game1.moves))
# print(game1.numPieces())
# print(game1.aiPlayer())
