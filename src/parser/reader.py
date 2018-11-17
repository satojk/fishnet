import numpy as np
import io
import chess
import chess.pgn
import collections


class Game:
    def __init__(self, pgn_text) :
        self.pgn_text = pgn_text
        self.moves = [m for m in chess.pgn.read_game(io.StringIO(self.pgn_text)).main_line()]
        self.board = chess.Board()


    def ai_player(self) :
        lines = self.pgn_text.split('\n')
        if lines[4] == "[White \"lichess AI level 5\"]":
            return 0
        if lines[5] == "[Black \"lichess AI level 5\"]":
            return 1
        raise Exception("No AI players detected")

    def turn_num(self) :
        return len(self.board.move_stack)
    
    def go_to_turn(self, turn) :
        if self.turn_num() == turn :
            return
        elif turn < 0 or turn > len(self.moves) :
            raise Exception("turn out of bounds in goToTurn")
        elif self.turn_num() < turn :
            for i in range(self.turn_num(), turn) :
                self.board.push(self.moves[i])
        else :
            for _ in range(self.turn_num() - turn) :
                self.board.pop()
 
    def num_pieces(self) :
        pieces = self.board.piece_map()
        white_num = 0
        for location, piece in pieces.items() :
            if piece.color == chess.WHITE :
                white_num += 1
        return (white_num, len(pieces.keys()) - white_num)

    def pieces_lost(self) :
        self.go_to_turn(0)
        captures = []
        pieces = collections.defaultdict(int)
        for location, piece in self.board.piece_map().items() :
            pieces[piece] += 1
        for i in range(len(self.moves)) :
            if self.board.is_capture(self.moves[i]) :
                self.board.push(self.moves[i])
                new_pieces = collections.defaultdict(int)
                for location, piece in self.board.piece_map().items() :
                    new_pieces[piece] += 1
                for piece, quantity in pieces.items() :
                    if new_pieces[piece] < quantity :
                        captures.append(piece.symbol())
                pieces = new_pieces
            else :
                self.board.push(self.moves[i])
        return captures

    def vectorize_moves(self):
        coords = np.array([to_coord(chess.Move.uci(m)) for m in self.moves]).flatten()
        return pad(coords, 35 * 2 * 4) # 35 turns, 2 players, 4 coords

def to_coord(uci):
    return [ord(uci[0]) - ord('a'), 
            int(uci[1]) - 1, 
            ord(uci[2]) - ord('a'), 
            int(uci[3]) - 1 ]

def pad(array, length):
    pad = np.zeros(length)
    end = min(len(array), length)
    pad[:end] = array[:end]
    return pad
