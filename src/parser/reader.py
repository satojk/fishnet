import numpy as np
import io
import chess
import chess.pgn
import collections


class Game:
    def __init__(self, pgn_text):
        self.pgn_text = pgn_text
        self.moves = [m for m in chess.pgn.read_game(io.StringIO(self.pgn_text)).main_line()]
        self.board = chess.Board()


    def ai_player(self):
        lines = self.pgn_text.split('\n')
        if lines[4] == "[White \"lichess AI level 5\"]":
            return 0
        if lines[5] == "[Black \"lichess AI level 5\"]":
            return 1
        raise Exception("No AI players detected")

    def move_num(self):
        return len(self.board.move_stack)
    
    def go_to_move(self, move):
        if self.move_num() == move:
            return
        elif move < 0 or move > len(self.moves):
            raise Exception("move out of bounds in go_to_move")
        elif self.move_num() < move:
            for i in range(self.move_num(), move):
                self.board.push(self.moves[i])
        else:
            for _ in range(self.move_num() - move):
                self.board.pop()
 
    def num_pieces(self):
        pieces = self.board.piece_map()
        white_num = 0
        for location, piece in pieces.items():
            if piece.color == chess.WHITE:
                white_num += 1
        return (white_num, len(pieces.keys()) - white_num)

    def pieces_lost(self):
        self.go_to_move(0)
        captures = []
        pieces = collections.defaultdict(int)
        for location, piece in self.board.piece_map().items():
            pieces[piece] += 1
        for i in range(len(self.moves)):
            if self.board.is_capture(self.moves[i]):
                self.board.push(self.moves[i])
                new_pieces = collections.defaultdict(int)
                for location, piece in self.board.piece_map().items():
                    new_pieces[piece] += 1
                for piece, quantity in pieces.items():
                    if new_pieces[piece] < quantity:
                        captures.append(piece.symbol())
                pieces = new_pieces
            else:
                self.board.push(self.moves[i])
        return captures

    def board_state(self, move_num):
        self.go_to_move(move_num)
        matrix = np.zeros((64, 2, 6), dtype = 'bool')
        for location, piece in self.board.piece_map().items():
            dimension2 = 0 if piece.color else 1
            matrix[location][dimension2][piece.piece_type-1] = True
        return matrix.flatten()

    def vectorize_moves(self, n):
        coords = np.array([to_coord(chess.Move.uci(m)) for m in self.moves]).flatten()
        return pad(coords, n * 2 * 4) # 35 moves, 2 players, 4 coords

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

