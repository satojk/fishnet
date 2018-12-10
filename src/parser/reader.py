import numpy as np
import io
import chess
import chess.pgn
import collections
from collections import defaultdict


class Game:
    def __init__(self, pgn_text):
        self.pgn_text = pgn_text

        self.obj = chess.pgn.read_game(io.StringIO(self.pgn_text))
        self.headers = defaultdict(str, self.obj.headers)

        self.moves = [m for m in self.obj.main_line()]
        self.board = chess.Board()


    def ai_player(self):
        # White Cases
        if self.headers["White"] == "lichess AI level 5":
            return 0
        if self.headers["WhiteIsComp"] == "Yes":
            return 0
        # Black Cases
        if self.headers["Black"] == "lichess AI level 5":
            return 1
        if self.headers["BlackIsComp"] == "Yes":
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
        matrix = np.zeros((64, 2, 6))
        for location, piece in self.board.piece_map().items():
            dimension2 = 0 if piece.color else 1
            matrix[location][dimension2][piece.piece_type-1] = 1
        return matrix.flatten()

    def vectorize_moves(self, n):
        coords = np.array([to_coord(chess.Move.uci(m)) for m in self.moves]).flatten()
        return pad(coords, n * 2 * 4) # 35 moves, 2 players, 4 coords

    def controlled_squares(self, move_num):
        player = move_num % 2
        controlled_squares = 0
        self.go_to_move(move_num)
        for i in range(64):
            controlled_squares += self.board.is_attacked_by(player, i)
        return controlled_squares

    def controlled_squares_spread(self, move_num):
        player = move_num % 2
        controlled_squares = []
        self.go_to_move(move_num)
        centroid = [0, 0]
        for i in range(64):
            if self.board.is_attacked_by(player, i):
                controlled_squares.append((i//8, i%8))
                centroid[0] += (i//8)
                centroid[1] += (i%8)
        centroid[0] /= len(controlled_squares)
        centroid[1] /= len(controlled_squares)
        total_euclid_distance = 0
        for elem in controlled_squares:
            total_euclid_distance += (elem[0] - centroid[0])**2 + (elem[1] - centroid[1])**2
        return total_euclid_distance / len(controlled_squares)


##########################################################
# Utils
##########################################################

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
