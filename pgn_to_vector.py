# makes heavy use of python-chess library

# plan of attack for generating dataset:
# 1. load pgns from databse
# 2. iterate through and for each game, choose 1-5 random positions from the game
# 3. use the UCI engine communication to evaluate each position
# 4. vectorize the positions and add to dataset, add evaluations to y-vector
# 5. pickle data? save in some easily loadable format for keras

import chess
import chess.pgn
import chess.uci
from random import randint
import pickle
import time

# dictionary corresponding to piece values, for vectorization
pieces = {'P': 1, 'N': 3, 'B': 4, 'R': 5, 'Q': 9, 'K': 15,
        'p': -1, 'n': -3, 'b': -4, 'r': -5, 'q': -9, 'k': -15}

#initialize chess engine
engine = chess.uci.popen_engine("/root/Stockfish/src/stockfish")
info_handler = chess.uci.InfoHandler()
engine.info_handlers.append(info_handler)

def main():
    start = time.time()

    training_positions = []
    training_scores = []

    # loop through pgns (Obviously change from a single game, but works on a single game too.)
    # Hideously long runtime on a large file.
    with open("large_chess_database.pgn", "r", encoding="latin-1") as pgn:
        game = chess.pgn.read_game(pgn)
        while game != None:
            node = game
            moves = node.main_line()
            game_length = node.board().variation_san(moves).count('.')
            game_length = (game_length * 2)
            if game_length <= 1:
                game = chess.pgn.read_game(pgn)
                continue
            position = randint(1, game_length - 1)
            for x in range(1, position):
                next_node = node.variations[0]
                node = next_node
            # send it to vectorizing function
            training_positions.append(vectorize_position(node.board().epd()))
            training_scores.append(comp_eval(node.board()))

            game = chess.pgn.read_game(pgn)

   # pickle data to load into model later
    data = [training_positions, training_scores]
    pickle.dump( data , open("training_set_1.p", "wb") )
    end = time.time()
    print(end - start)
    print(len(training_positions))




# translate epd to vector
# takes a string representation in epd format as input,
# output is a 1-dimensional vector of length 71
# values of pieced: pawn, 1, knight 3, bishop 4, rook 5, queen 9, king 15
def vectorize_position(epd):
    # create output vector
    vector = [0]*71
    #split epd into positional / rules info
    position, rules = epd.split(" ", 1)
    # iterate through string
    slot = 0 # keep track of position in vector
    for c in position:
        if c.isdigit():
            # fill next x slots with zeroes / leave them as is
            slot += eval(c)
        elif c == '/':
            pass
        else:
            vector[slot] = pieces[c]
            slot += 1
    turn = rules.split(" ")[0]
    castling = rules.split(" ")[1]
    # input rules info
    # whose turn
    if 'w' in turn:
        vector[slot] = 1
    slot +=1

    #castling rights
    if 'K' in castling:
        vector[slot] = 1
    slot +=1
    if 'Q' in castling:
        vector[slot] = 1
    slot += 1
    if 'k' in castling:
        vector[slot] = 1
    slot += 1
    if 'q' in castling:
        vector[slot] = 1

    return vector



# run position through chess engine
def comp_eval(board):
    engine.position(board)
    engine.go(movetime=1000)
    return info_handler.info["score"][1].cp

if __name__ == "__main__":
    main()
