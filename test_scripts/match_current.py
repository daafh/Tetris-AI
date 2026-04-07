from __future__ import print_function, division
import csv
import simplejson as json
import pandas as pd
import random
import unicodedata

from cogworks.tetris.game import State, Board, zoids
from cogworks.tetris import features, simulator
from cogworks import feature

#converts a json representation into a board element of a State object
def parse_board(rep):
    rows = len(rep)
    cols = None
    for row in rep:
        if cols is None:
            cols = len(row)
        else:
            assert cols == len(row)
    board = Board(rows, cols, zero=False)
    board.heights = [0] * cols
    for r in range(0, rows):
        for c in range(0, cols):
            board.data[r, c] = bool(rep[r][c])
            if not board.heights[c] and board.data[r, c]:
                board.heights[c] = r
    return board

#generates all possible placements given a starting board and an active zoid
def gen_futures(state, zoid, move_gen):
    return [state.future(zoid, *move) for move in move_gen(state, zoid)]

#grabs a random sample of episodes from an input file
def sample_tsv_lines(file_path, columns, sample_size):
    # Read the TSV file into a DataFrame
    df = pd.read_csv(file_path, sep='\t')
    
    # Sample rows
    sampled_rows = df.sample(n=sample_size, random_state=10) ### trying to test with the same samples TMH
    
    # Extract desired columns
    sampled_values = sampled_rows[columns]
    
    return sampled_values

#define feature weights
straightdrop = {
	features.landing_height: -3.383,
	features.eroded_cells: -9.277,
	features.row_trans: -2.700,
	features.col_trans: -6.786,
	features.pits: -12.668,
	features.cuml_wells: -0.396}

# Example usage
file_path = 'HumanData/startboards.tsv' 
#file_path = '../HumanData/startboards.tsv' ########## If this breaks things for you revert back to line above. TMH 0523
#grab only useful information from input file
#curr_zoid - letter value of current piece shape
#next_zoid - letter value of upcoming piece shape (I don't think we use this for anything)
#zoid_rot - numerical value designating the orientation of the zoid's final placement, can be between 0 and 3, depending on the zoid
#zoid_row - numerical value designating the placement of the zoid in vertical space
#zoid_col - numerical value designating the placement of the zoid in horizontal space
###important to note, row and col denote one cell of the zoid, presumably either the top left cell, or the bottom right. There have been inconsistencies before, so we should double check what is in use here
#start_board - a json string/array, 10x20 grid, 0 being an empty cell, numberical values corresponding to cells filled by different zoids. Represents the board at the beginning of an episode (but includes full lines from previous move, and needs to be cleared before handing to the model)
#adding a new one:
#board_rep - a json array representing the board at the end of the human's move. Includes active piece as placed by the human, and full lines are not yet cleared. 
#start_board (after clearing) and board_rep should differ by exactly four filled cells, representing the human's placement
columns_of_interest = ['curr_zoid', 'next_zoid', 'zoid_rot', 'zoid_row', 'zoid_col','start_board','board_rep']  # Specify the columns you want to extract
sample_size = 5  # Number of lines to sample

sampled_data = sample_tsv_lines(file_path, columns_of_interest, sample_size)

#creates an empty state. This can probably go inside the for loop, now that we don't need to keep history around in the state object
state = State(None, Board(20, 10))

all_zoids = {
        'I': zoids.classic.I,
        'T': zoids.classic.T,
        'L': zoids.classic.L,
        'J': zoids.classic.J,
        'O': zoids.classic.O,
        'Z': zoids.classic.Z,
        'S': zoids.classic.S
    }
print(sampled_data)

for index in range(sample_size):
    zoid = all_zoids[sampled_data.iloc[index, 0]]
    rot = sampled_data.iloc[index, 2]
    row = sampled_data.iloc[index, 3]
    col = sampled_data.iloc[index, 4]
    #print(rot)
    #print(row)
    #print(col)

    #human_board = state.future(zoid, rot, 20-row, col) ## This does not do anything? It doesn't, and actually appears to do a wrong thing (might be the issue with having different reference points for the zoid)
   
    #grabs the json string, should be a long string object
    extracted_board = sampled_data.iloc[index,5]
    #removes a single quote and the beginning and end of the string
    clipped_board = extracted_board[1:-1]

    #becomes a json object, should be an array of 20 strings of length 10
    json_board = json.loads(clipped_board)
    
    #converts to a board object, full cells true, empty cells false (note for future, this loses information about what each cell used to be, which is important for playback, but the true/false representation is smaller and faster for computation)
    start_board = parse_board(json_board)
    #print("Model Board:")
    #print(model_board.data) TMH debug
    #Testing if we need to clear things TMH debug
    
    #because start board came from the end of a previous placement, if that placement created any full lines, we need to remove them
    start_board.clear(start_board.full())
    #print("Modified Model Board:") TMH debug
    #print(model_board.data)TMH debug

    #load the board into the state object
    state.board = start_board
    print(state.board)
    #defines the feature values that will be used to evaluate each move
    feats = straightdrop

    #generates a list of states, each being one possible (as in, doesn't force an end to the game) moves
    possible = gen_futures(state, zoid, simulator.move_drop)


    #possible.sort(key=lambda x: -x[0])
    #scores, possible = zip(*possible)

    #if any moves are returned do the evaluation, probably need a case for how to record if the answer is zero (there shouldn't be any last moves in the data at all, but I think I had a few cases in testing, so we'll need to double check what is going on there)
    if len(possible) != 0:
        #print(possible[0].delta.zoid)

       ## I added a try except so that the loop just continues after error TMH
        try:
            #this print statement will give you, in order, the calculated value for each feature for the first item in the list of possible moves
            #note that the list is not yet sorted, so this is just the first one generated
            #the values are calculated by getting the raw feature number (so if there are three empty, inaccessible cells, the number of pits is 3) and multiplying it by the feature weight, defined previously
            #There are six features, and six values, in order. In the case I just debugged, the second value is 0, which makes sense, because that is the eroded cells feature. This move apparently doesn't clear any lines, so the raw number is zero
            #All six numbers added together form the move score
            print(feature.evaluate(possible[0], feats).values())
       
            #and in fact, that's exactly what happens here. For every move in the list of possible, calculate the move score
            possible = [(sum(feature.evaluate(s, feats).values()), s) for s in possible]
            #and now put them in order: highest score (often in our case, least negative) at the top
            possible.sort(key=lambda x: -x[0])
            #I'm not 100% sure what this does, but seems to create tuple objects of each state attached to its score, and it's useful for accessing that information later
            scores, possible = zip(*possible)
            #here, you are printing out the board associated with the first item in the (now sorted) list, which would be the move that the model would make. This board includes the cells that are being filled by the new zoid placement, but any full lines have not yet been cleared         
            print(possible[0].board)
            
            #so now, one way to potentially look for exact matches is to compare the possible[0].board against a variable that holds a parsed and loaded 'board_rep' - I think this is how my previous student did things
            #if it's not a match, or even if it is, we'll also want to look at the full list of scores
            #how long the list is, and what the spread of scores is at each decision point
            #in the case of a non-match, we want to know what rank order move the human did choose
            #there are rare cases where the human move is not on the list, usually because of a spin or tuck situation that the model can't see. In those cases, we have historically judged the human's choice to be better than the model

        except:
            print("HOHOHO an error occured, and was ignored")
            ## can add functionallity to log variable states for thrown errors.
            pass
