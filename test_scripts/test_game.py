# Starting Simulator

# Generic imports
import random

# From Simulator Specific Code
from cogworks.tetris.game import State, Board, zoids

from cogworks.tetris import features, simulator

from cogworks import feature

# testfeatures = {
#     features.landing_height: -3.383,
#     features.eroded_cells: -9.277,
#     features.row_trans: -2.700,
#     features.col_trans: -6.786,
#     features.pits: -12.668,
#     features.cuml_wells: -0.396
# }

# FEATURES OF DIFFERENTIAL
# testfeatures = {
#     features.landing_height: -434.919,
#     features.eroded_cells: -832.277,
#     features.row_trans: -274.661,
#     features.col_trans: -429.168,
#     features.pits: -1152.084,
#     features.cuml_wells: -38.673
# }

# FEATURES OF NORMAL GENETIC
testfeatures = {
    features.landing_height: -552.960,
    features.eroded_cells: -630.324,
    features.row_trans: -283.080,
    features.col_trans: -1400.750,
    features.pits: -828.032,
    features.cuml_wells: -39.970
}

## Create a piece generator (can use a list here, or a generator function)

# True Random Generator Function
def zoid_gen(zoids, rng):
    while True:
        yield rng.choice(zoids)

## Create a move generator (this is where overhang detection or time pressure filtering are implemented. move_drop is basic version)

move_gen = simulator.move_drop

## Define Features

feats = testfeatures

## Define Initial State

state = State(None, Board(20, 10))

## Pick a seed (if using)

seed = 101

## Create simulator "object"

sim = simulator.simulate(
    state,
    zoid_gen(zoids.classic, random.Random(seed)),
    move_gen,
    simulator.policy_best(
        lambda state: sum(feature.evaluate(state, feats).values()),
        random.Random(-seed).choice
    )
)

## Run the simulator
# Inside this loop, print episode level data to file (take a good look at state and delta classes in game.py)

for (episode, state) in enumerate(sim, 1):
    print(episode, state.delta.zoid)
    print(state.board)
    print()

    # set some kind of limit
    if episode >= 525:
        break

print(state.score())
print(state.lines_cleared(1))
print(state.lines_cleared(2))
print(state.lines_cleared(3))
print(state.lines_cleared(4))
