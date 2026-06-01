# Generic imports
import random
import matplotlib.pyplot as plt
import statistics

# From Simulator Specific Code
from cogworks.tetris.game import State, Board, zoids
from cogworks.tetris import features, simulator
from cogworks import feature, learning

seed = 0
local_seed = seed


def zoid_gen(zoids, rng):
	while True:
		yield rng.choice(zoids)


move_gen = simulator.move_drop


def test(feats):
	# if printing game information to a file, do it here
	state = State(None, Board(20, 10))
	sim = simulator.simulate(state, zoid_gen(zoids.classic, random.Random(seed)), move_gen, simulator.policy_best(lambda state: sum(feature.evaluate(state, feats).values()), random.Random(-seed).choice), lookahead = 1)

	for (episode, state) in enumerate(sim, 1):
		# some end condition
		if episode >= 525:
			break
		# if no end condition, [pass] inside for loop
	# return whatever value determines goodness, scores, lines cleared, etc
	# print(state.score())
	return state.score()


def evaluate(feats, n_runs=30) -> tuple[int, int]:
	"""
	Function used for evaluating a set of weights after training,
	and returns the mean score over 30 (or defined amount) of runs.

	Basically copies the test function, but changes the seed based on
	the number of the current game being played.
	"""
	local_seed = 100
	scores = []

	for run in range(n_runs):
		state = State(None, Board(20, 10))
		sim = simulator.simulate(
			state,
			zoid_gen(zoids.classic, random.Random(local_seed + run)),
			move_gen,
			simulator.policy_best(
				lambda state: sum(feature.evaluate(state, feats).values()),
				random.Random(-(local_seed + run)).choice
			),
			lookahead=1
		)

		for (episode, state) in enumerate(sim, 1):
			# some end condition
			if episode >= 525:
				break
		
		scores.append(state.score())

	mean_score = statistics.mean(scores)
	stdev_score = statistics.stdev(scores)

	return mean_score, stdev_score


# if starting at all zeroes, put features in a list, if features have actual values, define as a dictionary

feats = [	
	features.landing_height,
	features.eroded_cells,
	features.row_trans,
	features.col_trans,
	features.pits,
	features.cuml_wells
]

trainers = {
	"cem": learning.cross_entropy(feats, 10, 100, 10, random.Random(local_seed), test),
	"ga": learning.genetic_algorithm(feats, 10, 100, 10, random.Random(local_seed), test),
	"pso": learning.particle_swarm_optimization(feats, 10, 100, local_seed, test),
	"de": learning.differential_evolution(feats, 10, 100, local_seed, test),
	"cma-es": learning.cma_es(feats, 10, 100, local_seed, test)
}

# Store scores and final weights as dict entry of [name]: list[int | float]
scores_per_model = {}
final_weights = {}

for name, trainer in trainers.items():
	scores = []

	for (depth, (weights, stdev)) in enumerate(trainer, 1):
		seed += 1

		stable = True
		print('Iteration {:6d}: {:>9} {:>9}'.format(depth, 'mean', 'variation'))

		for feat in feats:
			var = abs(stdev[feat] / weights[feat])
			print('{:16}: {:> 9.3f} {:> 9.3}'.format(feat.__name__, weights[feat], var))
			if var > 0.1:
				stable = False

		mean, stdev = evaluate(weights)
		scores.append((mean, stdev))

		if stable:
			final_weights[name] = weights
			break
	
	scores_per_model[name] = scores

for name, weights in final_weights.items():
	print("\n", name)
	for feat in feats:
		print('{:16}: {:> 9.3f}'.format(feat.__name__, weights[feat]))

plt.figure()

for name, scores in scores_per_model.items():
	means = [m for m, s in scores]
	stdevs = [s for m, s in scores]
	lower = [m - s for m, s in zip(means, stdevs)]
	upper = [m + s for m, s in zip(means, stdevs)]

	generations = range(1, len(scores) + 1)
	plt.plot(generations, means, label=name)
	plt.fill_between(generations, lower, upper, alpha=0.2)

plt.xlabel("Generation")
plt.ylabel("Mean score")
plt.title("Learning Curve of Different Agents")

plt.legend()
plt.grid(True)
plt.show()
