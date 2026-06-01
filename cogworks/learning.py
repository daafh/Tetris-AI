from __future__ import division
from heapq import nlargest

import collections.abc
import numpy as np
import cma


def cross_entropy(feats, stdev, width, keep, rng, test_f, noise_f=None, map_f=map):
    # Interpret features
    if isinstance(feats, collections.abc.Mapping):
        # Weights provided
        feats, weights = zip(*feats.items())
        weights = list(weights)
    else:
        # No weights, default to zero
        feats = list(feats)
        weights = [0] * len(feats)

    # Interpret stdev
    if isinstance(stdev, collections.abc.Mapping):
        # Per-feature stdev
        stdev = [stdev[feat] for feat in feats]
    else:
        # Constant stdev
        stdev = [stdev] * len(feats)

    while True:
        if noise_f is not None:
            # Apply noise function
            stdev = [noise_f(s) for s in stdev]

        # Generate new weights around the mean
        generation = [
            [rng.gauss(weights[i], stdev[i]) for i in range(0, len(feats))]
            for _ in range(0, width)
        ]

        # Evaluate the weights
        results = map_f(lambda i: (test_f(dict(zip(feats, generation[i]))), i), range(0, width))

        # Collect top performers
        top_weights = list(zip(*(generation[i] for (_, i) in nlargest(keep, results))))

        # Compute mean weights and new deviations
        for i in range(0, len(feats)):
            weights[i] = np.mean(top_weights[i])
            stdev[i] = np.std(top_weights[i])

        # Reconstruct weight and stdev feature maps
        yield collections.OrderedDict(zip(feats, weights)), collections.OrderedDict(zip(feats, stdev))


def genetic_algorithm(feats, stdev, width, keep, rng, test_f, noise_f=None, map_f=map):
    # Interpret features
    if isinstance(feats, collections.abc.Mapping):
        # Weights provided
        feats, weights = zip(*feats.items())
        weights = list(weights)
    else:
        # No weights, default to zero
        feats = list(feats)
        weights = [0] * len(feats)

    # Interpret stdev
    if isinstance(stdev, collections.abc.Mapping):
        # Per-feature stdev
        stdev = [stdev[feat] for feat in feats]
    else:
        # Constant stdev
        stdev = [stdev] * len(feats)

    # Populate new generation
    population = [
        [rng.gauss(weights[i], stdev[i]) for i in range(len(feats))]
        for _ in range(width)
    ]

    while True:
        if noise_f is not None:
            # Apply noise function
            stdev = [noise_f(s) for s in stdev]

        # Evaluate the weights
        results = map_f(lambda i: (test_f(dict(zip(feats, population[i]))), i), range(width))
        results = sorted(results, reverse=True)

        # Get indices of #keep amount of best performers
        top_indices = [i for (_, i) in results[:keep]]
        top_performers = [population[i] for i in top_indices]

        # Create a new population and keep the top performers
        new_pop = []
        new_pop.extend(top_performers)

        # Populate the rest with mutations of top performers
        while len(new_pop) < width:
            p1 = top_performers[rng.randint(0, keep - 1)]
            p2 = top_performers[rng.randint(0, keep - 1)]

            child = []
            for i in range(len(feats)):
                a = rng.random()

                feat_gene = a * p1[i] + (1 - a) * p2[i]

                feat_gene += rng.gauss(0, stdev[i])
                child.append(feat_gene)

            new_pop.append(child)

        population = new_pop

        # Compute mean weights and new deviations
        for i in range(len(feats)):
            feature_values = [individual[i] for individual in top_performers]
            weights[i] = np.mean(feature_values)
            stdev[i] = np.std(feature_values)

        yield (
            collections.OrderedDict(zip(feats, weights)),
            collections.OrderedDict(zip(feats, stdev))
        )


def particle_swarm_optimization(feats, stdev, width, rng, test_f, noise_f=None, map_f=map):
    np_rng = np.random.default_rng(rng)

    particles = np_rng.normal(0, stdev, (width, len(feats)))
    velocities = np.zeros(np.shape(particles))

    best_positions = np.copy(particles)
    best_scores = np.array([test_f(dict(zip(feats, particles[i]))) for i in range(width)])

    swarm_best_position = best_positions[np.argmax(best_scores)]

    w = 0.2
    c1 = 1
    c2 = 2

    # Interpret stdev
    if isinstance(stdev, collections.abc.Mapping):
        # Per-feature stdev
        stdev = [stdev[feat] for feat in feats]
    else:
        # Constant stdev
        stdev = [stdev] * len(feats)

    while True:
        if noise_f is not None:
            # Apply noise function
            stdev = [noise_f(s) for s in stdev]

        r1, r2 = np_rng.random(2)

        velocities = np.array(
            w * velocities +
            c1 * r1 * (best_positions - particles) +
            c2 * r2 * (swarm_best_position - particles)
        )
        particles += velocities

        for i in range(width):
            score = test_f(dict(zip(feats, particles[i])))
            if score > best_scores[i]:
                best_scores[i] = score
                best_positions[i] = particles[i]

        swarm_best_position = best_positions[np.argmax(best_scores)]

        stdev = np.std(particles, axis=0)
        weights = np.mean(particles, axis=0)

        yield (
            collections.OrderedDict(zip(feats, weights)),
            collections.OrderedDict(zip(feats, stdev))
        )


def differential_evolution(feats, stdev, width, rng, test_f, noise_f=None, map_f=map):
    np_rng = np.random.default_rng(rng)

    population = np_rng.normal(0, stdev, (width, len(feats)))
    fitnesses = np.array([test_f(dict(zip(feats, population[i]))) for i in range(width)])

    # Interpret stdev
    if isinstance(stdev, collections.abc.Mapping):
        # Per-feature stdev
        stdev = [stdev[feat] for feat in feats]
    else:
        # Constant stdev
        stdev = [stdev] * len(feats)

    while True:
        if noise_f is not None:
            # Apply noise function
            stdev = [noise_f(s) for s in stdev]

        for i in range(width):
            candidates = [candidate for candidate in range(width) if candidate != i]
            # Mutation part of DE
            a, b, c = population[np_rng.choice(candidates, 3, replace=False)]
            mutant = a + 0.8 * (b - c)

            # Crossover part of DE
            cr = 0.7  # crossover rate
            trial = np.copy(population[i])

            for j in range(len(feats)):
                if np_rng.random() < cr:
                    trial[j] = mutant[j]

            trial_fitness = test_f(dict(zip(feats, trial)))

            if trial_fitness > fitnesses[i]:
                fitnesses[i] = trial_fitness
                population[i] = trial

        stdev = np.std(population, axis=0)
        weights = np.mean(population, axis=0)

        yield (
            collections.OrderedDict(zip(feats, weights)),
            collections.OrderedDict(zip(feats, stdev))
        )


def cma_es(feats, stdev, pop_size, rng, test_f):
    # Interpret features
    if isinstance(feats, collections.abc.Mapping):
        # Weights provided
        feats, weights = zip(*feats.items())
        weights = list(weights)
    else:
        # No weights, default to zero
        feats = list(feats)
        weights = [0] * len(feats)

    # Interpret stdev
    if isinstance(stdev, collections.abc.Mapping):
        # Per-feature stdev
        sigma = np.mean([stdev[feat] for feat in feats])
    else:
        # Constant stdev
        sigma = np.mean([stdev] * len(feats))

    es = cma.CMAEvolutionStrategy(
        weights,
        sigma,
        {'popsize': pop_size}
    )

    while not es.stop():
        solutions = es.ask()
        es.tell(solutions, [-test_f(dict(zip(feats, solution))) for solution in solutions])

        mean = es.mean
        stdev = es.stds
        yield (
            collections.OrderedDict(zip(feats, mean)),
            collections.OrderedDict(zip(feats, stdev))
        )
