from deap import base
from deap import tools
from deap import creator
import random
import numpy as np
import cart_pole
import elitism

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

BOUNDS_LOW, BOUNDS_HIGH = -1.0, 1.0

CROWDING_FACTOR = 10.0
POPULATION_SIZE = 20
MAX_GENERATIONS = 10
P_CROSSOVER = 0.9
P_MUTATION = 0.5
HALL_OF_FAME_SIZE = 3

cart = cart_pole.CartPole(RANDOM_SEED)
NUM_OF_PARAMS = len(cart)
toolbox = base.Toolbox()

creator.create('FitnessMax', base.Fitness, weights=(1.0, ))
creator.create('Individual', list, fitness=creator.FitnessMax)


def randomFloat(low, up):
    return [random.uniform(l, u) for l, u in zip([low] * NUM_OF_PARAMS, [up] * NUM_OF_PARAMS)]


toolbox.register('attrFloat', randomFloat, BOUNDS_LOW, BOUNDS_HIGH)
toolbox.register('individualCreator', tools.initIterate, creator.Individual, toolbox.attrFloat)
toolbox.register('populationCreator', tools.initRepeat, list, toolbox.individualCreator)


def score(individual):
    return cart.get_score(individual),


toolbox.register('evaluate', score)

toolbox.register('select', tools.selTournament, tournsize=2)

toolbox.register('mate', tools.cxSimulatedBinaryBounded,
                 low=BOUNDS_LOW,
                 up=BOUNDS_HIGH,
                 eta=CROWDING_FACTOR)

toolbox.register('mutate',
                 tools.mutPolynomialBounded,
                 low=BOUNDS_LOW,
                 up=BOUNDS_HIGH,
                 eta=CROWDING_FACTOR,
                 indpb=1.0/NUM_OF_PARAMS)


def main():
    population = toolbox.populationCreator(n=POPULATION_SIZE)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('max', np.max)
    stats.register('avg', np.mean)
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    population, logbook = elitism.eaSimpleWithElitism(population,
                                                      toolbox,
                                                      cxpb=P_CROSSOVER,
                                                      mutpb=P_MUTATION,
                                                      ngen=MAX_GENERATIONS,
                                                      stats=stats,
                                                      halloffame=hof,
                                                      verbose=True)

    best = hof.items[0]
    print('Best Solution = ', best)
    print('Best Score = ', best.fitness.values[0])
    cart.saved_params(best)

    # find average score of 100 episodes using the best solution found:
    print('Running 100 episodes using the best solution...')
    scores = []
    for test in range(100):
        scores.append(cart_pole.CartPole().get_score(best))

    print('scores = ', scores)
    print('Avg. score = ', sum(scores) / len(scores))


if __name__ == '__main__':
    main()