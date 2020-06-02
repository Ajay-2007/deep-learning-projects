from deap import base
from deap import tools
from deap import creator
import random
import elitism
import mountain_car
import numpy as np

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

MAX_GENERATIONS = 80
POPULATION_SIZE = 100
P_CROSSOVER = 0.9
P_MUTATION = 0.5
HALL_OF_FAME_SIZE = 20


car = mountain_car.MountainCar(RANDOM_SEED)

toolbox = base.Toolbox()


creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness=creator.FitnessMin)

toolbox.register('zeroOneOrTwo', random.randint, 0, 2)
toolbox.register('individualCreator', tools.initRepeat, creator.Individual, toolbox.zeroOneOrTwo, len(car))
toolbox.register('populationCreator', tools.initRepeat, list, toolbox.individualCreator)


def car_score(individual):
    return car.get_score(individual),

toolbox.register('evaluate', car_score)

toolbox.register('select', tools.selTournament, tournsize=2)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutUniformInt, low=0, up=2, indpb=1.0/len(car))


def main():
    population = toolbox.populationCreator(n=POPULATION_SIZE)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('min', np.min)
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
    print('Best Fitness = ', best.fitness.values[0])
    car.save_actions(best)


if __name__ == '__main__':
    main()