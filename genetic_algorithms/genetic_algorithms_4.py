import random
import numpy
from deap import tools
from deap import creator
from deap import algorithms
from deap import base
import matplotlib.pyplot as plt
# problem constants:

ONE_MAX_LENGTH = 100    # length of bit string to be optimized

# Genetic Algorithm constants:
POPULATION_SIZE = 200   # number of individuals in population
P_CROSSOVER = 0.9       # probability for crossover
P_MUTATION = 0.1        # probability for mutating an individual
MAX_GENERATIONS = 50    # max number of generations for stopping condition
RANDOM_SEED = 42
HALL_OF_FAME_SIZE = 10

random.seed(RANDOM_SEED)

toolbox = base.Toolbox()

toolbox.register('zero_or_one', random.randint, 0, 1)
creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', list, fitness=creator.FitnessMax)

toolbox.register('individual_creator', tools.initRepeat, creator.Individual, toolbox.zero_or_one, ONE_MAX_LENGTH)
toolbox.register('populationCreator', tools.initRepeat, list, toolbox.individual_creator)

def oneMaxFitness(individual):
    return sum(individual), # return a tuple

toolbox.register('evaluate', oneMaxFitness)

toolbox.register('select', tools.selTournament, tournsize=3)
toolbox.register('mate', tools.cxOnePoint)
toolbox.register('mutate', tools.mutFlipBit, indpb=1.0/ONE_MAX_LENGTH)

stats = tools.Statistics(lambda ind: ind.fitness.values)

stats.register('max', numpy.max)
stats.register('avg', numpy.mean)
population = toolbox.populationCreator(n=POPULATION_SIZE)

hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

population, logbook = algorithms.eaSimple(population,
                                          toolbox,
                                          cxpb=P_CROSSOVER,
                                          mutpb=P_MUTATION,
                                          ngen=MAX_GENERATIONS,
                                          stats=stats,
                                          halloffame=hof,
                                          verbose=True)

maxFitnessValues, meanFitnessValues = logbook.select('max', 'avg')

for item in hof.items:
    print('Hall of Fame Individuals = ', sum(item))
print('Best Ever Individual = ', hof.items[0])
plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Generation')
plt.ylabel('Max / Average Fitness')
plt.title('Max and Average fitness over Generations')
plt.show()