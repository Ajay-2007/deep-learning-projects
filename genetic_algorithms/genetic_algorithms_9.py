import multiprocessing
from deap import base
from deap import creator
from deap import tools
import random
import array
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import elitism
# import tsp
import vrp
# set the random seed for repeatable results
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# create the desired travelling salesman problem instance:
TSP_NAME = "bayg29"
NUM_OF_VEHICLES = 6
DEPOT_INDEX = 12
vrp = vrp.VehicleRoutingProblem(TSP_NAME, NUM_OF_VEHICLES, DEPOT_INDEX)

# Genetic Algorithm Constants:
POPULATION_SIZE = 500
MAX_GENERATIONS = 1000
HALL_OF_FAME_SIZE = 30
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATE = 0.2  # probability for mutating an individual

toolbox = base.Toolbox()

# define a single objective, minimizing fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# create the Individual class based on the list of integers:
creator.create("Individual", array.array, typecode="i", fitness=creator.FitnessMin)

# create an operator that generates randomly shuffled indices:
toolbox.register("randomOrder", random.sample, range(len(vrp)), len(vrp))

# create the individual creation operator to fill up an Individual instance with shuffled indices:
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomOrder)

# create the population creation operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


# fitness calculation - compute the total distance of the list of cities represented by indices:
def vrpDistance(individual):
    return vrp.getMaxDistance(individual),  # return a tuple


toolbox.register("evaluate", vrpDistance)

# Genetic operators:
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", tools.cxUniformPartialyMatched, indpb=2.0/len(vrp))
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0 / len(vrp))


# Genetic Algorithm Flow:

def main():
    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # perform the Genetic Algorithm flow with hof feature added:
    population, logbook = elitism.eaSimpleWithElitism(population,
                                                      toolbox,
                                                      cxpb=P_CROSSOVER,
                                                      mutpb=P_MUTATE,
                                                      ngen=MAX_GENERATIONS,
                                                      stats=stats,
                                                      halloffame=hof,
                                                      verbose=True)

    # print best individual info:
    best = hof.items[0]
    print("-- Best Ever Individual = ", best)
    print("-- Best Ever Fitness = ", best.fitness.values[0])

    # plot best solution
    plt.figure(1)
    vrp.plotData(best)

    # plot statistics
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

    plt.figure(2)
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color="red")
    plt.plot(meanFitnessValues, color="green")
    plt.xlabel("Generation")
    plt.ylabel("Min / Average Fitness")
    plt.title("Min and Average fitness over Generations")

    # show both plots:
    plt.show()


if __name__ == "__main__":
    thread = multiprocessing.Process(target=main,
                                     name="thread")
    thread.start()
    thread.join()
    # main()
