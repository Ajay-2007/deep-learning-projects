from deap import base
from deap import tools
from deap import creator
import queens
import random
import array
import elitism
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from elitism_parallel_class import EaSimpleWithElitism

NUM_OF_QUEENS = 16
POPULATION_SIZE = 500
MAX_GENERATIONS = 1000
HALL_OF_FAME_SIZE = 300
P_CROSSOVER = 0.95
P_MUTATION = 0.15
RANDOM_SEED = 90
random.seed(RANDOM_SEED)

nQueens = queens.NQueensProblem(NUM_OF_QUEENS)
toolbox = base.Toolbox()

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)

toolbox.register("randomOrder", random.sample, range(len(nQueens)), len(nQueens))
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomOrder)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


def getViolationsCount(individual):
    return nQueens.getViolationsCount(individual), # return a tuple

toolbox.register("evaluate", getViolationsCount)

toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", tools.cxUniformPartialyMatched, indpb=2.0 / len(nQueens))
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=2.0 / len(nQueens))


def main():
    population = toolbox.populationCreator(n=POPULATION_SIZE)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    population, logbook = elitism.eaSimpleWithElitism(population,
                                                      toolbox,
                                                      cxpb=P_CROSSOVER,
                                                      mutpb=P_MUTATION,
                                                      ngen=MAX_GENERATIONS,
                                                      stats=stats,
                                                      halloffame=hof,
                                                      verbose=True
                                                      )
    #
    # easimple = EaSimpleWithElitism(population,
    #                               toolbox,
    #                               cxpb=P_CROSSOVER,
    #                               mutpb=P_MUTATION,
    #                               ngen=MAX_GENERATIONS,
    #                               stats=stats,
    #                               halloffame=hof,
    #                               verbose=True
    #                               )
    #
    # population, logbook = easimple.starting()
    print("-- Best solutions are: ")
    for i in range(HALL_OF_FAME_SIZE):
        print(i, ": ", hof.items[i].fitness.values[0], " --> ", hof.items[i])

    # plot statistics:
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")
    plt.figure(1)
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')

    # plot best solution:
    sns.set_style("whitegrid", {'axes.grid': False})
    nQueens.plotBoard(hof.items[0])

    # show both plots:
    plt.show()


if __name__ == "__main__":
    main()
    # main()
