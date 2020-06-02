from deap import base
from deap import tools
from deap import creator

import friedman
import random

NUM_OF_FEATURES = 15
NUM_OF_SAMPLES = 60
RANDOM_SEED = 42

friedman = friedman.FriedmanTest(NUM_OF_FEATURES, NUM_OF_SAMPLES, RANDOM_SEED)

toolbox = base.Toolbox()

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox.register("zeroOrOne", random.randint, 0, 1)
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, len(friedman))
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


def friedmanTestScore(individual):
    return friedman.get_mse(individual),


toolbox.register("evaluate", friedmanTestScore)

toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / len(friedman))

import numpy as np
from elitism import eaSimpleWithElitism
import matplotlib.pyplot as plt
import seaborn as sns

POPULATION_SIZE = 30
HALL_OF_FAME_SIZE = 5
GENERATIONS_MAX = 30
P_CROSSOVER = 0.9
P_MUTATION = 0.2


def main():
    population = toolbox.populationCreator(POPULATION_SIZE)
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.mean)
    stats.register("avg", np.mean)

    population, logbook = eaSimpleWithElitism(population,
                                              toolbox,
                                              cxpb=P_CROSSOVER,
                                              mutpb=P_MUTATION,
                                              ngen=GENERATIONS_MAX,
                                              stats=stats,
                                              halloffame=hof,
                                              verbose=True)

    best = hof.items[0]
    print("-- Best Ever Individual = ", best)
    print("-- Best Ever Fitness = ", best.fitness.values[0])

    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color="red")
    plt.plot(meanFitnessValues, color="green")
    plt.xlabel("Generation")
    plt.ylabel("Min / Average Fitness")
    plt.title("Min and Average fitness over Generations")
    plt.show()


if __name__ == "__main__":
    main()
