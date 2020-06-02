from deap import base
from deap import tools
from deap import creator

import zoo
import random

NUM_OF_FEATURES = 15
NUM_OF_SAMPLES = 60
RANDOM_SEED = 42
FEATURE_PENALTY_FACTOR = 0.001

zoo = zoo.Zoo(RANDOM_SEED)

toolbox = base.Toolbox()

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox.register("zeroOrOne", random.randint, 0, 1)
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, len(zoo))
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


def zooClassificationAccuracy(individual):
    num_featured_used = sum(individual)
    if num_featured_used == 0:
        return 0.0
    else:
        accuracy = zoo.get_mean_accuracy(individual)
        return accuracy - FEATURE_PENALTY_FACTOR * num_featured_used, # return a tuple



toolbox.register("evaluate", zooClassificationAccuracy)

toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / len(zoo))

import numpy as np
from elitism import eaSimpleWithElitism
import matplotlib.pyplot as plt
import seaborn as sns

POPULATION_SIZE = 50
HALL_OF_FAME_SIZE = 5
GENERATIONS_MAX = 50
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

    print("-- Best Solutions are:")
    for i in range(HALL_OF_FAME_SIZE):
        print(i, ": ", hof.items[i], ", fitness = ", hof.items[0].fitness.values[0],
              ", accuracy = ", zoo.get_mean_accuracy(hof.items[i]),
              ", features = ", sum(hof.items[i]))


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
