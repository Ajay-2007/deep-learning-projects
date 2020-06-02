import math

from deap import base
from deap import tools
from deap import creator
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import elitism

DIMENSIONS = 2  # number of dimensions
BOUND_LOW, BOUND_UP = -5.0, 5.0  # boundaries, same for all dimensions
CROWDING_FACTOR = 20.0

POPULATION_SIZE = 300
HALL_OF_FAME_SIZE = 30
MAX_GENERATIONS = 300
P_CROSSOVER = 0.9
P_MUTATION = 0.5
DISTANCE_THRESHOLD = 0.1
SHARING_EXTENT = 5.0

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

toolbox = base.Toolbox()


def randomFloat(low, up):
    return [random.uniform(l, u) for l, u in zip([low] * DIMENSIONS, [up] * DIMENSIONS)]


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox.register("attrFloat", randomFloat, BOUND_LOW, BOUND_UP)
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.attrFloat)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


def himmelblau_inverted(individual):
    x = individual[0]
    y = individual[1]
    f = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    return 2000.0 - f,  # return a tuple


toolbox.register("evaluate", himmelblau_inverted)


# Genetic operators:
def selTournamentWithSharing(individuals, k, tournsize, fit_attr="fitness"):
    # get original fitnesses:
    orig_fitnesses = [ind.fitness.values[0] for ind in individuals]

    # apply sharing to each individual:
    for i in range(len(individuals)):
        sharing_sum = 1

        # iterate over all individuals
        for j in range(len(individuals)):
            if i != j:
                # calculate euclidean distance between individuals:
                distance = math.sqrt(
                    ((individuals[i][0] - individuals[j][0]) ** 2) + ((individuals[i][1] - individuals[j][1]) ** 2)
                )

                if distance < DISTANCE_THRESHOLD:
                    sharing_sum += (1 - distance / (SHARING_EXTENT * DISTANCE_THRESHOLD))

        # reduce fitness accordingly:
        individuals[i].fitness.values = orig_fitnesses[i] / sharing_sum,

    # apply original tools.selTournament() using modified fitness:
    selected = tools.selTournament(individuals, k, tournsize, fit_attr)

    # retrieve original fitness:
    for i, ind in enumerate(individuals):
        ind.fitness.values = orig_fitnesses[i],

    return selected


toolbox.register("select", selTournamentWithSharing, tournsize=2)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR,
                 indpb=1.0 / DIMENSIONS)


def main():
    population = toolbox.populationCreator(n=POPULATION_SIZE)
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    population, logbook = elitism.eaSimpleWithElitism(population=population,
                                                      toolbox=toolbox,
                                                      cxpb=P_CROSSOVER,
                                                      mutpb=P_MUTATION,
                                                      ngen=MAX_GENERATIONS,
                                                      stats=stats,
                                                      halloffame=hof,
                                                      verbose=True)

    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")
    best = hof.items[0]

    print("-- Best Individual = ", best)
    print("-- Best solutions are: ")
    for i in range(HALL_OF_FAME_SIZE):
        print(i, ": ", hof.items[i].fitness.values[0], " --> ", hof.items[i])

    sns.set_style("whitegrid")

    plt.figure(1)
    global_maxima = [[3.0, 2.0], [-2.805118, 3.131312], [-3.779310,
                                                         -3.283186], [3.584458, -1.848126]]
    plt.scatter(*zip(*global_maxima), marker="X", color="red", zorder=1)
    plt.scatter(*zip(*population), marker=".", color="blue", zorder=0)

    plt.figure(2)
    plt.scatter(*zip(*global_maxima), marker="X", color="red", zorder=1)
    plt.scatter(*zip(*hof.items), marker=".", color="blue", zorder=0)
    plt.figure(3)
    plt.plot(minFitnessValues, color="red")
    plt.plot(meanFitnessValues, color="green")
    plt.xlabel("Generation")
    plt.ylabel("Min / Average Fitness")
    plt.title("Min and Average Fitness over Generations")

    plt.show()


if __name__ == "__main__":
    main()
