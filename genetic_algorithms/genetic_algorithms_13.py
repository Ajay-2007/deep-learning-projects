import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from deap import base
from deap import tools
from deap import creator

import elitism
import graphs

HARD_CONSTRAINT_PENALTY = 10
MAX_COLORS = 5

POPULATION_SIZE = 100
MAX_GENERATIONS = 100
P_CROSSOVER = 0.9
P_MUTATION = 0.1
HALL_OF_FAME_SIZE = 5

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# gcp = graphs.GraphColoringProblem(nx.petersen_graph(), HARD_CONSTRAINT_PENALTY)
gcp = graphs.GraphColoringProblem(nx.mycielski_graph(5), hard_constraint_penalty=HARD_CONSTRAINT_PENALTY)
toolbox = base.Toolbox()

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox.register("Integers", random.randint, 0, MAX_COLORS - 1)
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.Integers, len(gcp))
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


def getCost(individual):
    return gcp.get_cost(individual), # return a tuple


toolbox.register("evaluate", getCost)

toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=MAX_COLORS - 1, indpb=1.0 / len(gcp))


def main():
    population = toolbox.populationCreator(n=POPULATION_SIZE)
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    population, logbook = elitism.eaSimpleWithElitism(population,
                                                      toolbox=toolbox,
                                                      cxpb=P_CROSSOVER,
                                                      mutpb=P_MUTATION,
                                                      ngen=MAX_GENERATIONS,
                                                      stats=stats,
                                                      halloffame=hof,
                                                      verbose=True)

    best = hof.items[0]
    print("-- Best Individual = ", best)
    print("-- Best Fitness = ", best.fitness.values[0])
    print()
    print("Number of colors = ", gcp.get_number_of_colors(best))
    print("Number of violations = ", gcp.get_violations_count(best))
    print("Cost = ", gcp.get_cost(best))

    # plot the best solution

    plt.figure(1)
    gcp.plot_graph(best)

    # extract statistics
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

    # plot statistics:
    plt.figure(2)
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color="red")
    plt.plot(meanFitnessValues, color="green")
    plt.xlabel("Generation")
    plt.ylabel("Min / Avg Fitness")
    plt.title("Min and Average fitness over Generations")
    plt.show()


if __name__ == "__main__":
    main()
