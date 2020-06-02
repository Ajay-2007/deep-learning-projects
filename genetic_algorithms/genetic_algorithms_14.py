from deap import base
from deap import tools
from deap import creator
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import elitism

DIMENSIONS = 2  # number of dimensions
BOUND_LOW, BOUND_UP = -512.0, 512.0 # boundaries, same for all dimensions
CROWDING_FACTOR = 20.0


POPULATION_SIZE = 300
HALL_OF_FAME_SIZE = 30
MAX_GENERATIONS = 300
P_CROSSOVER = 0.9
P_MUTATION = 0.5


RANDOM_SEED = 42
random.seed(RANDOM_SEED)

toolbox = base.Toolbox()

def randomFloat(low, up):
    return [random.uniform(l, u) for l, u in zip([low] * DIMENSIONS, [up] * DIMENSIONS)]

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox.register("attrFloat", randomFloat, BOUND_LOW, BOUND_UP)
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.attrFloat)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

def egg_holder(individual):
    x = individual[0]
    y = individual[1]
    f = (-(y + 47.0) * np.sin(np.sqrt(abs(x/2.0 + (y + 47.0)))) - x * np.sin(np.sqrt(abs(x - (y + 47.0)))))
    return f, # return a tuple

toolbox.register("evaluate", egg_holder)

# Genetic operators:
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR,
                 indpb=1.0/DIMENSIONS)

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
    print("-- Best Fitness = ", best.fitness.values[0])

    sns.set_style("whitegrid")

    plt.plot(minFitnessValues, color="red")
    plt.plot(meanFitnessValues, color="green")
    plt.xlabel("Generation")
    plt.ylabel("Min / Average Fitness")
    plt.title("Min and Average Fitness over Generations")

    plt.show()

if __name__ == "__main__":
    main()
















