import os

from deap import base
from deap import tools
from deap import creator

import numpy as np
import random

import elitism_callback
import image_test
import matplotlib.pyplot as plt
import seaborn as sns

P_CROSSOVER = 0.9
P_MUTATION = 0.5
HALL_OF_FAME_SIZE = 20
POPULATION_SIZE = 200
MAX_GENERATIONS = 5000
CROWDING_FACTOR = 10.0

BOUNDS_LOW, BOUNDS_HIGH = 0.0, 1.0

POLYGON_SIZE = 3
NUM_OF_POLYGONS = 100
# two coordinates per vertex, 3 color values, one alpha value
NUM_OF_PARAMS = NUM_OF_POLYGONS * (POLYGON_SIZE * 2 + 4)
image_test = image_test.ImageTest("Mona_Lisa_head.png", POLYGON_SIZE)
toolbox = base.Toolbox()

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


def randomFloat(low, up):
    return [random.uniform(l, u) for l, u in zip([low] * NUM_OF_PARAMS, [up] * NUM_OF_PARAMS)]


toolbox.register("attrFloat", randomFloat, BOUNDS_LOW, BOUNDS_HIGH)

toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.attrFloat)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


def get_diff(individual):
    return image_test.get_difference(individual, "MSE"),
    # return image_test.get_difference(individual, "SSIM"),


toolbox.register("evaluate", get_diff)

toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUNDS_LOW, up=BOUNDS_HIGH, eta=CROWDING_FACTOR)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUNDS_LOW, up=BOUNDS_HIGH, eta=CROWDING_FACTOR,
                 indpb=1.0 / NUM_OF_PARAMS)


# save the best current drawing every 100 generations (used as a callback):

def save_image(gen, polygon_data):
    # only every 100 generations:
    if gen % 100 == 0:
        # create a folder if does not exist:
        folder = "images/results/run-{}-{}".format(POLYGON_SIZE, NUM_OF_POLYGONS)
        if not os.path.exists(folder):
            os.makedirs(folder)

        # save the image in the folder
        image_test.save_image(polygon_data,
                              "{}/after-{}-gen.png".format(folder, gen),
                              "After {} Generations".format(gen))


# Genetic Algorithm Flow:
def main():
    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    population, logbook = elitism_callback.eaSimpleWithElitismAndCallback(population,
                                                                          toolbox,
                                                                          cxpb=P_CROSSOVER,
                                                                          mutpb=P_MUTATION,
                                                                          ngen=MAX_GENERATIONS,
                                                                          stats=stats,
                                                                          halloffame=hof,
                                                                          callback=save_image,
                                                                          verbose=True)

    # print best solution found:
    best = hof.items[0]
    print()
    print("Best Solution = ", best)
    print("Best Score = ", best.fitness.values[0])
    print()

    # draw the image next to reference image:
    image_test.plot_image(image_test.polygon_data_to_image(best))
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

    sns.set_style("whitegrid")
    plt.figure("Stats")
    plt.plot(minFitnessValues, color="red")
    plt.plot(meanFitnessValues, color="green")
    plt.xlabel("Generation")
    plt.ylabel("Min / Average Fitness")
    plt.title("Min and Average fitness over Generations")
    plt.show()


if __name__ == "__main__":
    main()
