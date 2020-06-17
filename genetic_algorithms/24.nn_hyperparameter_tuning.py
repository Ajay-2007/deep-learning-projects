from deap import base
from deap import tools
from deap import creator
import random
import elitism
import mlp_hyperparameters_test_2
import numpy as np
import warnings
# warnings.ignore()

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
# [layer_layer_1_size, hidden_layer_2_size, hidden_layer_3_size, hidden_layer_4_size]

# "hidden_layer_sizes" : first four values
# "activation": 0..2.99
# "solver": 0..2.99
# "alpha": 0.0001..2.0
# "learning_rate": 0..2.99
BOUNDS_LOW = [5, -5, -10, -20, 0,    0,    0.0001, 0]
BOUNDS_HIGH = [15, 10, 10, 10, 2.99, 2.99, 2.0,    2.99]

NUM_OF_PARAMS = len(BOUNDS_HIGH)
CROWDING_FACTOR = 10.0
P_CROSSOVER = 0.9
P_MUTATION = 0.5
HALL_OF_FAME_SIZE = 3
POPULATION_SIZE = 20
MAX_GENERATIONS = 5
test = mlp_hyperparameters_test_2.MLpHyperparametersTest(RANDOM_SEED)

toolbox = base.Toolbox()

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
for i in range(NUM_OF_PARAMS):
    toolbox.register("layer_size_attribute" + str(i),
                     random.uniform,
                     BOUNDS_LOW[i],
                     BOUNDS_HIGH[i])

layer_size_attributes = ()
for i in range(NUM_OF_PARAMS):
    layer_size_attributes = layer_size_attributes + (toolbox.__getattribute__("layer_size_attribute" + str(i)),)

toolbox.register("individualCreator", tools.initCycle, creator.Individual, layer_size_attributes, n=1)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


def classification_accuracy(individual):
    return test.get_accuracy(individual),


toolbox.register("evaluate", classification_accuracy)

toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                 low=BOUNDS_LOW,
                 up=BOUNDS_HIGH,
                 eta=CROWDING_FACTOR)
toolbox.register("mutate",
                 tools.mutPolynomialBounded,
                 low=BOUNDS_LOW,
                 up=BOUNDS_HIGH,
                 eta=CROWDING_FACTOR,
                 indpb=1.0 / NUM_OF_PARAMS)


def main():

    population = toolbox.populationCreator(n=POPULATION_SIZE)
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    elitism.eaSimpleWithElitism(population,
                                toolbox,
                                cxpb=P_CROSSOVER,
                                mutpb=P_MUTATION,
                                ngen=MAX_GENERATIONS,
                                stats=stats,
                                halloffame=hof,
                                verbose=True)

    print("-- Best solution is : hidden_layer_sizes = ", test.format_params(hof.items[0]), "accuracy = ", hof.items[0].fitness.values[0])


if __name__ == "__main__":
    main()
