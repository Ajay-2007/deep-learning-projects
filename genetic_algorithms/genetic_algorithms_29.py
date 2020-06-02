from deap import base, gp
from deap import tools
from deap import creator

import random
import operator
import numpy as np
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import elitism

# population constants
NUM_INPUTS = 3
NUM_COMBINATIONS = 2 ** NUM_INPUTS

# Genetic Algorithm Constants:
POPULATION_SIZE = 60
P_CROSSOVER = 0.9
P_MUTATION = 0.5
MAX_GENERATIONS = 20
HALL_OF_FAME_SIZE = 10

# Genetic Programming specific constants:
MIN_TREE_HEIGHT = 3
MAX_TREE_HEIGHT = 5
LIMIT_TREE_HEIGHT = 17
MUT_MIN_TREE_HEIGHT = 0
MUT_MAX_TREE_HEIGHT = 2

# set the random seed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

toolbox = base.Toolbox()

# calculate the truth table for even parity check:
parity_in = list(itertools.product([0, 1], repeat=NUM_INPUTS))
parity_out = []

for row in parity_in:
    parity_out.append(sum(row) % 2)

# create the primitive set:
primitive_set = gp.PrimitiveSet("main", NUM_INPUTS, "in_")
primitive_set.addPrimitive(operator.and_, 2)
primitive_set.addPrimitive(operator.or_, 2)
primitive_set.addPrimitive(operator.xor, 2)
primitive_set.addPrimitive(operator.not_, 1)

# add terminal values
primitive_set.addTerminal(1)
primitive_set.addTerminal(0)

# define a single objective, minimizing fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# create the Individual class based on the primitive tree:
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# create a helper function for creating random trees using the primitive set:
toolbox.register("expr", gp.genFull, pset=primitive_set, min_=MIN_TREE_HEIGHT, max_=MAX_TREE_HEIGHT)

# create the individual operator to fill up an Individual instance:
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.expr)

# create the population operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

# create an operator to compile the primitive tree into python code:
toolbox.register("compile", gp.compile, pset=primitive_set)


# calculate the difference between the results of the generated function and the expected parity results:
def parity_error(individual):
    func = toolbox.compile(expr=individual)
    return sum(func(*p_in) != p_out for p_in, p_out in zip(parity_in, parity_out))


# fitness measure:
def get_cost(individual):
    return parity_error(individual) + (individual.height / 100) ,  # return a tuple


toolbox.register("evaluate", get_cost)

# genetic operators:
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genGrow, min_=MUT_MIN_TREE_HEIGHT, max_=MUT_MAX_TREE_HEIGHT)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=primitive_set)

# bloat control
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=LIMIT_TREE_HEIGHT))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=LIMIT_TREE_HEIGHT))


# Genetic Algorithm Flow:
def main():
    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # define the hall-of-fame object
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # perform the Genetic Algorithm flow with elitism
    population, logbook = elitism.eaSimpleWithElitism(population,
                                                      toolbox,
                                                      cxpb=P_CROSSOVER,
                                                      mutpb=P_MUTATION,
                                                      ngen=MAX_GENERATIONS,
                                                      stats=stats,
                                                      halloffame=hof,
                                                      verbose=True)

    # print info for best solution found:
    best = hof.items[0]
    print("-- Best Individual = ", best)
    print("-- length = {}, height = {}".format(len(best), best.height))
    print("-- Best Fitness = ", best.fitness.values[0])
    print("-- Best Parity Error = ", parity_error(best))

    # plot best tree
    nodes, edges, labels = gp.graph(best)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.spring_layout(g)

    nx.draw_networkx_nodes(g, pos, node_color="cyan")
    nx.draw_networkx_nodes(g, pos, nodelist=[0], node_color="red", node_size=400)

    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, **{"labels": labels, "font_size": 8})

    plt.show()


if __name__ == "__main__":
    main()
