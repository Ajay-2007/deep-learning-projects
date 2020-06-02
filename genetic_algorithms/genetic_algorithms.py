import deap
from deap import base
from deap import tools
import random

toolbox = base.Toolbox()

toolbox.register('select', tools.selTournament, tournsize=3)
toolbox.register('crossover', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutFlipBit, indpb=0.02)

random_list = tools.initRepeat(list, random.random, 30)
# print(random_list)

def zero_or_one():
    return random.randint(0, 1)

random_list = tools.initRepeat(list, zero_or_one, 30)
# print(random_list)

toolbox.register('zero_or_one', random.randint, 0, 1)
random_list = tools.initRepeat(list, toolbox.zero_or_one, 30)
# print(random_list)

toolbox.register('random_number', random.random)
random_list = tools.initRepeat(list, toolbox.random_number, 30)
# print(random_list)

def some_fitness_calculation_function(individual):
    return toolbox.selection(individual)

toolbox.register('evaluate', some_fitness_calculation_function)

