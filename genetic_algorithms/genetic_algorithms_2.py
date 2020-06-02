from deap import tools
from deap import base
from deap import creator
import matplotlib.pyplot as plt
import random

# problem constants:

ONE_MAX_LENGTH = 100    # length of bit string to be optimized

# Genetic Algorithm constants:
POPULATION_SIZE = 200   # number of individuals in population
P_CROSSOVER = 0.9       # probability for crossover
P_MUTATION = 0.1        # probability for mutating an individual
MAX_GENERATIONS = 50    # max number of generations for stopping condition
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


toolbox = base.Toolbox()

toolbox.register('zero_or_one', random.randint, 0, 1)
creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', list, fitness=creator.FitnessMax)

toolbox.register('individual_creator', tools.initRepeat, creator.Individual, toolbox.zero_or_one, ONE_MAX_LENGTH)
toolbox.register('populationCreator', tools.initRepeat, list, toolbox.individual_creator)

def oneMaxFitness(individual):
    return sum(individual), # return a tuple

toolbox.register('evaluate', oneMaxFitness)

toolbox.register('select', tools.selTournament, tournsize=3)
toolbox.register('mate', tools.cxOnePoint)
toolbox.register('mutate', tools.mutFlipBit, indpb=1.0/ONE_MAX_LENGTH)

def main():
    population = toolbox.populationCreator(n=POPULATION_SIZE)
    generationCounter = 0

    fitnessValues = list(map(toolbox.evaluate, population))

    for individual, fitnessValue in zip(population, fitnessValues):
        individual.fitness.values = fitnessValue

    fitnessValues = [individual.fitness.values[0] for individual in population]
    maxFitnessValues = []
    meanFitnessValues = []

    while max(fitnessValues) < ONE_MAX_LENGTH and generationCounter < MAX_GENERATIONS:
        generationCounter += 1
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CROSSOVER:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < P_MUTATION:
                toolbox.mutate(mutant)
                del mutant.fitness.values


        freshIndividuals = [ind for ind in offspring if not ind.fitness.valid]
        freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))

        for individual, fitnessValue in zip(freshIndividuals, freshFitnessValues):
            individual.fitness.values = fitnessValue

        population[:] = offspring
        fitnessValues = [ind.fitness.values[0] for ind in population]

        maxFitness = max(fitnessValues)
        meanFitness = sum(fitnessValues) / len(population)
        maxFitnessValues.append(maxFitness)
        meanFitnessValues.append(meanFitness)
        print('- Generation {}: Max Fitness = {}, Avg Fitness = {}'
              .format(generationCounter, maxFitness, meanFitness))

        best_index = fitnessValues.index(max(fitnessValues))
        print('Best Individual = ', *population[best_index], '\n')




    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average fitness over Generations')
    plt.show()

if __name__ == '__main__':
    main()
