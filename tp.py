import random
import operator
import math
import numpy as np
from deap import base, creator, tools, gp, algorithms

# Define the sequence
sequence = []

# Define the symbolic regression problem
def evaluate(individual):
    func = gp.compile(expr=individual, pset=pset)
    errors = [(func(i) - sequence[i])**2 for i in range(len(sequence))]
    return sum(errors),


# Define the primitive set
pset = gp.PrimitiveSet("MAIN", arity=1)
pset.addPrimitive(operator.add, arity=2)
pset.addPrimitive(operator.sub, arity=2)
pset.addPrimitive(operator.mul, arity=2)
pset.addPrimitive(operator.neg, arity=1)
pset.addPrimitive(math.cos, arity=1)
pset.addPrimitive(math.sin, arity=1)
pset.addPrimitive(math.exp, arity=1)
pset.addPrimitive(math.log, arity=1)
pset.addEphemeralConstant("rand", lambda: random.uniform(-10, 10))

# Define the fitness and individual
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# Initialize the toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the necessary operators and functions
toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Set up the statistics and hall of fame
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)
hall_of_fame = tools.HallOfFame(maxsize=1)

# Run the evolutionary algorithm
population = toolbox.population(n=100)
population, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, stats=stats, halloffame=hall_of_fame)

# Print the best individual (mathematical expression)
best_individual = hall_of_fame[0]
best_func = toolbox.compile(expr=best_individual)
print("Best Individual:", best_individual)
print("Best Expression:", best_func)

# Generate predictions for the next term
next_term = best_func(len(sequence))
print("Predicted Next Term:", next_term)

