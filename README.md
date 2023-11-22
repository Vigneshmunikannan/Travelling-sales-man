# Travelling-sales-man

# Algorithm
1. Initialize the population randomly.
2. Determine the fitness of the chromosome.
3. Until done repeat:
1. Select parents.
2. Perform crossover and mutation.
3. Calculate the fitness of the new population.
4. Append it to the gene pool

# Code
import numpy as np

class GeneticAlgorithmTSP:
    def __init__(self, num_cities, population_size, generations, crossover_rate, mutation_rate):
        self.num_cities = num_cities
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        self.cities = np.random.rand(num_cities, 2)  # Randomly generate city coordinates
        self.population = np.array([np.random.permutation(num_cities) for _ in range(population_size)])

    def calculate_distance(self, route):
        total_distance = 0
        for i in range(self.num_cities - 1):
            city1, city2 = route[i], route[i + 1]
            total_distance += np.linalg.norm(self.cities[city1] - self.cities[city2])
        return total_distance

    def evaluate_fitness(self, population):
        return np.array([1 / self.calculate_distance(route) for route in population])

    def selection(self, fitness):
        return np.random.choice(np.arange(self.population_size), size=self.population_size, p=fitness / fitness.sum())

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            crossover_point = np.random.randint(1, self.num_cities)
            child = np.hstack((parent1[:crossover_point], [city for city in parent2 if city not in parent1[:crossover_point]]))
        else:
            child = parent1.copy()
        return child

    def mutate(self, route):
        if np.random.rand() < self.mutation_rate:
            mutation_points = np.random.choice(np.arange(self.num_cities), size=2, replace=False)
            route[mutation_points[0]], route[mutation_points[1]] = route[mutation_points[1]], route[mutation_points[0]]
        return route

    def run_genetic_algorithm(self):
        for generation in range(self.generations):
            fitness = self.evaluate_fitness(self.population)
            selected_indices = self.selection(fitness)
            selected_population = self.population[selected_indices]

            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = selected_population[np.random.choice(np.arange(self.population_size), size=2, replace=False)]
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])

            self.population = np.array(new_population)

        best_route_index = np.argmax(fitness)
        best_route = self.population[best_route_index]
        best_distance = self.calculate_distance(best_route)

        return best_route, best_distance

#Example usage:
num_cities = 10
population_size = 100
generations = 500
crossover_rate = 0.8
mutation_rate = 0.02

tsp_solver = GeneticAlgorithmTSP(num_cities, population_size, generations, crossover_rate, mutation_rate)
best_route, best_distance = tsp_solver.run_genetic_algorithm()

print(f"Best Route: {best_route}")
print(f"Best Distance: {best_distance}")



# Link to run if this copy paste is not working
https://replit.com/@vigneshm2021csb/Travelling-sales-man-problem

# input
num_cities = 10
population_size = 100
generations = 500
crossover_rate = 0.8
mutation_rate = 0.02

# output
Best Route: [4 3 2 9 0 1 5 6 7 8]
Best Distance: 4.165813593826875
