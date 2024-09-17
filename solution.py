import numpy as np
import csv

def load_data(file_path):
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        for i in range(len(layers) - 1):
            weight = np.random.normal(0, 0.01, (layers[i], layers[i + 1]))
            bias = np.zeros((1, layers[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        self.a = [X]
        for i in range(len(self.weights) - 1):
            z = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
            self.a.append(a)
        z = np.dot(self.a[-1], self.weights[-1]) + self.biases[-1]
        self.a.append(z)
        return z
    
    def compute_mse(self, y_true, y_pred):
        N = y_pred.shape[0]
        return np.mean((y_true - y_pred) ** 2 / N)

class GeneticAlgorithm:
    def __init__(self, nn_config, pop_size, elitism, mutation_prob, mutation_scale, iterations):
        self.nn_config = nn_config
        self.pop_size = pop_size
        self.elitism = elitism
        self.mutation_prob = mutation_prob
        self.mutation_scale = mutation_scale
        self.iterations = iterations
        self.population = [NeuralNetwork(nn_config) for _ in range(pop_size)]
    
    def fitness(self, nn, X, y):
        y_pred = nn.forward(X)
        return -nn.compute_mse(y, y_pred)
    
    def select_parents(self, fitnesses):
        probs = fitnesses / np.sum(fitnesses)
        return np.random.choice(self.population, size=self.pop_size, p=probs)
    
    def crossover(self, parent1, parent2):
        child = NeuralNetwork(self.nn_config)
        for i in range(len(child.weights)):
            child.weights[i] = (parent1.weights[i] + parent2.weights[i]) / 2
            child.biases[i] = (parent1.biases[i] + parent2.biases[i]) / 2
        return child
    
    def mutate(self, nn):
        for i in range(len(nn.weights)):
            if np.random.rand() < self.mutation_prob:
                nn.weights[i] += np.random.normal(0, self.mutation_scale, nn.weights[i].shape)
                nn.biases[i] += np.random.normal(0, self.mutation_scale, nn.biases[i].shape)
    
    def evolve(self, X_train, y_train, X_test, y_test):
        for iteration in range(self.iterations):
            fitnesses = np.array([self.fitness(nn, X_train, y_train) for nn in self.population])
            best_index = np.argmax(fitnesses)
            if iteration % 2000 == 0:
                print(f"[Train error @ {iteration}]: {-fitnesses[best_index]}")
            new_population = [self.population[best_index]]
            parents = self.select_parents(fitnesses)
            for _ in range(self.pop_size - self.elitism):
                parent1, parent2 = np.random.choice(parents, 2)
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)
            self.population = new_population
        best_nn = self.population[np.argmax(fitnesses)]
        test_error = -self.fitness(best_nn, X_test, y_test)
        print(f"[Test error]: {test_error}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--test', type=str, required=True)
    parser.add_argument('--nn', type=str, required=True)
    parser.add_argument('--popsize', type=int, required=True)
    parser.add_argument('--elitism', type=int, required=True)
    parser.add_argument('--p', type=float, required=True)
    parser.add_argument('--K', type=float, required=True)
    parser.add_argument('--iter', type=int, required=True)
    
    args = parser.parse_args()
    
    X_train, y_train = load_data(args.train)
    X_test, y_test = load_data(args.test)
    
    nn_config = [X_train.shape[1]] + [int(layer) for layer in args.nn if layer.isdigit()] + [1]
    
    ga = GeneticAlgorithm(nn_config, args.popsize, args.elitism, args.p, args.K, args.iter)
    ga.evolve(X_train, y_train, X_test, y_test)