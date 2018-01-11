from cnn_load_data import *
from cnn_parameters import *
from cnn_core_model import *
from cnn_start import *
import random
class Optimizer(object):
    def __init__(self,random_select, retain, mutate_chance):
        self.parameter_choices = ["weight_minimum_limit" , "weight_maximum_limit", "learning_rate"]
        self.weight_matrix_min = [-0.07, -0.06, -0.05, -0.04, -0.03]
        self.weight_matrix_max = [0.07, 0.06, 0.05, 0.04, 0.03]
        self.learning_rate = [0.05,0.04,0.01,0.02,0.03]
        self.population_size = 5
        self.children_size = 2
        self.nn_param_choices = 3
        self.random_select = random_select
        self.retain = retain
        self.mutate_chance = mutate_chance
    """Create a population of neural networks"""
    def create_population(self):
        population = []
        index = 0
        for i in range(0, self.population_size):
            print ("Initializing candidate " , i)
            nn_object = Main()
            nn_object.initialize_data()
            nn_object.common_param.weight_minimum_limit = self.weight_matrix_min[index]
            nn_object.common_param.weight_maximum_limit = self.weight_matrix_max[index]
            nn_object.common_param.learning_rate = self.learning_rate[index]
            index += 1
            # Add the network to our population.
            population.append(nn_object)

        return population

    """Define the fitness function"""
    def fitness(self,network):
        return network.net.accuracy
    
    """Crossover the candidates to create new candidates"""
    def breed(self,father,mother):
        children = []
        """Make children as parts of their parents."""
        for i in range(self.children_size):
            child = Main()
            child.initialize_data()
            minimum_list = [father.common_param.weight_minimum_limit, mother.common_param.weight_minimum_limit]
            maximum_list = [father.common_param.weight_maximum_limit, mother.common_param.weight_maximum_limit]
            learning_rate_list = [father.common_param.learning_rate,mother.common_param.learning_rate]
            choice = random.choice(range(len(minimum_list)))
            child.common_param.learning_rate = learning_rate_list[choice]
            child.common_param.weight_minimum_limit = minimum_list[choice]
            choice ^= 1
            child.common_param.weight_maximum_limit = maximum_list[choice]
            
            if self.mutate_chance > random.random():
                network = self.mutate(child)
                child.net.initialize_layers()
                
            children.append(child)
            
        return children
            
    """Mutate the hyper parameters of the network"""
    def mutate(self,network):
        mutation = random.randint(0,self.nn_param_choices)
        if(mutation == 0):
            network.common_param.weight_minimum_limit = random.choice(self.weight_matrix_min)
        elif(mutation == 1):
            network.common_param.weight_maximum_limit = random.choice(self.weight_matrix_max)
        else:
            network.common_param.learning_rate = random.choice(self.learning_rate)
        return network

    """Evolve a population of networks"""
    def evolve(self,population):
        graded_population = [(self.fitness(network), network ) for network in population]
        graded_population = [x[1] for x in sorted(graded_population,key=lambda x: x[0], reverse=True)]

        """Population of winners to be retained"""
        retain_length = int(len(graded_population)*self.retain)
        parents = graded_population[:retain_length]

        """Some random loser population"""
        for individual in graded_population[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        """find out how many spots we have left to fill"""
        parents_length = len(parents)
        desired_length = len(population) - parents_length
        children = []

        """Adding children breed from existing parents"""
        while len(children) < desired_length:

            """Get a random father and mother"""
            male = random.randint(0, parents_length-1)
            female = random.randint(0, parents_length-1)

            """Checking whether if they are not same"""
            if male != female:
                male = parents[male]
                female = parents[female]

                """Breed them"""
                babies = self.breed(male, female)

                """Add the children one at a time"""
                for baby in babies:
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)

        return parents

def main():
    print ("Initialization")
    generations = 2
    start_time = time.time()
    optimizer = Optimizer(random_select = 0.1,retain = 0.6, mutate_chance = 0.1)
    networks = optimizer.create_population()
    for i in range(generations):

        average_accuracy = 0.0
        print("Training")
        """Train the networks"""
        for network in networks:
            if(network.trained_status != True):
                network.train_network()
                network.test_network()
                network.trained_status = True

        for network in networks:
            average_accuracy += network.net.accuracy

        average_accuracy = average_accuracy/len(networks)

        if(i != generations - 1):
            networks = optimizer.evolve(networks)
    networks = sorted(networks, key=lambda x: x.net.accuracy,reverse=True)
    print ("Generations over")
    for network in networks:
        print ("Minimum weight limit : " , network.common_param.weight_minimum_limit)
        print ("Maximum weight limit : " , network.common_param.weight_maximum_limit)
        print ("Learning rate : " , network.common_param.learning_rate)
        print ("Accuracy : " , network.net.accuracy)
        network.generate_gui()
    end_time = time.time()
    total_execution_time = round(end_time - start_time)
    print ("Total Time to run : ", total_execution_time)
    for network in networks:
        total_execution_time -= round(network.end_time - network.start_time)
    print ("Time taken for parameter optimization alone : " , total_execution_time)
    
if __name__ == '__main__':
    main()
