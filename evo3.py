import torch
import torch.optim as optim # Import optimizer
from test6 import NetCustom, MNISTDataLoader, ParameterG2, NetBuilder # Ensure ParameterG2 is imported if needed for optimizer
import time
import sys
import copy # Import copy for deepcopying state dicts
import random # Import random for weighted choice
import csv  # Import for CSV writing
import os
import pickle

torch.manual_seed(42)

# --- Evolutionary Algorithm Parameters ---
POPULATION_SIZE = 30
NUM_EVAL_BATCHES = 2048 # Number of batches to train for fitness evaluation
NUM_TEST_EVAL_BATCHES = 64 # Use a smaller number for the initial function test
NUM_GENERATIONS = 1000000 # Example number of generations
ELITISM_COUNT = 3 # Keep the top 3 individuals directly
MUTATION_MASK = True # If True, mutation_rate is used to fine-tune the amount of mutated values. Otherwise, all values are mutated
MUTATION_RATE = 0.02 # Probability of mutating a single parameter. At least one value will be mutated
MUTATION_STRENGTH = [1.0, 0.001] # Std deviation of Gaussian noise added
batch_size = 64
learning_rate = 0.0003

# Store the population as a list of dictionaries containing cloned+detached TENSORS
def generate_population_old():
    population = []
    for i in range(POPULATION_SIZE):
        individual_layer_rule_tensors = {}
        tmp_net = NetCustom()
        for layer_name, nn_dict in tmp_net.manager.networks.items():
            individual_layer_rule_tensors[layer_name] = {}
            for nn_name, nn in nn_dict.items():
                individual_layer_rule_tensors[layer_name][nn_name] = []
                for param in nn.parameters():
                    individual_layer_rule_tensors[layer_name][nn_name].append(param.data.clone().detach())
                    
        population.append(individual_layer_rule_tensors)
        del tmp_net
    return population

def test_batches(batches):
    total_sum = 0
    for images, labels in batches:
        total_sum += images.sum()
        total_sum += labels.sum()
    #last 4 digits of total_sum
    return total_sum# % 10000

test_net = NetCustom()
def test_global_net(test_batch, eval_batch, net_i=test_net, ):
    sum = 0
    with torch.no_grad():
        for images, labels in test_batch:
            outputs, loss = net_i(images, labels)
            sum+=outputs.sum()
            sum+=loss.sum()
        for images, labels in eval_batch:
            outputs, loss = net_i(images, labels)
            sum+=outputs.sum()
            sum+=loss.sum()
    return sum

def state_dict_to_net(net, state_dict):
    #loop over statedict and make copy with clone ditch
    new_state_dict = {}
    for key, value in state_dict.items():
        new_state_dict[key] = value.clone().detach()
    net.load_state_dict(new_state_dict)

#function to insert parameters to the network
def insert_parameters(net, individual):
    for (_, nn_dict_original), (_, nn_dict_individual) in zip(net.manager.networks.items(), individual.items()):
        for nn_name, nn in nn_dict_original.items():
            for param, param_ind in zip(nn.parameters(), nn_dict_individual[nn_name]):
                param.data = param_ind


#evaluate the fitness of the population
def evaluate_fitness(population, train_batches, eval_batches, eval=False):
    
    #create random variables for neurla network
    net_sizes = []
    last = random.randint(32, 1024)
    for l in [random.randint(128, 1024), random.randint(32, 512), random.randint(16, 256)]:
        if l*1.1 <= last:
            net_sizes.append(l)
        else:
            net_sizes.append(last)
        last = min(l, last)
    random_val = random.randint(1, 7)
    if random_val >= 4:
        net_sizes = net_sizes[-1:]
    elif random_val >= 2:
        net_sizes = net_sizes[-2:]

    net_sizes.append(10)

    #randamize learning rate
    learning_rate = 1/random.randint(10, 1000)


    fitness = []
    #build random neural netwrok
    net_global = NetBuilder(sizes=net_sizes)
    state_dict = net_global.state_dict()
        
    #control
    net = NetBuilder(sizes=net_sizes)
    state_dict_to_net(net, state_dict)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    #contorl (startsat Adam) neural network training
    test_accuracy = []
    for images, labels in train_batches:
        optimizer.zero_grad()
        net.zero_grads_g2()
        outputs, loss = net(images, labels)
        loss.backward()
        optimizer.step()
        accuracy = torch.sum(outputs.argmax(dim=1) == labels).item() / len(labels)
        test_accuracy.append(accuracy)
        control = max(test_accuracy)
        #print('control:',loss.item())
        

    
    if eval:
        with torch.no_grad():
            net.eval()
            eval_accuracy = []
            for images, labels in eval_batches:
                outputs, loss = net(images, labels)
                eval_accuracy.append(torch.sum(outputs.argmax(dim=1) == labels).item() / len(labels))
            
            eval_accuracy = sum(eval_accuracy) / len(eval_accuracy)
            control = eval_accuracy
    del net
    del optimizer
    #print('control:',control)
    
    #train neurl netowrks with evograd mechanism
    with torch.no_grad():
        for individual in population:
            fintes_score = None
            grad_sum = 0
            net = NetBuilder(sizes=net_sizes)
            state_dict_to_net(net, state_dict)
            insert_parameters(net, individual)
            optimizer = optim.Adam(net.parameters(), lr=learning_rate)
            
            # Train using pre-fetched batches
            test_accuracy = []
            n = 0
            for images, labels in train_batches:
                n+=1
                optimizer.zero_grad()
                net.zero_grads_g2()
                outputs, loss = net(images, labels)
                #print('loss:',loss.item())
                net.backprop_adv(test=False)
                '''
                updates = net.optimizer_step(learning_rate=learning_rate)
                optimizer.step()
                for update, param in zip(updates, net.parameters()):
                    param.data.add_(update)
                '''
                adam_delta, updates = net.optimizer_step(learning_rate=learning_rate)
                for p, d, u in zip(net.parameters(), adam_delta, updates):
                    p.data.add_(u)
                    p.data.add_(d)

                accuracy = torch.sum(outputs.argmax(dim=1) == labels).item() / len(labels)
                test_accuracy.append(accuracy)
                fintes_score = max(test_accuracy)

            if eval:
                # Evaluate using pre-fetched batches
                losses = []
                accuracy = []
                for images, labels in eval_batches:
                    outputs, loss = net(images, labels)
                    losses.append(loss.item())
                    accuracy.append(torch.sum(outputs.argmax(dim=1) == labels).item() / len(labels))
                loss = sum(losses) / len(losses)
                fintes_score = sum(accuracy) / len(accuracy)
            #print(grad_sum)
            fitness.append(fintes_score)
            #print('fintes_score:',fintes_score)
    
    
    return fitness, control, net_sizes

def select_parents(population, fitness):
    population_size = len(population) # Use actual population size
    if population_size != POPULATION_SIZE:
        raise ValueError("Population size does not match the expected population size")
    
    all_sorted_indices = sorted(range(population_size), key=lambda i: fitness[i], reverse=True)
    elite_indices = all_sorted_indices[:ELITISM_COUNT]
    elites = [population[i] for i in elite_indices] 
    
    num_offspring_needed = population_size - ELITISM_COUNT 
    num_parents_to_select = num_offspring_needed * 2 
    if num_parents_to_select <= 0:
        raise ValueError("No parents need to be selected")
    weights = [population_size - rank for rank in range(population_size)] 

    selected_parent_original_indices = random.choices(population=all_sorted_indices, weights=weights, k=num_parents_to_select) 
    selected_parents = [population[i] for i in selected_parent_original_indices]
    return elites, selected_parents

def crossover(parents):
    offsprings = []
    counter = 0
    while counter < len(parents):
        p1 = parents[counter]
        p2 = parents[counter+1]
        counter += 2
        offspring = {}
        for layer_name, nn_dict in p1.items():
            offspring[layer_name] = {}
            for nn_name, param_list1 in nn_dict.items():
                offspring[layer_name][nn_name] = []
                param_list2 = p2[layer_name][nn_name]
                for t1, t2 in zip(param_list1, param_list2):
                    mask = torch.randint(0, 2, (t1.shape))
                    inv_mask = 1 - mask
                    t_offspring = t1*mask + t2*inv_mask
                    offspring[layer_name][nn_name].append(t_offspring.clone().detach())
        offsprings.append(offspring)
    return offsprings

#layer weigths caucaltion. Used to mutate parameters uniformly
def calculate_parameter_weights(individual_structure):
    """
    Calculates hierarchical weights for layers, networks, and tensors based on parameter counts.
    """
    weights = {'layers': {'names': [], 'probs': []}, 'networks': {}, 'tensors': {}}
    total_params = 0
    layer_params = {}

    for layer_name, nn_dict in individual_structure.items():
        weights['networks'][layer_name] = {'names': [], 'probs': []}
        weights['tensors'][layer_name] = {}
        current_layer_params = 0
        network_params = {}

        for nn_name, param_list in nn_dict.items():
            weights['tensors'][layer_name][nn_name] = {'indices': [], 'probs': []}
            current_network_params = 0
            tensor_params = {}

            for i, param in enumerate(param_list):
                numel = param.numel()
                tensor_params[i] = numel
                current_network_params += numel

            # Calculate tensor weights within the network
            if current_network_params > 0:
                for i, numel in tensor_params.items():
                    weights['tensors'][layer_name][nn_name]['indices'].append(i)
                    weights['tensors'][layer_name][nn_name]['probs'].append(numel / current_network_params)
            else: # Handle case with no parameters in a network
                 weights['tensors'][layer_name][nn_name]['indices'] = []
                 weights['tensors'][layer_name][nn_name]['probs'] = []


            network_params[nn_name] = current_network_params
            current_layer_params += current_network_params

        # Calculate network weights within the layer
        if current_layer_params > 0:
            for nn_name, numel in network_params.items():
                weights['networks'][layer_name]['names'].append(nn_name)
                weights['networks'][layer_name]['probs'].append(numel / current_layer_params)
        else: # Handle case with no parameters in a layer
            weights['networks'][layer_name]['names'] = []
            weights['networks'][layer_name]['probs'] = []


        layer_params[layer_name] = current_layer_params
        total_params += current_layer_params

    # Calculate layer weights
    if total_params > 0:
        for layer_name, numel in layer_params.items():
            weights['layers']['names'].append(layer_name)
            weights['layers']['probs'].append(numel / total_params)
    else: # Handle case with no parameters at all
        weights['layers']['names'] = []
        weights['layers']['probs'] = []


    return weights


def mutate_old(population):
    for layer_name, nn_dict in individual.items():
        new_individual[layer_name] = {}
        for nn_name, param_list in nn_dict.items():
            new_individual[layer_name][nn_name] = []
            for param in param_list:
                mask = torch.rand(param.shape) < MUTATION_RATE
                noise = torch.rand(param.shape) * mut_strength_tmp
                new_param = param + mask * noise
                new_individual[layer_name][nn_name].append(new_param)
    new_population.append(new_individual)

    return new_population

def mutate(population, weights, use_mask=MUTATION_MASK):
    """
    Mutates exactly one tensor per individual using weighted random selection.
    """
    new_population = []
    shape = None
    for i, individual in enumerate(population):
        # Deep copy to avoid modifying the original offspring
        new_individual = individual

        # Select mutation strength
        mut_strength_tmp = random.choice(MUTATION_STRENGTH)

        # 1. Select Layer
        selected_layer_name = random.choices(weights['layers']['names'], weights=weights['layers']['probs'], k=1)[0]

        # 2. Select Network within Layer
        layer_networks = weights['networks'][selected_layer_name]
        selected_nn_name = random.choices(layer_networks['names'], weights=layer_networks['probs'], k=1)[0]

        # 3. Select Tensor within Network
        network_tensors = weights['tensors'][selected_layer_name][selected_nn_name]
        selected_param_index = random.choices(network_tensors['indices'], weights=network_tensors['probs'], k=1)[0]

        # Apply mutation to the selected tensor
        target_tensor = individual[selected_layer_name][selected_nn_name][selected_param_index]
        if use_mask:
            mask = torch.rand(target_tensor.shape) < MUTATION_RATE
            #if mask is all 0, then add 1 to the mask to the random position
            if mask.sum() == 0:
                mask = mask.flatten()
                rand_pos = random.randint(0, mask.shape[0]-1)
                mask[rand_pos] = 1
                mask = mask.reshape(target_tensor.shape)
        else:
            mask = torch.ones(target_tensor.shape)
        

        noise = 2 * ( torch.rand(target_tensor.shape)-torch.rand(target_tensor.shape) )
        param = individual[selected_layer_name][selected_nn_name][selected_param_index]
        
        new_param = param + ( mask * noise * mut_strength_tmp )
  
        
        #print(mask)
        #print('before:',new_individual[selected_layer_name][selected_nn_name][selected_param_index])
        new_individual[selected_layer_name][selected_nn_name][selected_param_index] = new_param
        #print('after:',new_individual[selected_layer_name][selected_nn_name][selected_param_index])
        
        #val = torch.allclose(individual[selected_layer_name][selected_nn_name][selected_param_index], new_individual[selected_layer_name][selected_nn_name][selected_param_index], atol=1e-6)
        #val = torch.allclose(param, new_param, atol=1e-6)   
        #print('val:',val)
        new_population.append(new_individual)
        #sys.exit()
        #print('mutate done')


    return new_population

def evolve(population, fitness, weights):
    elites, selected_parents = select_parents(population, fitness)
    offsprings_old = crossover(selected_parents)
    offsprings = mutate(offsprings_old, weights)
    new_population = elites + offsprings
    return new_population

#create compeltely new generation
def generate_population(train_batches, eval_batches):
    population = []
    for i in range(10):
        individual_layer_rule_tensors = {}
        tmp_net = NetCustom()
        for layer_name, nn_dict in tmp_net.manager.networks.items():
            individual_layer_rule_tensors[layer_name] = {}
            for nn_name, nn in nn_dict.items():
                individual_layer_rule_tensors[layer_name][nn_name] = []
                for param in nn.parameters():
                    individual_layer_rule_tensors[layer_name][nn_name].append(param.data.clone().detach())
                    
        population.append(individual_layer_rule_tensors)
        del tmp_net
    
    fitness, _, _ = evaluate_fitness(population, train_batches, eval_batches)
    best_index = fitness.index(max(fitness))
    best_individual = population[best_index]
    tmp_net = NetCustom()
    insert_parameters(tmp_net, best_individual)
    population = []
    for i in range(POPULATION_SIZE):
        individual_layer_rule_tensors = {}
        
        for layer_name, nn_dict in tmp_net.manager.networks.items():
            individual_layer_rule_tensors[layer_name] = {}
            for nn_name, nn in nn_dict.items():
                individual_layer_rule_tensors[layer_name][nn_name] = []
                for param in nn.parameters():
                    individual_layer_rule_tensors[layer_name][nn_name].append(param.data.clone().detach())
                    
        population.append(individual_layer_rule_tensors)
    del tmp_net

    return population

def get_batches():
    dataloader = MNISTDataLoader(batch_size=batch_size)
    train_batches = []
    eval_batches = []
    for _ in range(NUM_EVAL_BATCHES):
        images, labels = dataloader.get_batch(mode='train')
        train_batches.append((images, labels))
    for _ in range(NUM_TEST_EVAL_BATCHES):
        images, labels = dataloader.get_batch(mode='eval')
        eval_batches.append((images, labels))
    return train_batches, eval_batches

#evolution loop
elites_value = False
#train_batches, eval_batches = get_batches()
#population = generate_population(train_batches, eval_batches)
#population = generate_population_old()
with open('population3.pkl', 'rb') as f:
    population = pickle.load(f)
weights = calculate_parameter_weights(population[0])

if os.path.exists('fitness4.csv'):
    os.remove('fitness4.csv')
for generation in range(NUM_GENERATIONS):
    train_batches, eval_batches = get_batches()
    fitness, control, net_size = evaluate_fitness(population, train_batches, eval_batches, eval=True)
    population = evolve(population, fitness, weights)
    best_fitness = max(fitness)
    mean_fitness = sum(fitness) / len(fitness)
    print(f"# {generation} best: {best_fitness:.4f} mean: {mean_fitness:.4f}, control: {control:.4f}, network: {net_size}")
    #save best and mean fitness to csv
    with open('fitness4.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([generation, best_fitness, mean_fitness, control, net_size])
    if generation % 30 == 0:
        #save population to file
        with open('population4.pkl', 'wb') as f:
            pickle.dump(population, f)
