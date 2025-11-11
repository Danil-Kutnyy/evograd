import torch
import pickle
import os
import sys
import torch.optim as optim # Import optimizer
import torch.nn.functional as F # Import functional for softmax
import matplotlib.pyplot as plt # Import matplotlib for plotting
from test6 import NetCustom # Assuming these are needed
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
import random
# --- Configuration ---
POPULATION_FILE = 'population4.pkl'
BATCH_SIZE = 64
NUM_TRAIN_BATCHES = 2000 # Number of batches to train for evaluation
LEARNING_RATE = 0.0003 # Learning rate for the main network optimizer

# --- Helper Functions ---
# --- Data Loader Class ---
class MNISTDataLoader:
    def __init__(self, batch_size=4, data_root='./data', train_split_ratio=0.8, fashion=True):
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.fashion = fashion # Store the flag

        # Load the full dataset based on the fashion flag
        if self.fashion:
            print("Loading Fashion-MNIST dataset...")
            dataset_class = torchvision.datasets.FashionMNIST
        else:
            print("Loading MNIST dataset...")
            dataset_class = torchvision.datasets.MNIST

        full_trainset = dataset_class(root=data_root, train=True, download=True, transform=self.transform)
        
        # Calculate split sizes
        n_samples = len(full_trainset)
        n_train = int(n_samples * train_split_ratio)
        n_eval = n_samples - n_train
        
        # Split the dataset
        train_dataset, eval_dataset = torch.utils.data.random_split(full_trainset, [n_train, n_eval], generator=torch.Generator().manual_seed(42))
        
        # Create DataLoaders for each split
        self.trainloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.evalloader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Initialize iterators
        self._reset_iterator()

    def _reset_iterator(self):
        self.trainiter = iter(self.trainloader)
        self.evaliter = iter(self.evalloader)

    def get_batch(self, mode='train'):
        if mode == 'train':
            iterator = self.trainiter
        elif mode == 'eval':
            iterator = self.evaliter
        try:
            images, labels = next(iterator)
        except StopIteration:
            self._reset_iterator()
            if mode == 'train':
                iterator = self.trainiter
            elif mode == 'eval':
                iterator = self.evaliter
            images, labels = next(iterator)
        return images, labels
    
# Function to insert LEARNING RULE parameters into the managed networks
def insert_parameters(net, individual):
    # Iterate through the network's managed components and the individual's structure
    for (layer_name, nn_dict_original), (_, nn_dict_individual) in zip(net.manager.networks.items(), individual.items()):
        for nn_name, nn_managed in nn_dict_original.items():
            # Get the parameter list for the corresponding part of the individual
            individual_params = nn_dict_individual[nn_name]
            network_params = list(nn_managed.parameters())

            for param_net, param_ind in zip(network_params, individual_params):
                 param_net.data = param_ind.clone().detach()

# Function to get training batches
def get_train_batches(dataloader, num_batches):
    dataloader._reset_iterator()
    train_batches = []
    # Let StopIteration propagate if not enough batches
    for _ in range(num_batches):
        # Use mode='train'
        images, labels = dataloader.get_batch(mode='train')
        train_batches.append((images.clone().detach(), labels.clone().detach()))
    return train_batches
# Function to evaluate a single individual by TRAINING a network
def evaluate_individual(individual, train_batches, learning_rate):
    net = NetCustom() # Create a fresh network instance
    net_control = NetCustom() # Create a fresh network instance
    insert_parameters(net, individual) # Insert the learning rule parameters from the individual

    #copy parameters from net to net_control
    for param_net, param_control in zip(net.parameters(), net_control.parameters()):
        param_control.data = param_net.data.clone().detach()

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    optimizer_control = optim.Adam(net_control.parameters(), lr=learning_rate)
    
    best_accuracy = 0.0
    best_accuracy_control = 0.0
    net.train() # Set network to training mode
    net_control.train() # Set network to training mode

    with torch.no_grad():
        for i, (images, labels) in enumerate(train_batches):
            optimizer.zero_grad()
            net.zero_grads_g2()
            
            outputs, loss = net(images, labels)
            #loss.backward() # Calculate gradients for main optimizer


            net.backprop_adv(test=False)

            adam_delta, updates = net.optimizer_step(learning_rate=learning_rate)
            d_sum = 0
            for p, d, u in zip(net.parameters(), adam_delta, updates):
                d_sum += d.sum()
                p.data.add_(u)
                p.data.add_(d)


            
            accuracy = torch.sum(outputs.argmax(dim=1) == labels).item() / len(labels)
            if i % 100 == 0:
                print(f'#{i}', accuracy)
            if accuracy > best_accuracy:
                best_accuracy = accuracy

    #visualize
    images, labels = train_batches[0]
    outputs, loss = net(images, labels)
    train_batches
    # --- Visualization Code --- 
    # Get the first 4 images, labels, and outputs
    num_images_to_show = 4
    images_to_show = images[:num_images_to_show]
    labels_to_show = labels[:num_images_to_show]
    outputs_to_show = outputs[:num_images_to_show]

# Calculate probabilities using softmax
    probabilities = F.softmax(outputs_to_show, dim=1)
    predicted_labels = torch.argmax(probabilities, dim=1)

    # Create figure and axes
    fig, axes = plt.subplots(num_images_to_show, 2, figsize=(8, 2 * num_images_to_show))
    fig.suptitle("Image vs. Predicted Probabilities")

    for i in range(num_images_to_show):
        # Display image
        ax_img = axes[i, 0]
        img = images_to_show[i].squeeze().cpu().numpy() # Remove channel dim, move to CPU, convert to numpy
        ax_img.imshow(img, cmap='gray')
        ax_img.set_title(f"True: {labels_to_show[i].item()}, Pred: {predicted_labels[i].item()}")
        ax_img.axis('off')

        # Display probability distribution
        ax_prob = axes[i, 1]
        probs = probabilities[i].detach().cpu().numpy() # Detach, move to CPU, convert to numpy
        classes = range(len(probs))
        ax_prob.bar(classes, probs)
        ax_prob.set_xticks(classes)
        ax_prob.set_ylim(0, 1)
        ax_prob.set_ylabel("Probability")
        if i == num_images_to_show - 1:
            ax_prob.set_xlabel("Class")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()
    # --- End Visualization Code ---

    sys.exit()

        
    # Return the peak accuracy achieved during this short training run
    return best_accuracy, best_accuracy_control

# --- Main Execution ---

print(f"Loading population from '{POPULATION_FILE}'...")
with open(POPULATION_FILE, 'rb') as f:
    population = pickle.load(f)
print(f"Loaded {len(population)} individuals.")

print("Initializing DataLoader...")
dataloader = MNISTDataLoader(batch_size=BATCH_SIZE)

print(f"Fetching {NUM_TRAIN_BATCHES} training batches...")
# Use get_train_batches instead of get_eval_batches
train_batches = get_train_batches(dataloader, NUM_TRAIN_BATCHES)
print(f"Using {len(train_batches)} training batches.")

best_score = -1.0
best_individual_index = -1

print("\nEvaluating population by training...")
#individual = population[8]

#score, score_control = evaluate_individual(individual, train_batches, LEARNING_RATE)
#print(f"#{0} acc: {score:.4f}, acc control: {score_control:.4f}, diff: {score-score_control:.4f}")
#sys.exit()

#control
'''
net_control = NetCustom() # Create a fresh network instance

optimizer_control = optim.Adam(net_control.parameters(), lr=learning_rate)
best_accuracy_control = 0.0
net_control.train() # Set network to training mode

control_accuracy = []
for images, labels in train_batches:
    optimizer_control.zero_grad()
    net_control.zero_grads_g2()
    
    outputs, loss = net(images, labels)
    loss.backward() # Calculate gradients for main optimizer

    optimizer_control.step()

    
    accuracy = torch.sum(outputs.argmax(dim=1) == labels).item() / len(labels)
    for
'''
if __name__ == "__main__":
    for i, individual in enumerate(population):
        #print(f"--- Evaluating Individual {i} ---")
        # Pass train_batches and LEARNING_RATE to evaluate_individual
        score, score_control = evaluate_individual(individual, train_batches, LEARNING_RATE)
        print(f"#{i} acc: {score:.4f}, acc control: {score_control:.4f}, diff: {score-score_control:.4f}")

        if score > best_score:
            best_score = score
            best_individual_index = i

    print("\n--- Evaluation Summary ---")
    print(f"Best Performing Individual (Index): {best_individual_index}")
    print(f"Best Peak Training Accuracy: {best_score:.4f}") 