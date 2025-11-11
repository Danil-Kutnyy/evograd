import csv
import matplotlib.pyplot as plt
import numpy as np

# Moving average window size
MV = 32 # Set to 1 to disable moving average
start = 16
minimum = 0.8

def moving_average(data, window_size):
    """Calculates the moving average of a list."""
    if window_size <= 1:
        return np.array(data) # Ensure output is numpy array for subtraction
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Read data from CSV file
# Note: This script runs in an infinite loop to re-plot when the CSV updates.
# You might need to interrupt it manually (e.g., Ctrl+C).
while True:
    generations = []
    best_fitness = []
    mean_fitness = []
    control_fitness = []

    try:
        with open('fitness4.csv', 'r') as f:
            reader = csv.reader(f)
            counter = 0
            for row in reader:
                if counter >= start:
                    # Expecting [generation, best_fitness, mean_fitness, control, ...]
                    if len(row) >= 4:
                        try:
                            generations.append(int(row[0]))
                            best_fit = float(row[1])
                            mean_fit = float(row[2])
                            control_fit = float(row[3])
                            if best_fit < minimum:
                                best_fit = minimum
                            if mean_fit < minimum:
                                mean_fit = minimum
                            if control_fit < minimum:
                                control_fit = minimum
                            best_fitness.append(best_fit)
                            mean_fitness.append(mean_fit)
                            control_fitness.append(control_fit) # Read control value
                        except ValueError:
                            print(f"Skipping invalid numeric data in row: {row}")
                    else:
                        print(f"Skipping row with insufficient columns: {row}")
                counter+=1
    except FileNotFoundError:
        print("fitness.csv not found. Waiting...")
        plt.pause(5) # Wait 5 seconds before trying again
        continue
    except Exception as e:
        print(f"An error occurred: {e}")
        plt.pause(5)
        continue

    if not generations: # Skip plotting if no data was read
        print("No data read from fitness.csv. Waiting...")
        plt.pause(5)
        continue

    # Convert lists to numpy arrays for easier calculations
    generations = np.array(generations)
    best_fitness = np.array(best_fitness)
    mean_fitness = np.array(mean_fitness)
    control_fitness = np.array(control_fitness)

    # Apply moving average if MV > 1
    generations_to_plot = generations
    best_fitness_to_plot = best_fitness
    mean_fitness_to_plot = mean_fitness
    control_fitness_to_plot = control_fitness

    if MV > 1 and len(generations) >= MV:
        # Adjust generations to match the length of the moving average data
        generations_to_plot = generations[MV-1:]
        best_fitness_to_plot = moving_average(best_fitness, MV)
        mean_fitness_to_plot = moving_average(mean_fitness, MV)
        control_fitness_to_plot = moving_average(control_fitness, MV)
    elif MV > 1:
        print(f"Not enough data points ({len(generations)}) for moving average window ({MV}). Plotting raw data.")
        # Reset to raw data if not enough points for MV
        generations_to_plot = generations
        best_fitness_to_plot = best_fitness
        mean_fitness_to_plot = mean_fitness
        control_fitness_to_plot = control_fitness


    # Ensure all arrays have the same length after potential moving average calculation
    min_len = len(generations_to_plot)
    best_fitness_to_plot = best_fitness_to_plot[:min_len]
    mean_fitness_to_plot = mean_fitness_to_plot[:min_len]
    control_fitness_to_plot = control_fitness_to_plot[:min_len]


    # Calculate deltas
    delta_best_control = best_fitness_to_plot - control_fitness_to_plot
    delta_mean_control = mean_fitness_to_plot - control_fitness_to_plot

    # Create the plot with two subplots
    plt.figure(figsize=(10, 8)) # Adjust figure size

    # Subplot 1: Absolute Fitness Values
    plt.subplot(2, 1, 1) # 2 rows, 1 column, first plot
    plt.plot(generations_to_plot, best_fitness_to_plot, label='Best Fitness',)# marker='.', linestyle='-')
    plt.plot(generations_to_plot, mean_fitness_to_plot, label='Mean Fitness',)# marker='.', linestyle='--')
    plt.plot(generations_to_plot, control_fitness_to_plot, label='Control Fitness',)# marker='.', linestyle=':') # Plot control
    plt.ylabel('Fitness (Accuracy)')
    plt.title('Evolution of Fitness over Generations')
    plt.legend()
    plt.grid(True)

    # Subplot 2: Delta Fitness Values
    plt.subplot(2, 1, 2) # 2 rows, 1 column, second plot
    plt.plot(generations_to_plot, delta_best_control, label='Best - Control',)# marker='.', linestyle='-')
    plt.plot(generations_to_plot, delta_mean_control, label='Mean - Control',)# marker='.', linestyle='--')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Delta')
    plt.title('Fitness Delta Compared to Control')
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.show()
