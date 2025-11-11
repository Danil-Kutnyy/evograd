import csv
import matplotlib.pyplot as plt
import numpy as np # Import numpy for moving average

steps = []
accuracies = []
end = 3000
start = 150
window_size = 64 # Window size for moving average
#step 1 windiw size = 8. WAit until 3 windo low - current ~ 22
#step 2 windiw size = 32. WAit until 8 windo low - current ~ 141
#step 3 windiw size = 128. WAit until 128 windo low - current ~ 1128
#step 4 windiw size = 512. WAit until 512 windo low - current ~ 10000
#step 5 windiw size = 1024. WAit until 1024 windo low - current ~ 25000
# Read data from CSV
with open('control_data.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader) # Skip header
    for row in csv_reader:
        steps.append(int(row[0]))
        accuracies.append(float(row[1]))

# Slice the data *before* calculating moving average if needed
steps = steps
accuracies = accuracies

# Calculate moving average
# Pad the start with NaNs so the moving average array aligns with the steps
if len(accuracies) >= window_size:
    moving_avg = np.convolve(accuracies[start:end], np.ones(window_size)/window_size, mode='valid')
    # Pad with NaN to align with original data for plotting
    moving_avg_padded = np.concatenate((np.full(window_size - 1, np.nan), moving_avg))
else:
    # Handle cases where data is shorter than window size
    moving_avg_padded = np.full(len(accuracies[start:end]), np.nan)

# Plot the data
plt.figure(figsize=(10, 5))
#plt.plot(steps[start:end], accuracies[start:end], label='Accuracy', alpha=0.5) # Original accuracy
plt.plot(steps[start:end], moving_avg_padded, label=f'Moving Average (window={window_size})', color='red') # Moving average
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.title('Training Accuracy with Moving Average')
plt.grid(True)
plt.legend() # Add legend
plt.savefig('accuracy_plot.png') # Save the plot
plt.show() # Display the plot

# Note: Make sure you have numpy installed: pip install numpy 