# Configuration for validation experiments

# Population and data settings
POPULATION_FILE = 'population4.pkl'
DATASETS = ['mnist', 'fashion_mnist', 'cifar10']

# Statistical validation settings
N_SEEDS = 10  # Number of random seeds for statistical significance
N_TOP_INDIVIDUALS = 5  # Test top N individuals from population
N_SCREENING_SEEDS = 3  # Quick screening before full validation

# Training parameters
BATCH_SIZE = 64
NUM_TRAIN_BATCHES = 200  # Reduce for faster testing
NUM_TEST_BATCHES = 50
LEARNING_RATE = 0.0003
NUM_EPOCHS = 3

# Quick test settings (for development)
QUICK_TEST_SEEDS = [42, 123, 456]
QUICK_TEST_BATCHES = 50
QUICK_TEST_EPOCHS = 2

# Statistical significance threshold
ALPHA = 0.05  # p-value threshold

# Output settings
SAVE_DETAILED_LOGS = True
SAVE_RAW_RESULTS = True
GENERATE_PLOTS = False  # Set to True if you add plotting functionality
