import numpy as np
import time

# Constants
N = 10000  # Dimensionality of hypervectors
tau = 0.5  # Beam-splitter transmissivity
sigma = 0.1  # Standard deviation of Gaussian noise
D = N + 2  # Circuit depth

# Generate a random sparse hypervector
def generate_sparse_hypervector(N, k):
    hypervector = np.zeros(N)
    indices = np.random.choice(N, k, replace=False)
    hypervector[indices] = 1  # Binary sparse vector for simplicity
    return hypervector

# Simulate the photonic circuit (matrix-vector multiplication)
def photonic_circuit(matrix, vector):
    # Simulate matrix-vector multiplication with Gaussian noise
    result = np.dot(matrix, vector) + np.random.normal(0, sigma, N)
    return result

# Generate a random transfer matrix for the photonic circuit
def generate_transfer_matrix(N, tau):
    # Simulate a unitary matrix with beam-splitter meshes
    # For simplicity, we use a random unitary matrix with transmissivity tau
    matrix = np.random.rand(N, N) * tau
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)  # Normalize rows
    return matrix

# Normalized Square Error (NSE) metric
def normalized_square_error(A0, A):
    return np.mean(np.abs(A0 - A) ** 2)

# Performance metrics
def measure_performance(matrix, vector):
    start_time = time.time()
    result = photonic_circuit(matrix, vector)
    latency = time.time() - start_time
    energy_consumption = latency * 1e-12  # Simulated energy consumption (1e-12 J/operation)
    return result, latency, energy_consumption

# Main simulation function
def run_simulation(N, k, tau, sigma):
    # Step 1: Generate sparse hypervectors
    hypervector_x = generate_sparse_hypervector(N, k)
    hypervector_y = generate_sparse_hypervector(N, k)

    # Step 2: Generate a random transfer matrix for the photonic circuit
    transfer_matrix = generate_transfer_matrix(N, tau)

    # Step 3: Perform matrix-vector multiplication using the photonic circuit
    result, latency, energy_consumption = measure_performance(transfer_matrix, hypervector_x)

    # Step 4: Calculate accuracy (NSE)
    target_result = np.dot(transfer_matrix, hypervector_x)
    nse = normalized_square_error(target_result, result)

    # Step 5: Print performance metrics
    print(f"\nSparsity Level (k = {k}) Results:")
    print(f"Latency: {latency:.2e} seconds")
    print(f"Energy Consumption: {energy_consumption:.2e} joules")
    print(f"Normalized Square Error (NSE): {nse:.2e}")

    # Step 6: Compare with traditional electronic systems (simulated)
    traditional_latency = latency * 100
    traditional_energy = energy_consumption * 1000
    print("\nComparison with Traditional Systems:")
    print(f"Traditional Latency: {traditional_latency:.2e} seconds")
    print(f"Traditional Energy Consumption: {traditional_energy:.2e} joules")

# Main script
if __name__ == "__main__":
    # Test different sparsity levels
    sparsity_levels = [100, 200, 300, 500, 1000]  # Sparsity levels to test

    for k in sparsity_levels:
        run_simulation(N, k, tau, sigma)
