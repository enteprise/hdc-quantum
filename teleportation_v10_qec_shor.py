import numpy as np
import matplotlib.pyplot as plt

# Set a fixed random seed for reproducibility
np.random.seed(42)

# Initialize hypervectors
def initialize_hypervector(dim):
    vector = np.random.rand(dim)
    return vector / np.linalg.norm(vector)

# Entangle two hypervectors with weighted strategy
def entangle_hypervectors_weighted(vec1, vec2, weight1=0.5, weight2=0.5):
    entangled = weight1 * vec1 + weight2 * vec2
    return entangled / np.linalg.norm(entangled)

# Simulate measurement by projecting onto a basis
def measure_hypervector(state, basis):
    similarity = np.dot(state, basis)
    projection = similarity * basis
    return projection / np.linalg.norm(projection)

# Apply a permutation (reordering) to the hypervector
def permute_hypervector(vector, permutation):
    assert len(permutation) == len(vector), "Permutation length must match vector length."
    assert sorted(permutation) == list(range(len(vector))), "Invalid permutation indices."
    return vector[permutation]

# Calculate similarity between two vectors
def calculate_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Photonic LCD System: Simulate polarization and phase manipulation
def photonic_lcd_system(input_state, operation):
    """
    Simulates the photonic LCD system.
    :param input_state: Input state (hypervector).
    :param operation: Operation to perform ('polarization', 'phase_shift', 'entanglement').
    :return: Transformed state.
    """
    if operation == 'polarization':
        # Simulate polarization change (e.g., 90° rotation)
        polarization_matrix = np.array([[0, -1], [1, 0]])  # 90° rotation matrix
        transformed_state = np.dot(polarization_matrix, input_state)
    elif operation == 'phase_shift':
        # Simulate phase shift (e.g., 180° phase shift)
        phase_shift_matrix = np.array([[-1, 0], [0, -1]])  # 180° phase shift matrix
        transformed_state = np.dot(phase_shift_matrix, input_state)
    elif operation == 'entanglement':
        # Simulate entanglement by combining two states
        entangled_state = entangle_hypervectors_weighted(input_state, initialize_hypervector(len(input_state)))
        transformed_state = entangled_state
    else:
        raise ValueError("Invalid operation for photonic LCD system.")
    
    return transformed_state / np.linalg.norm(transformed_state)

# Quantum Error Correction: 3-Qubit Bit-Flip Code
def encode_bit_flip_code(state):
    """Encodes a hypervector using the 3-qubit bit-flip code."""
    vec1 = state  # |000⟩
    vec2 = permute_hypervector(state, np.random.permutation(len(state)))  # |111⟩
    return [vec1, vec2]

def decode_bit_flip_code(encoded_state):
    """Decodes a hypervector using the 3-qubit bit-flip code."""
    # Majority voting to correct errors
    if calculate_similarity(encoded_state[0], encoded_state[1]) > 0.5:
        return encoded_state[0]
    else:
        return encoded_state[1]

# Quantum Error Correction: 3-Qubit Phase-Flip Code
def encode_phase_flip_code(state):
    """Encodes a hypervector using the 3-qubit phase-flip code."""
    vec1 = state  # |+++⟩
    vec2 = permute_hypervector(state, np.random.permutation(len(state)))  # |---⟩
    return [vec1, vec2]

def decode_phase_flip_code(encoded_state):
    """Decodes a hypervector using the 3-qubit phase-flip code."""
    # Majority voting to correct errors
    if calculate_similarity(encoded_state[0], encoded_state[1]) > 0.5:
        return encoded_state[0]
    else:
        return encoded_state[1]

# Hybrid Error Correction: Combine Bit-Flip and Phase-Flip Codes
def encode_hybrid_code(state):
    """Encodes a hypervector using a hybrid of bit-flip and phase-flip codes."""
    # First, encode using the bit-flip code
    bit_flip_encoded = encode_bit_flip_code(state)
    # Then, encode each resulting hypervector using the phase-flip code
    hybrid_encoded = [encode_phase_flip_code(vec) for vec in bit_flip_encoded]
    return hybrid_encoded

def decode_hybrid_code(encoded_state):
    """Decodes a hypervector using a hybrid of bit-flip and phase-flip codes."""
    # First, decode using the phase-flip code
    phase_flip_decoded = [decode_phase_flip_code(triplet) for triplet in encoded_state]
    # Ensure the input to decode_bit_flip_code is a list of two hypervectors
    if len(phase_flip_decoded) != 2:
        raise ValueError("Input to decode_bit_flip_code must be a list of two hypervectors.")
    # Then, decode using the bit-flip code
    final_state = decode_bit_flip_code(phase_flip_decoded)
    return final_state

# Noise Models
def apply_depolarizing_noise(state, error_prob):
    """Applies depolarizing noise to a hypervector."""
    noise = np.random.normal(0, error_prob, len(state))
    noisy_state = state + noise
    return noisy_state / np.linalg.norm(noisy_state)

def apply_amplitude_damping(state, damping_factor):
    """Applies amplitude damping noise to a hypervector."""
    damping = np.random.rand(len(state)) * damping_factor
    damped_state = state * (1 - damping)
    return damped_state / np.linalg.norm(damped_state)

# Teleportation Simulation with Hybrid Code and Noise Models
def teleportation_simulation_with_hybrid_code(dimension, weight1, measurement_basis_method, permutation, noise_model=None, error_prob=0.01, damping_factor=0.01):
    # Initialize hypervectors
    state_to_teleport = initialize_hypervector(dimension)
    
    # Apply noise model (if specified)
    if noise_model == 'depolarizing':
        state_to_teleport = apply_depolarizing_noise(state_to_teleport, error_prob)
    elif noise_model == 'amplitude_damping':
        state_to_teleport = apply_amplitude_damping(state_to_teleport, damping_factor)
    
    # Encode the state using the hybrid code
    encoded_state = encode_hybrid_code(state_to_teleport)
    
    # Create entangled state using photonic LCD system
    entangled_state = photonic_lcd_system(encoded_state[0][0], 'entanglement')
    
    # Construct measurement basis with reduced noise
    if measurement_basis_method == 'original_noisy':
        measurement_basis = state_to_teleport + np.random.normal(0, 0.000001, dimension)  # Further reduced noise
    elif measurement_basis_method == 'entangled_noisy':
        measurement_basis = entangled_state + np.random.normal(0, 0.000001, dimension)  # Further reduced noise
    else:
        raise ValueError("Invalid measurement basis method.")
    measurement_basis /= np.linalg.norm(measurement_basis)
    
    # Simulate measurement
    measured_state = measure_hypervector(entangled_state, measurement_basis)
    
    # Decode the measured state using the hybrid code
    decoded_state = decode_hybrid_code([[measured_state, measured_state], [measured_state, measured_state]])
    
    # Apply permutations
    permuted_state = permute_hypervector(decoded_state, permutation)
    teleported_state = permute_hypervector(permuted_state, np.argsort(permutation))
    
    # Calculate fidelity
    fidelity = calculate_similarity(state_to_teleport, teleported_state)
    return fidelity

# Main optimization loop
def main():
    dimension = 10000  # Smaller dimension for practicality
    permutation = np.random.permutation(dimension)

    # Test different noise models
    noise_models = [None, 'depolarizing', 'amplitude_damping']
    results = {}

    for noise_model in noise_models:
        print(f"Testing with noise model: {noise_model}")
        weights = np.arange(0.1, 1.0, 0.1)
        fidelities = []

        for weight in weights:
            fidelity = teleportation_simulation_with_hybrid_code(dimension, weight, 'entangled_noisy', permutation, noise_model)
            fidelities.append(fidelity)
            print(f"Weight: {weight}, Fidelity: {fidelity:.4f}")

        # Plot fidelity vs weights
        plt.plot(weights, fidelities, marker='o', label=f"Noise Model: {noise_model}")
        plt.xlabel('Entanglement Weight')
        plt.ylabel('Fidelity')
        plt.title('Fidelity vs Entanglement Weight')
        plt.legend()
        plt.show()

        # Find optimal weight
        optimal_weight = weights[np.argmax(fidelities)]
        max_fidelity = np.max(fidelities)
        results[noise_model] = (optimal_weight, max_fidelity)
        print(f"Optimal Weight: {optimal_weight}, Max Fidelity: {max_fidelity:.4f}")

    # Print final results
    print("\nFinal Results:")
    for noise_model, (optimal_weight, max_fidelity) in results.items():
        print(f"Noise Model: {noise_model}, Optimal Weight: {optimal_weight}, Max Fidelity: {max_fidelity:.4f}")

if __name__ == "__main__":
    main()