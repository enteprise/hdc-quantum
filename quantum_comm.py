import numpy as np
import matplotlib.pyplot as plt

# Set a fixed random seed for reproducibility
np.random.seed(42)

# Initialize hypervectors
def initialize_hypervector(dim):
   """
   Initialize a random hypervector and normalize it.
   """
   vector = np.random.rand(dim)
   return vector / np.linalg.norm(vector)

# Entangle two hypervectors with weighted strategy
def entangle_hypervectors_weighted(vec1, vec2, weight1=0.5, weight2=0.5):
   """
   Entangle two hypervectors using a weighted sum and normalize the result.
   """
   entangled = weight1 * vec1 + weight2 * vec2
   return entangled / np.linalg.norm(entangled)

# Simulate measurement by projecting onto a basis
def measure_hypervector(state, basis):
   """
   Measure a hypervector by projecting it onto a basis and normalize the result.
   """
   similarity = np.dot(state, basis)
   projection = similarity * basis
   return projection / np.linalg.norm(projection)

# Apply a permutation (reordering) to the hypervector
def permute_hypervector(vector, permutation):
   """
   Permute a hypervector using the given permutation indices.
   """
   assert len(permutation) == len(vector), "Permutation length must match vector length."
   assert sorted(permutation) == list(range(len(vector))), "Invalid permutation indices."
   return vector[permutation]

# Calculate similarity between two vectors
def calculate_similarity(vec1, vec2):
   """
   Calculate the cosine similarity between two vectors.
   """
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

# Quantum Communication System Simulation
def quantum_communication_system(dimension, noise_model=None, error_prob=0.001, damping_factor=0.001):
   """
   Simulate a quantum communication system with noise and error correction.
   """
   # Step 1: Alice prepares the quantum state
   state_to_send = initialize_hypervector(dimension)

   # Step 2: Alice encodes the state using the hybrid code
   encoded_state = encode_hybrid_code(state_to_send)

   # Step 3: Simulate noise in the quantum channel
   if noise_model == 'depolarizing':
       noisy_state = [apply_depolarizing_noise(vec, error_prob) for triplet in encoded_state for vec in triplet]
   elif noise_model == 'amplitude_damping':
       noisy_state = [apply_amplitude_damping(vec, damping_factor) for triplet in encoded_state for vec in triplet]
   else:
       noisy_state = [vec for triplet in encoded_state for vec in triplet]

   # Step 4: Bob receives and decodes the state
   decoded_state = decode_hybrid_code([noisy_state[i:i+2] for i in range(0, len(noisy_state), 2)])

   # Step 5: Calculate fidelity of the received state
   fidelity = calculate_similarity(state_to_send, decoded_state)
   return fidelity

# Main function to test the quantum communication system
def main():
   dimension = 10000  # Fixed dimension
   noise_models = [None, 'depolarizing', 'amplitude_damping']  # Noise models to test
   noise_strengths = np.linspace(0.001, 0.01, 10)  # Noise strengths to test
   num_trials = 20  # Number of trials for statistical analysis

   # Store results for plotting
   results = {model: [] for model in noise_models}

   # Test different noise models and noise strengths
   for noise_model in noise_models:
       print(f"\nTesting with noise model: {noise_model}")
       for strength in noise_strengths:
           fidelities = []
           for trial in range(num_trials):
               fidelity = quantum_communication_system(dimension, noise_model, error_prob=strength, damping_factor=strength)
               fidelities.append(fidelity)
           mean_fidelity = np.mean(fidelities)
           std_fidelity = np.std(fidelities)
           results[noise_model].append(mean_fidelity)
           print(f"Noise Strength: {strength:.4f}, Mean Fidelity: {mean_fidelity:.4f}, Std Dev: {std_fidelity:.4f}")

   # Visualize results
   for noise_model, fidelities in results.items():
       plt.plot(noise_strengths, fidelities, marker='o', label=f"Noise Model: {noise_model}")
   plt.xlabel('Noise Strength')
   plt.ylabel('Fidelity')
   plt.title('Fidelity vs Noise Strength (Hybrid Error Correction)')
   plt.legend()
   plt.grid(True)
   plt.show()

if __name__ == "__main__":
   main()
