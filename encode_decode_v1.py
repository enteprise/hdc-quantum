import numpy as np
import os
from scipy.sparse import csr_matrix

# High-dimensional logic gates
def high_dim_x_gate(state):
    return np.roll(state, 1)

def high_dim_y_gate(state):
    """High-dimensional Y-gate (combination of X and Z gates)."""
    state = high_dim_x_gate(state)  # Apply X-gate
    state = high_dim_z_gate(state, len(state))  # Apply Z-gate
    return state

def high_dim_z_gate(state, d):
    phases = np.array([np.cos(np.pi * j / (4 * d)) for j in range(d)])
    return state * phases

def hadamard_like_gate(state):
    d = len(state)
    H = np.ones((d, d)) / np.sqrt(d)
    return np.dot(H, state)

# Encoding into HDC space with alignment loop
def encode_hdc_vector_with_alignment(data, gates, max_iter=10):
    d = len(data)
    hypervector = data.copy()
    best_similarity = 0
    best_hypervector = hypervector.copy()

    for _ in range(max_iter):
        improved = False
        for gate in gates:
            if gate == 'X':
                candidate = high_dim_x_gate(hypervector)
            elif gate == 'Y':
                candidate = high_dim_y_gate(hypervector)
            elif gate == 'Z':
                candidate = high_dim_z_gate(hypervector, d)
            elif gate == 'H':
                candidate = hadamard_like_gate(hypervector)
            else:
                raise ValueError(f"Unknown gate: {gate}")

            candidate = candidate / np.linalg.norm(candidate)
            similarity = np.dot(candidate, data) / (np.linalg.norm(candidate) * np.linalg.norm(data))

            if similarity > best_similarity:
                best_similarity = similarity
                best_hypervector = candidate
                improved = True

        if not improved:
            break

        hypervector = best_hypervector

    return best_hypervector

# Decoding from HDC space
def decode_hdc_vector(hypervector, original_data):
    # Ensure both vectors are dense NumPy arrays and flattened
    if isinstance(hypervector, csr_matrix):
        hypervector = hypervector.toarray().flatten()
    if isinstance(original_data, csr_matrix):
        original_data = original_data.toarray().flatten()
    
    # Ensure both vectors are 1D and have the same length
    hypervector = np.asarray(hypervector).flatten()
    original_data = np.asarray(original_data).flatten()
    
    if hypervector.shape != original_data.shape:
        raise ValueError(f"Dimension mismatch: {hypervector.shape} vs {original_data.shape}")
    
    similarity = np.dot(hypervector, original_data) / (np.linalg.norm(hypervector) * np.linalg.norm(original_data))
    return abs(similarity)

# Initialize hypervector
def initialize_hypervector(dim):
    vector = np.random.rand(dim)
    return vector / np.linalg.norm(vector)

# Apply a permutation (reordering) to the hypervector
def permute_hypervector(vector, permutation):
    assert len(permutation) == len(vector), "Permutation length must match vector length."
    assert sorted(permutation) == list(range(len(vector))), "Invalid permutation indices."
    return vector[permutation]

# Quantum Error Correction: Repetition Code
def encode_repetition_code(state, n_copies=3):
    encoded = np.zeros_like(state)
    for _ in range(n_copies):
        perm = np.random.permutation(len(state))
        encoded += permute_hypervector(state, perm)
    return encoded / np.sqrt(n_copies)

# Encode file data into a hypervector
def encode_file_data(file_path, dimension, gates):
    # Read the file in binary mode
    with open(file_path, "rb") as file:
        file_data = file.read()
    
    # Convert binary data to a NumPy array of integers
    file_data_array = np.frombuffer(file_data, dtype=np.uint8)
    
    # Normalize the data to the range [0, 1]
    file_data_normalized = file_data_array / 255.0
    
    # Pad or truncate the data to match the desired dimension
    if len(file_data_normalized) < dimension:
        file_data_normalized = np.pad(file_data_normalized, (0, dimension - len(file_data_normalized)), mode='constant')
    else:
        file_data_normalized = file_data_normalized[:dimension]
    
    # Encode the data into a hypervector
    encoded_vector = encode_hdc_vector_with_alignment(file_data_normalized, gates)
    return encoded_vector

# Main workflow
def main():
    # Step 1: Define the file path and hypervector dimension
    file_path = "large_text_file.txt"  # Replace with any file path
    dimension = 64  # Hypervector dimensionality
    gates = ['X', 'Y', 'Z', 'H']

    # Step 2: Encode the file data into a hypervector
    encoded_vector = encode_file_data(file_path, dimension, gates)
    print("\nFile data encoded into hypervector with shape:", encoded_vector.shape)

    # Step 3: Save the encoded hypervector to a compressed .npz file
    memory_file_path = "encoded_hypervector.npz"
    np.savez_compressed(memory_file_path, encoded_vector=encoded_vector)  # Save as dense array
    print(f"\nEncoded hypervector saved to {memory_file_path}")

    # Step 4: Compare file sizes
    original_size = os.path.getsize(file_path)
    memory_size = os.path.getsize(memory_file_path)
    print(f"Size of original file: {original_size} bytes")
    print(f"Size of encoded hypervector file: {memory_size} bytes")
    print(f"Compression ratio: {original_size / memory_size:.2f}x")

    # Step 5: Decode the hypervector and calculate the return percentage
    loaded_data = np.load(memory_file_path, allow_pickle=True)  # Allow pickle to load the array
    loaded_encoded_vector = loaded_data['encoded_vector']
    
    # Ensure the loaded encoded vector has the correct shape
    loaded_encoded_vector = np.asarray(loaded_encoded_vector).flatten()
    
    # Ensure the original encoded vector is flattened
    encoded_vector = encoded_vector.flatten()
    
    decode_percentage = decode_hdc_vector(loaded_encoded_vector, encoded_vector) * 100
    print(f"Decode return percentage: {decode_percentage:.2f}%")

if __name__ == "__main__":
    main()