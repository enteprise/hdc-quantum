import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt

# =============================================
# Symmetric Group S_n
# =============================================

class SymmetricGroup:
    def __init__(self, n):
        self.n = n
        self.elements = list(permutations(range(n)))

    def bind(self, g, h):
        return tuple(g[h[i]] for i in range(self.n))

    def similarity(self, g, h):
        return sum(1 for i in range(self.n) if g[i] == h[i]) / self.n

    def permute(self, g, v):
        return tuple(v[g[i]] for i in range(self.n))

# =============================================
# Quaternion Group Q_8
# =============================================

class QuaternionGroup:
    def __init__(self):
        self.elements = {
            '1': [1, 0, 0, 0],
            '-1': [-1, 0, 0, 0],
            'i': [0, 1, 0, 0],
            '-i': [0, -1, 0, 0],
            'j': [0, 0, 1, 0],
            '-j': [0, 0, -1, 0],
            'k': [0, 0, 0, 1],
            '-k': [0, 0, 0, -1],
        }

    def bind(self, g, h):
        a1, b1, c1, d1 = g
        a2, b2, c2, d2 = h
        result = [
            a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
            a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
            a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
            a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2
        ]
        return result

    def similarity(self, g, h):
        # Ensure similarity is clamped to [-1, 1]
        dot_product = sum(gi * hi for gi, hi in zip(g, h))
        return max(-1.0, min(1.0, dot_product))

    def exp_quaternion(self, q):
        norm = np.linalg.norm(q[1:])
        if norm == 0:
            return [1, 0, 0, 0]
        else:
            w = np.cos(norm)
            x, y, z = np.array(q[1:]) * np.sin(norm) / norm
            return [w, x, y, z]

# =============================================
# Quantum Spin System Simulation
# =============================================

class QuantumSpinSystem:
    def __init__(self, group, num_spins, hamiltonian):
        self.group = group
        self.num_spins = num_spins
        self.hamiltonian = hamiltonian

        if isinstance(group, SymmetricGroup):
            self.spin_states = [group.elements[0] for _ in range(num_spins)]
        elif isinstance(group, QuaternionGroup):
            self.spin_states = [group.elements['1'] for _ in range(num_spins)]
        else:
            raise ValueError("Unsupported group type.")

        self.observables_history = []

    def time_evolution(self, time_step, num_steps):
        for _ in range(num_steps):
            for i in range(self.num_spins):
                if isinstance(self.group, QuaternionGroup):
                    # Compute the time evolution operator U = exp(-i * H * t)
                    H = self.hamiltonian[i]
                    # Scale Hamiltonian by -1 and time_step
                    H_scaled = [0, -H[1] * time_step, -H[2] * time_step, -H[3] * time_step]
                    U = self.group.exp_quaternion(H_scaled)
                    self.spin_states[i] = self.group.bind(U, self.spin_states[i])
                elif isinstance(self.group, SymmetricGroup):
                    self.spin_states[i] = self.group.permute(self.hamiltonian[i], self.spin_states[i])
            self.observables_history.append(self.get_observables())

    def compute_ground_state(self, num_iterations=100):
        if isinstance(self.group, SymmetricGroup):
            for _ in range(num_iterations):
                for i in range(self.num_spins):
                    min_energy = float('inf')
                    best_perm = self.spin_states[i]
                    for perm in self.group.elements:
                        energy = -self.group.similarity(perm, self.hamiltonian[i])
                        if energy < min_energy:
                            min_energy = energy
                            best_perm = perm
                    self.spin_states[i] = best_perm
                self.observables_history.append(self.get_observables())
        elif isinstance(self.group, QuaternionGroup):
            for _ in range(num_iterations):
                for i in range(self.num_spins):
                    # Simple gradient descent for quaternions
                    grad = [self.group.similarity(self.spin_states[i], self.hamiltonian[i]) for _ in range(4)]
                    learning_rate = 0.01
                    new_state = [self.spin_states[i][j] - learning_rate * grad[j] for j in range(4)]
                    # Normalize to maintain unit quaternion
                    norm = np.linalg.norm(new_state)
                    if norm > 0:
                        self.spin_states[i] = [x / norm for x in new_state]
                self.observables_history.append(self.get_observables())

    def get_observables(self):
        if isinstance(self.group, SymmetricGroup):
            magnetization = sum(self.group.similarity(self.spin_states[i], self.hamiltonian[i]) for i in range(self.num_spins))
        elif isinstance(self.group, QuaternionGroup):
            magnetization = sum(self.group.similarity(self.spin_states[i], self.hamiltonian[i]) for i in range(self.num_spins))
        else:
            magnetization = 0.0
        return {"magnetization": magnetization / self.num_spins}

    def plot_observables(self):
        time_steps = range(len(self.observables_history))
        magnetization = [obs["magnetization"] for obs in self.observables_history]

        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, magnetization, label="Magnetization")
        plt.xlabel("Time Step")
        plt.ylabel("Magnetization")
        plt.title("Observables Over Time")
        plt.legend()
        plt.grid()
        plt.show()

# =============================================
# Testing the Implementation
# =============================================

if __name__ == "__main__":
    # Test with Symmetric Group S_3
    print("Testing Symmetric Group S_3:")
    symmetric_group = SymmetricGroup(3)
    g = symmetric_group.elements[0]  # Identity permutation
    h = symmetric_group.elements[1]  # First non-trivial permutation
    print(f"Binding g and h: {symmetric_group.bind(g, h)}")
    print(f"Similarity between g and h: {symmetric_group.similarity(g, h)}")

    # Test with Quaternion Group Q_8
    print("\nTesting Quaternion Group Q_8:")
    quaternion_group = QuaternionGroup()
    g = quaternion_group.elements['i']
    h = quaternion_group.elements['j']
    print(f"Binding g and h: {quaternion_group.bind(g, h)}")
    print(f"Similarity between g and h: {quaternion_group.similarity(g, h)}")

    # Test Quantum Spin System with Symmetric Group
    print("\nTesting Quantum Spin System with Symmetric Group:")
    hamiltonian = [symmetric_group.elements[i] for i in range(3)]
    spin_system = QuantumSpinSystem(symmetric_group, num_spins=3, hamiltonian=hamiltonian)
    spin_system.time_evolution(time_step=1, num_steps=50)
    spin_system.compute_ground_state(num_iterations=50)
    print(f"Final Observables: {spin_system.get_observables()}")
    spin_system.plot_observables()

    # Test Quantum Spin System with Quaternion Group
    print("\nTesting Quantum Spin System with Quaternion Group:")
    # Define Hamiltonian as a combination of i, j, k
    H = [0, 1, 1, 1]  # i + j + k
    hamiltonian_quat = [H for _ in range(3)]  # Simple Hamiltonian
    spin_system_quat = QuantumSpinSystem(quaternion_group, num_spins=3, hamiltonian=hamiltonian_quat)
    spin_system_quat.time_evolution(time_step=1, num_steps=50)
    spin_system_quat.compute_ground_state(num_iterations=50)
    print(f"Final Observables: {spin_system_quat.get_observables()}")
    spin_system_quat.plot_observables()