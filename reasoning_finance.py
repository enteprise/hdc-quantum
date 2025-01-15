import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# ==============================
# 1. Load Financial Dataset
# ==============================

def load_stock_data(ticker, start_date, end_date):
    """
    Load historical stock price data using Yahoo Finance.
    :param ticker: Stock ticker symbol (e.g., "AAPL").
    :param start_date: Start date in "YYYY-MM-DD" format.
    :param end_date: End date in "YYYY-MM-DD" format.
    :return: DataFrame with stock price data.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def encode_stock_data(stock_data, feature_columns, dimensionality):
    """
    Encode stock price data as high-dimensional vectors.
    :param stock_data: DataFrame with stock price data.
    :param feature_columns: List of columns to use as features (e.g., ["Open", "High", "Low", "Close", "Volume"]).
    :param dimensionality: Dimensionality of the output vectors.
    :return: Array of encoded vectors.
    """
    # Select relevant features
    features = stock_data[feature_columns].values
    
    # Normalize the features
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)
    
    # Pad or truncate to match the desired dimensionality
    if normalized_features.shape[1] < dimensionality:
        # Pad with zeros
        padding = np.zeros((normalized_features.shape[0], dimensionality - normalized_features.shape[1]))
        encoded_vectors = np.hstack([normalized_features, padding])
    else:
        # Truncate
        encoded_vectors = normalized_features[:, :dimensionality]
    
    return encoded_vectors.astype(np.float32)

def compute_rewards(predictions, true_values):
    """
    Compute rewards based on the magnitude of the prediction error.
    :param predictions: Predicted stock price movements.
    :param true_values: Actual stock price movements.
    :return: Array of rewards.
    """
    rewards = 1.0 - np.abs(predictions - true_values)  # Reward is higher for smaller errors
    return rewards.reshape(-1, 1)

# ==============================
# 2. Policy SLM (Small Language Model)
# ==============================

class PolicySLM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PolicySLM, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Output a scalar
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

# ==============================
# 3. Process Reward Model (PPM) - Improved
# ==============================

class ProcessRewardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ProcessRewardModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Additional layer
        self.fc3 = nn.Linear(hidden_dim, 1)  # Outputs a single reward score
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Add dropout

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.activation(self.fc2(x))
        x = self.dropout(x)  # Apply dropout
        x = torch.sigmoid(self.fc3(x))
        return x

# ==============================
# 4. Monte Carlo Tree Search (MCTS) - Updated
# ==============================

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state  # Current state (e.g., reasoning step vector)
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0  # Q-value
        self.decision = None  # Store the decision ("buy" or "sell") as a separate attribute

    def add_child(self, child_state, decision):
        child = MCTSNode(child_state, self)
        child.decision = decision  # Store the decision in the child node
        self.children.append(child)
        return child

def generate_reasoning_step(policy_slm, input_vector):
    """
    Generates a reasoning step using the policy SLM.
    :param policy_slm: PolicySLM (trained small language model)
    :param input_vector: numpy array (input vector)
    :return: str ("buy" or "sell")
    """
    input_tensor = torch.tensor(input_vector, dtype=torch.float32)
    output_tensor = policy_slm(input_tensor)
    decision = "buy" if output_tensor.item() > 0.5 else "sell"  # Classify as "buy" or "sell"
    return decision

def evaluate_reasoning_step(ppm, reasoning_step_decision, true_value):
    """
    Evaluates the quality of a reasoning step using the PPM.
    :param ppm: ProcessRewardModel (trained process reward model)
    :param reasoning_step_decision: str ("buy" or "sell")
    :param true_value: int (1 for "buy", 0 for "sell")
    :return: float (reward score)
    """
    # Convert decision to a reward score
    if reasoning_step_decision == "buy" and true_value == 1:
        reward_score = 1.0  # Correct decision
    elif reasoning_step_decision == "sell" and true_value == 0:
        reward_score = 1.0  # Correct decision
    else:
        reward_score = 0.0  # Incorrect decision
    return reward_score

def mcts(root_state, policy_slm, ppm, num_iterations=1000):
    """
    Performs Monte Carlo Tree Search to explore reasoning steps.
    :param root_state: numpy array (initial state)
    :param policy_slm: PolicySLM (trained small language model)
    :param ppm: ProcessRewardModel (trained process reward model)
    :param num_iterations: int (number of MCTS iterations)
    :return: str (best reasoning step decision: "buy" or "sell")
    """
    root = MCTSNode(root_state)

    for _ in range(num_iterations):
        node = root
        # Selection
        while node.children:
            node = max(node.children, key=lambda n: n.value / (n.visits + 1e-6))

        # Expansion
        reasoning_step_decision = generate_reasoning_step(policy_slm, node.state)
        child_state = node.state  # Use the same state for the child node
        child = node.add_child(child_state, reasoning_step_decision)

        # Simulation
        reward = evaluate_reasoning_step(ppm, child.decision, true_value=1)  # Replace with actual true value
        # Note: You need to pass the true value (1 for "buy", 0 for "sell") here.

        # Backpropagation
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent

    # Return the best reasoning step decision
    return max(root.children, key=lambda n: n.value / (n.visits + 1e-6)).decision

# ==============================
# 5. Training and Validation
# ==============================

def train_policy_slm(policy_slm, optimizer, criterion, train_loader, val_loader, epochs=10):
    """
    Trains the Policy SLM and validates it on a validation set.
    """
    policy_slm.train()
    for epoch in range(epochs):
        # Training
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = policy_slm(inputs.float())
            loss = criterion(outputs, targets.float())
            loss.backward()
            optimizer.step()
        
        # Validation
        policy_slm.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = policy_slm(inputs.float())
                val_loss += criterion(outputs, targets.float()).item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {loss.item()}, Validation Loss: {val_loss}")

def train_ppm(ppm, optimizer, criterion, train_loader, val_loader, epochs=10):
    """
    Trains the Process Reward Model and validates it on a validation set.
    """
    ppm.train()
    for epoch in range(epochs):
        # Training
        for inputs, rewards in train_loader:
            optimizer.zero_grad()
            outputs = ppm(inputs.float())
            loss = criterion(outputs, rewards.float())
            loss.backward()
            optimizer.step()
        
        # Validation
        ppm.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, rewards in val_loader:
                outputs = ppm(inputs.float())
                val_loss += criterion(outputs, rewards.float()).item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {loss.item()}, Validation Loss: {val_loss}")

# ==============================
# 6. Main Script - Updated
# ==============================

if __name__ == "__main__":
    # Parameters
    dimensionality = 10000  # High-dimensional vector size
    hidden_dim = 256        # Increased hidden dimension for PPM
    epochs = 20             # Number of training epochs

    # Load financial dataset
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2025-01-15"
    stock_data = load_stock_data(ticker, start_date, end_date)
    feature_columns = ["Open", "High", "Low", "Close", "Volume"]
    encoded_vectors = encode_stock_data(stock_data, feature_columns, dimensionality)

    # Define targets (e.g., next day's price movement)
    targets = (stock_data["Close"].shift(-1) > stock_data["Close"]).astype(int).values[:-1]
    encoded_vectors = encoded_vectors[:-1]  # Remove the last row (no target for the last day)

    # Compute rewards (for PPM training)
    rewards = compute_rewards(targets, targets)  # Replace with predicted targets in practice

    # Split into training and validation sets
    (train_inputs, val_inputs,
     train_targets, val_targets,
     train_rewards, val_rewards) = train_test_split(encoded_vectors, targets, rewards, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors and create DataLoader for Policy SLM
    train_dataset_slm = TensorDataset(torch.tensor(train_inputs), torch.tensor(train_targets))
    val_dataset_slm = TensorDataset(torch.tensor(val_inputs), torch.tensor(val_targets))
    train_loader_slm = DataLoader(train_dataset_slm, batch_size=10, shuffle=True)
    val_loader_slm = DataLoader(val_dataset_slm, batch_size=10, shuffle=False)

    # Convert to PyTorch tensors and create DataLoader for PPM
    train_dataset_ppm = TensorDataset(torch.tensor(train_inputs), torch.tensor(train_rewards))
    val_dataset_ppm = TensorDataset(torch.tensor(val_inputs), torch.tensor(val_rewards))
    train_loader_ppm = DataLoader(train_dataset_ppm, batch_size=10, shuffle=True)
    val_loader_ppm = DataLoader(val_dataset_ppm, batch_size=10, shuffle=False)

    # Initialize Policy SLM and PPM
    policy_slm = PolicySLM(dimensionality, hidden_dim)  # Output size is 1 (scalar)
    ppm = ProcessRewardModel(dimensionality, hidden_dim)

    # Define optimizers and loss functions
    policy_optimizer = torch.optim.Adam(policy_slm.parameters(), lr=0.001)
    ppm_optimizer = torch.optim.Adam(ppm.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Train Policy SLM
    print("Training Policy SLM...")
    train_policy_slm(policy_slm, policy_optimizer, criterion, train_loader_slm, val_loader_slm, epochs)

    # Train Process Reward Model
    print("Training Process Reward Model...")
    train_ppm(ppm, ppm_optimizer, criterion, train_loader_ppm, val_loader_ppm, epochs)

    # Example problem: Initial state vector
    initial_state = train_inputs[0].astype(np.float32)  # Use the first training input as the initial state

    # Perform MCTS to find the best reasoning step
    print("Running MCTS...")
    best_reasoning_decision = mcts(initial_state, policy_slm, ppm)

    # Evaluate the Best Reasoning Step
    print(f"Best Reasoning Decision: {best_reasoning_decision}")