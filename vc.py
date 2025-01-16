import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE  # For handling class imbalance
from sklearn.base import BaseEstimator, ClassifierMixin  # For compatibility with sklearn

# Step 1: Generate Realistic Synthetic VC Data
def generate_realistic_vc_data(num_startups=1000):
    """
    Generate realistic synthetic data for VC analytics.
    Features include: funding_rounds, revenue, burn_rate, market_size, team_size, founder_experience, and success_label.
    """
    np.random.seed(42)
    
    # Synthetic features
    funding_rounds = np.random.randint(1, 5, num_startups)  # Number of funding rounds
    revenue = np.random.lognormal(mean=1.5, sigma=0.5, size=num_startups)  # Revenue in millions (log-normal distribution)
    burn_rate = np.random.uniform(0.1, 2.0, num_startups)   # Monthly burn rate in millions
    market_size = np.random.lognormal(mean=3.0, sigma=0.5, size=num_startups)  # Market size in billions (log-normal distribution)
    team_size = np.random.randint(5, 50, num_startups)      # Team size
    founder_experience = np.random.randint(0, 10, num_startups)  # Founder experience in years
    
    # Synthetic target: success_label (1 = successful, 0 = not successful)
    # Success is based on a combination of features with some noise
    success_label = (
        (funding_rounds > 2) & 
        (revenue > 5) & 
        (burn_rate < 1.0) & 
        (market_size > 50) & 
        (team_size > 20) & 
        (founder_experience > 5)
    ).astype(int)
    
    # Introduce some noise to make the dataset more realistic
    noise = np.random.choice([0, 1], size=num_startups, p=[0.95, 0.05])  # 5% noise (reduced from 10%)
    success_label = np.logical_xor(success_label, noise).astype(int)
    
    # Create a DataFrame
    data = pd.DataFrame({
        'funding_rounds': funding_rounds,
        'revenue': revenue,
        'burn_rate': burn_rate,
        'market_size': market_size,
        'team_size': team_size,
        'founder_experience': founder_experience,
        'success_label': success_label
    })
    
    # Add engineered features
    data['revenue_to_burn_rate'] = data['revenue'] / data['burn_rate']
    data['market_size_per_team_member'] = data['market_size'] / data['team_size']
    data['funding_per_founder_experience'] = data['funding_rounds'] / (data['founder_experience'] + 1)
    
    return data

# Step 2: Hyperdimensional Computing (HDC) Encoding
def hdc_encoding(data, dimensionality=10000):
    """
    Simulate HDC encoding by projecting data into a high-dimensional space using random projections.
    """
    np.random.seed(42)
    num_samples, num_features = data.shape
    random_projection_matrix = np.random.randn(num_features, dimensionality)
    hdc_vectors = np.dot(data, random_projection_matrix)
    return hdc_vectors

# Step 3: Custom Boosting with Weak Learners
class BoostHD(BaseEstimator, ClassifierMixin):
    def __init__(self, n_learners=100, dimensionality=10000, max_depth=5, min_samples_split=10):
        self.n_learners = n_learners
        self.dimensionality = dimensionality
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.learners = []
        self.learner_weights = []

    def fit(self, X, y):
        """
        Fit the BoostHD model using AdaBoost with weak learners.
        This method is required for compatibility with cross_val_score.
        """
        self.learners = []  # Reset learners
        self.learner_weights = []  # Reset learner weights
        
        # Initialize sample weights with higher weight for the minority class
        sample_weights = np.ones(len(X)) / len(X)
        minority_class_weight = 5.0  # Higher weight for minority class
        sample_weights[y == 1] *= minority_class_weight
        sample_weights /= np.sum(sample_weights)  # Normalize sample weights
        
        for i in range(self.n_learners):
            print(f"\nTraining Weak Learner {i+1}...")
            
            # Train a weak learner (Decision Tree with regularization)
            weak_learner = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            weak_learner.fit(X, y, sample_weight=sample_weights)
            
            # Make predictions
            y_pred = weak_learner.predict(X)
            
            # Calculate error and learner weight
            incorrect = (y_pred != y)
            error = np.sum(sample_weights * incorrect) / np.sum(sample_weights)
            print(f"Error for Weak Learner {i+1}: {error:.4f}")
            
            # Handle the case where error = 0 (perfect classification)
            if error == 0:
                print(f"Early stopping: Weak learner {i+1} achieved perfect classification.")
                self.learners.append(weak_learner)
                self.learner_weights.append(1.0)  # Assign maximum weight
                break
            
            # Handle the case where error >= 0.5 (weak learner is worse than random guessing)
            if error >= 0.5:
                print(f"Early stopping: Weak learner {i+1} has error >= 0.5.")
                break
            
            learner_weight = 0.5 * np.log((1 - error) / error)
            print(f"Learner Weight for Weak Learner {i+1}: {learner_weight:.4f}")
            
            # Update sample weights
            sample_weights *= np.exp(-learner_weight * y * y_pred)
            sample_weights /= np.sum(sample_weights)  # Normalize sample weights
            print(f"Sample Weights after Weak Learner {i+1}:")
            print(sample_weights[:5])  # Print the first 5 sample weights for debugging
            
            # Save the learner and its weight
            self.learners.append(weak_learner)
            self.learner_weights.append(learner_weight)
        
        return self

    def predict(self, X):
        """
        Make predictions using the ensemble of weak learners.
        """
        predictions = np.zeros(len(X))
        for learner, weight in zip(self.learners, self.learner_weights):
            predictions += weight * learner.predict(X)
        return np.sign(predictions)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        Required for compatibility with sklearn.
        """
        return {
            "n_learners": self.n_learners,
            "dimensionality": self.dimensionality,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split
        }

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        Required for compatibility with sklearn.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

# Step 4: Visualizations
def plot_feature_distributions(data):
    """
    Plot distributions of key features.
    """
    plt.figure(figsize=(15, 10))
    plt.suptitle("Feature Distributions", fontsize=16)
    
    plt.subplot(2, 3, 1)
    sns.histplot(data['revenue'], kde=True, color='blue')
    plt.title("Revenue Distribution")
    
    plt.subplot(2, 3, 2)
    sns.histplot(data['burn_rate'], kde=True, color='green')
    plt.title("Burn Rate Distribution")
    
    plt.subplot(2, 3, 3)
    sns.histplot(data['market_size'], kde=True, color='orange')
    plt.title("Market Size Distribution")
    
    plt.subplot(2, 3, 4)
    sns.histplot(data['team_size'], kde=True, color='purple')
    plt.title("Team Size Distribution")
    
    plt.subplot(2, 3, 5)
    sns.histplot(data['founder_experience'], kde=True, color='red')
    plt.title("Founder Experience Distribution")
    
    plt.tight_layout()
    plt.show()

def plot_hdc_encoding(hdc_vectors):
    """
    Visualize HDC encoding using PCA for dimensionality reduction.
    """
    pca = PCA(n_components=2)
    hdc_2d = pca.fit_transform(hdc_vectors)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(hdc_2d[:, 0], hdc_2d[:, 1], alpha=0.5, color='blue')
    plt.title("HDC Encoding Visualization (2D PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    """
    Plot a confusion matrix for model evaluation.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# Step 5: Main Function to Demonstrate BoostHD for VC Analytics
def main():
    # Step 1: Generate realistic VC data
    vc_data = generate_realistic_vc_data(num_startups=1000)
    print("Generated Realistic VC Data:")
    print(vc_data.head())
    
    # Step 2: Plot feature distributions
    plot_feature_distributions(vc_data)
    
    # Step 3: Simulate HDC encoding
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(vc_data.drop(columns=['success_label']))
    hdc_vectors = hdc_encoding(scaled_data, dimensionality=10000)
    print("\nHDC-Encoded Data Shape:", hdc_vectors.shape)
    
    # Step 4: Visualize HDC encoding
    plot_hdc_encoding(hdc_vectors)
    
    # Step 5: Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(hdc_vectors, vc_data['success_label'])
    
    # Step 6: Apply custom BoostHD for predictive modeling
    print("\nTraining BoostHD Model...")
    boosthd_model = BoostHD(n_learners=100, dimensionality=10000, max_depth=5, min_samples_split=10)
    boosthd_model.fit(X_resampled, y_resampled)
    
    # Step 7: Evaluate the model using cross-validation
    print("\nEvaluating BoostHD Model with Cross-Validation...")
    cv_scores = cross_val_score(boosthd_model, X_resampled, y_resampled, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy: {np.mean(cv_scores):.2f} (±{np.std(cv_scores):.2f})")
    
    # Step 8: Evaluate the model on the test set
    X_train, X_test, y_train, y_test = train_test_split(hdc_vectors, vc_data['success_label'], test_size=0.2, random_state=42)
    y_pred = boosthd_model.predict(X_test)
    y_pred = (y_pred > 0).astype(int)  # Convert predictions to binary labels
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Step 9: Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)
    
    # Step 10: Simulate a new startup prediction
    new_startup = pd.DataFrame({
        'funding_rounds': [3],
        'revenue': [6.5],
        'burn_rate': [0.8],
        'market_size': [75],
        'team_size': [25],
        'founder_experience': [7]
    })
    new_startup_scaled = scaler.transform(new_startup)
    new_startup_hdc = hdc_encoding(new_startup_scaled, dimensionality=10000)
    prediction = boosthd_model.predict(new_startup_hdc)
    prediction_label = 'Successful' if prediction[0] > 0 else 'Not Successful'
    print(f"\nPrediction for New Startup: {prediction_label}")

if __name__ == "__main__":
    main()