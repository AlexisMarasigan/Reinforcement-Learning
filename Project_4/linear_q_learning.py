import numpy as np
import pickle
from collections import defaultdict
import os

class LinearQLearning:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        
        # Initialize weights with proper feature size
        self.feature_size = None  # Will be determined on first state observation
        self.initialized = False
        self.weights = None  # Will be initialized with proper size later
    
    def _initialize_weights(self, state):
        """Initialize weights based on the first observed state's feature size"""
        features = self.featurize_state(state)
        self.feature_size = features.shape[0]
        print(f"Initializing weights with feature size: {self.feature_size}")
        self.weights = defaultdict(lambda: np.zeros(self.feature_size, dtype=np.float32))
        self.initialized = True
    
    def featurize_state(self, state):
        """Convert raw state into feature representation."""
        # Handle different types of state returns
        if isinstance(state, tuple):
            state = state[0]  # Extract the observation if env returns (obs, info)
            
        try:
            # Convert to numpy array with error handling
            state = np.array(state, dtype=np.float32)  # Convert to numpy array
            
            # Replace any NaN or inf values
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
                
            if len(state.shape) > 1:
                # For image-based observations, flatten
                state = state.flatten()
                
            # Normalize values if they appear to be pixel values
            # Use safe division to avoid NaN results
            if np.max(np.abs(state)) > 1.0:
                max_val = np.max(np.abs(state))
                if max_val > 0:
                    state = state / max_val
                    
            # Final safety check to ensure no NaN values
            state = np.nan_to_num(state, nan=0.0)
            
            return state
            
        except Exception as e:
            print(f"Error in featurize_state: {e}")
            # Return zero vector as fallback
            if self.feature_size:
                return np.zeros(self.feature_size, dtype=np.float32)
            else:
                # Make a guess if we haven't initialized yet
                return np.zeros(100, dtype=np.float32)
    
    def q_value(self, state, action):
        """Compute Q-value as a linear function of weights and features."""
        if not self.initialized:
            self._initialize_weights(state)
            
        features = self.featurize_state(state)
        
        # Debug the dimensions
        if features.shape[0] != self.weights[action].shape[0]:
            print(f"Dimension mismatch: Features {features.shape}, Weights {self.weights[action].shape}")
            # Reinitialize weights for this action if dimensions don't match
            self.weights[action] = np.zeros(features.shape[0], dtype=np.float32)
            
        # Safe dot product calculation
        dot_product = np.dot(self.weights[action], features)
        
        # Check for NaN result and handle it
        if np.isnan(dot_product) or np.isinf(dot_product):
            return 0.0
            
        return dot_product
    
    def best_action(self, state):
        """Select best action based on current Q-values."""
        q_values = [self.q_value(state, a) for a in range(self.env.action_space.n)]
        
        # Handle case where all Q-values are NaN (shouldn't happen with our fixes)
        if all(np.isnan(q) for q in q_values):
            return self.env.action_space.sample()  # Fall back to random action
            
        return np.argmax(q_values)
    
    def policy(self, state):
        """Epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Random action
        return self.best_action(state)
    
    def update(self, state, action, reward, next_state, done):
        """Perform Q-learning update using a linear function approximation."""
        if not self.initialized:
            self._initialize_weights(state)
            
        features = self.featurize_state(state)
        
        # Handle potential NaN or inf in reward
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
            
        next_q = 0 if done else self.q_value(next_state, self.best_action(next_state))
        target = reward + self.gamma * next_q
        current_q = self.q_value(state, action)
        
        # Calculate update with safety checks
        update = self.alpha * (target - current_q) * features
        
        # Check for NaN values in the update
        if np.any(np.isnan(update)) or np.any(np.isinf(update)):
            return
            
        self.weights[action] += update
    
    def train(self, episodes=10000):
        """Train the agent using Q-learning."""
        rewards_history = []
        
        for episode in range(episodes):
            # Handle different reset API (gym vs gymnasium)
            try:
                state, info = self.env.reset()
            except (ValueError, TypeError):
                try:
                    state = self.env.reset()
                except TypeError:
                    # Handles the case where reset() returns a tuple in older versions
                    state = self.env.reset()[0]
            
            # Initialize weights if not already done
            if not self.initialized:
                self._initialize_weights(state)
            
            done = False
            total_reward = 0
            steps = 0
            max_steps = 10000  # Safety limit to prevent infinite loops
            
            while not done and steps < max_steps:
                action = self.policy(state)
                
                # Handle different step API (gym vs gymnasium)
                try:
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                except (ValueError, TypeError):
                    try:
                        next_state, reward, done, info = self.env.step(action)
                    except ValueError:
                        # Handle case where info might not be returned
                        result = self.env.step(action)
                        if len(result) == 4:
                            next_state, reward, done, _ = result
                        else:
                            next_state, reward, done = result
                
                # Safety check for reward
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0.0
                
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
            
            rewards_history.append(total_reward)
            avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
            print(f"Episode {episode}: Total Reward = {total_reward}, Avg Last 100 = {avg_reward:.2f}, Steps = {steps}")
        
        # Create directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Save weights
        with open("models/linear_q_weights.pkl", "wb") as f:
            # Convert defaultdict to regular dict for saving
            weights_dict = {k: v for k, v in self.weights.items()}
            pickle.dump({
                'weights': weights_dict,
                'feature_size': self.feature_size
            }, f)
        
        return rewards_history
    
    def test(self, render=False):
        """Run a trained policy."""
        try:
            with open("models/linear_q_weights.pkl", "rb") as f:
                saved_data = pickle.load(f)
                
            if isinstance(saved_data, dict) and 'weights' in saved_data:
                # New format
                loaded_weights = saved_data['weights']
                self.feature_size = saved_data['feature_size']
            else:
                # Old format - just weights
                loaded_weights = saved_data
                # Will determine feature size from first state
                
            # Initialize if we have feature size
            if self.feature_size:
                self.weights = defaultdict(lambda: np.zeros(self.feature_size, dtype=np.float32))
                self.initialized = True
                
                # Load saved weights
                for k, v in loaded_weights.items():
                    self.weights[k] = v
                
            # Handle different reset API
            try:
                state, info = self.env.reset()
            except (ValueError, TypeError):
                try:
                    state = self.env.reset()
                except TypeError:
                    state = self.env.reset()[0]
            
            # Initialize weights if not done in loading
            if not self.initialized:
                self._initialize_weights(state)
                # Now try to load weights again with correct size
                for k, v in loaded_weights.items():
                    if v.shape[0] == self.feature_size:
                        self.weights[k] = v
                    else:
                        print(f"Warning: Skipping weights for action {k} due to dimension mismatch")
            
            done = False
            total_reward = 0
            steps = 0
            max_steps = 10000  # Safety limit
            
            while not done and steps < max_steps:
                action = self.best_action(state)
                
                # Handle different step API
                try:
                    state, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                except (ValueError, TypeError):
                    try:
                        state, reward, done, info = self.env.step(action)
                    except ValueError:
                        # Handle case where info might not be returned
                        result = self.env.step(action)
                        if len(result) == 4:
                            state, reward, done, _ = result
                        else:
                            state, reward, done = result
                
                # Safety check for reward
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0.0
                
                total_reward += reward
                steps += 1
                
                if render:
                    try:
                        self.env.render()
                    except Exception as e:
                        print(f"Warning: Couldn't render - {e}")
            
            print(f"Test episode completed with total reward: {total_reward}")
            return total_reward
            
        except FileNotFoundError:
            print("No saved model found. Please train the agent first.")
            return 0
        except Exception as e:
            print(f"Error loading model: {e}")
            return 0