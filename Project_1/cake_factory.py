import numpy as np
import matplotlib.pyplot as plt

class CakeProductionMDP:
    def __init__(self, theta=0.2, R=5, C=4, Q=10, gamma=0.9, theta_convergence=1e-6):
        self.states = ['pristine', 'worn', 'broken']
        self.actions = ['continue', 'repair']
        self.theta = theta
        self.R = R
        self.C = C
        self.Q = Q
        self.gamma = gamma
        self.theta_convergence = theta_convergence
        self.transition_probs = {
            'pristine': {
                'continue': {'pristine': 1 - theta, 'worn': theta, 'broken': 0},
                'repair': {'pristine': 1, 'worn': 0, 'broken': 0},
            },
            'worn': {
                'continue': {'pristine': 0, 'worn': 1 - theta, 'broken': theta},
                'repair': {'pristine': 1, 'worn': 0, 'broken': 0},
            },
            'broken': {
                'continue': {'pristine': 1, 'worn': 0, 'broken': 0},
                'repair': {'pristine': 1, 'worn': 0, 'broken': 0},
            }
        }
        self.rewards = {
            'pristine': {'continue': C, 'repair': -R},
            'worn': {'continue': C / 2, 'repair': -R},
            'broken': {'continue': -Q, 'repair': -Q},
        }
        self.V = {s: 0 for s in self.states}
        self.policy = {s: np.random.choice(self.actions) for s in self.states}
    
    def value_iteration(self):
        iteration_values = []
        while True:
            delta = 0
            for s in self.states:
                v = self.V[s]
                self.V[s] = max(
                    sum(
                        p * (r + self.gamma * self.V[s_next])
                        for s_next, p in self.transition_probs[s][a].items()
                        for r in [self.rewards[s][a]]
                    )
                    for a in self.actions
                )
                delta = max(delta, abs(v - self.V[s]))
            iteration_values.append(sum(self.V.values()))
            if delta < self.theta_convergence:
                break
        
        for s in self.states:
            self.policy[s] = max(
                self.actions,
                key=lambda a: sum(
                    p * (r + self.gamma * self.V[s_next])
                    for s_next, p in self.transition_probs[s][a].items()
                    for r in [self.rewards[s][a]]
                )
            )
        return self.V, self.policy, iteration_values

# Plot learning curves for different gamma values
gamma_values = [0.5, 0.7, 0.9]
plt.figure(figsize=(10, 6))
for gamma in gamma_values:
    env = CakeProductionMDP(gamma=gamma)
    _, _, iteration_values = env.value_iteration()
    plt.plot(iteration_values, label=f'Gamma = {gamma}')
    print(env.policy)
    print(env.V)

plt.xlabel('Iterations')
plt.ylabel('Sum of Value Function')
plt.title('Learning Curves for Different Gamma Values')
plt.legend()
plt.grid()
plt.show()

