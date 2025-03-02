import gymnasium as gym
import argparse
import ale_py
from config import ENV_NAME, EPISODES
from linear_q_learning import LinearQLearning

def main():
    parser = argparse.ArgumentParser(description="Train RL Agent on Atari Games (Linear Function Approximation)")
    parser.add_argument("--train", action="store_true", help="Train the agent")
    parser.add_argument("--test", action="store_true", help="Test the trained model")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration rate")

    args = parser.parse_args()

    gym.register_envs(ale_py)
    env = gym.make(ENV_NAME, render_mode="human")
    
    agent = LinearQLearning(env, alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon)

    if args.train:
        agent.train(EPISODES)
    if args.test:
        agent.test(render=True)

    env.close()

if __name__ == "__main__":
    main()
