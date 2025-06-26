from environment import GridWorld
from agent import QLearningAgent
import matplotlib.pyplot as plt

# Initialize environment
env = GridWorld(
    width=5, height=5,
    start=(0, 0), goal=(4, 4),
    obstacles=[(1,1), (2,2), (3,3)]
)

# Initialize agent
actions = env.get_actions()
agent = QLearningAgent(actions, alpha=0.1, gamma=0.9, epsilon=0.2)

episodes = 200
rewards_per_episode = []

for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    while True:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state
        total_reward += reward
        if done:
            break

    rewards_per_episode.append(total_reward)
    if episode % 20 == 0:
        print(f"Episode {episode} Total Reward: {total_reward}")

# Plot results
plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-learning Maze Training Progress")
plt.grid()
plt.show()
