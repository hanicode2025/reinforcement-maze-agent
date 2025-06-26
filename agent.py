import numpy as np
import random

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = {}  # state-action pairs
        self.actions = actions
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.epsilon = epsilon  # exploration rate

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # explore
        else:
            q_vals = [self.get_q(state, a) for a in self.actions]
            max_q = max(q_vals)
            max_actions = [a for a, q in zip(self.actions, q_vals) if q == max_q]
            return random.choice(max_actions)  # exploit

    def learn(self, state, action, reward, next_state):
        max_q_next = max([self.get_q(next_state, a) for a in self.actions])
        old_q = self.get_q(state, action)
        new_q = old_q + self.alpha * (reward + self.gamma * max_q_next - old_q)
        self.q_table[(state, action)] = new_q
