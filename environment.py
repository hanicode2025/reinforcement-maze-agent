import numpy as np

class GridWorld:
    def __init__(self, width=5, height=5, start=(0, 0), goal=(4, 4), obstacles=[]):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.reset()

    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos

    def get_state(self):
        return self.agent_pos

    def is_done(self):
        return self.agent_pos == self.goal

    def get_actions(self):
        return ['up', 'down', 'left', 'right']

    def step(self, action):
        x, y = self.agent_pos

        if action == 'up':
            y = max(0, y - 1)
        elif action == 'down':
            y = min(self.height - 1, y + 1)
        elif action == 'left':
            x = max(0, x - 1)
        elif action == 'right':
            x = min(self.width - 1, x + 1)

        if (x, y) in self.obstacles:
            reward = -10
            done = False
        elif (x, y) == self.goal:
            reward = 100
            done = True
        else:
            reward = -1
            done = False

        self.agent_pos = (x, y)
        return self.agent_pos, reward, done

if __name__ == "__main__":
    env = GridWorld(
        width=5, height=5, 
        start=(0, 0), goal=(4, 4),
        obstacles=[(1,1), (2,2), (3,3)]
    )

    print("Initial State:", env.reset())

    actions = ['right', 'right', 'down', 'down', 'down', 'right']
    for action in actions:
        state, reward, done = env.step(action)
        print(f"Action: {action}, State: {state}, Reward: {reward}, Done: {done}")
        if done:
            break

