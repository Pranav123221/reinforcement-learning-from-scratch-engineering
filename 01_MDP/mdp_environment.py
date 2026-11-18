
import numpy as np


class GridWorld:

    def __init__(self):

        # Environment dimensions
        self.rows = 4
        self.cols = 4

        # State space
        self.states = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
        ]

        # Action space
        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]

        # Terminal states
        self.goal = (3, 3)
        self.pit = (1, 3)

        # Rewards
        self.goal_reward = 10
        self.pit_reward = -10
        self.step_reward = -1

        # Discount factor
        self.gamma = 0.9

        self.reset()

    def reset(self):

        self.agent_position = (0, 0)

        return self.agent_position

    def step(self, action):

        row, col = self.agent_position

        # Action execution
        if action == "UP":
            row -= 1

        elif action == "DOWN":
            row += 1

        elif action == "LEFT":
            col -= 1

        elif action == "RIGHT":
            col += 1

        else:
            raise ValueError("Invalid action")

        # Boundary conditions
        row = max(0, min(row, self.rows - 1))
        col = max(0, min(col, self.cols - 1))

        next_state = (row, col)

        # Reward calculation
        if next_state == self.goal:

            reward = self.goal_reward
            done = True

        elif next_state == self.pit:

            reward = self.pit_reward
            done = True

        else:

            reward = self.step_reward
            done = False

        self.agent_position = next_state

        return next_state, reward, done

    def render(self):

        for r in range(self.rows):

            row_display = []

            for c in range(self.cols):

                position = (r, c)

                if position == self.agent_position:
                    symbol = "A"

                elif position == self.goal:
                    symbol = "G"

                elif position == self.pit:
                    symbol = "P"

                else:
                    symbol = "."

                row_display.append(symbol)

            print(" ".join(row_display))

        print()


# Create environment
env = GridWorld()

# Initial state
state = env.reset()

print("Initial State:", state)
print("\nInitial Environment:")

env.render()


# Agent takes actions
actions = ["RIGHT", "RIGHT", "DOWN", "DOWN", "DOWN", "RIGHT"]

for action in actions:

    next_state, reward, done = env.step(action)

    print("Action:", action)
    print("Next State:", next_state)
    print("Reward:", reward)
    print("Terminal:", done)

    env.render()

    if done:
        print("Episode Finished!")
        break
