
import numpy as np


# ==========================================
# ENVIRONMENT
# ==========================================

class GridWorld:

    def __init__(self):

        self.rows = 4
        self.cols = 4

        self.start_state = (0, 0)
        self.goal_state = (3, 3)
        self.pit_state = (1, 3)

        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]

        self.agent_position = self.start_state

        self.total_steps = 0
        self.max_steps = 30

    def reset(self):

        self.agent_position = self.start_state
        self.total_steps = 0

        return self.agent_position

    def step(self, action):

        row, col = self.agent_position

        # Execute selected action
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

        # Apply environment boundaries
        row = np.clip(row, 0, self.rows - 1)
        col = np.clip(col, 0, self.cols - 1)

        next_state = (int(row), int(col))

        self.agent_position = next_state
        self.total_steps += 1

        # Calculate reward
        if next_state == self.goal_state:

            reward = 10
            terminated = True

        elif next_state == self.pit_state:

            reward = -10
            terminated = True

        else:

            reward = -1
            terminated = False

        # Time-limit termination
        truncated = (
            self.total_steps >= self.max_steps
            and not terminated
        )

        return next_state, reward, terminated, truncated

    def render(self):

        for r in range(self.rows):

            row_display = []

            for c in range(self.cols):

                position = (r, c)

                if position == self.agent_position:
                    symbol = "A"

                elif position == self.goal_state:
                    symbol = "G"

                elif position == self.pit_state:
                    symbol = "P"

                else:
                    symbol = "."

                row_display.append(symbol)

            print(" ".join(row_display))

        print()


# ==========================================
# AGENT
# ==========================================

class RandomAgent:

    def __init__(self, actions):

        self.actions = actions

    def select_action(self, state):

        # Random action selection
        action = np.random.choice(self.actions)

        return action


# ==========================================
# RL INTERACTION
# ==========================================

def run_episode(environment, agent):

    state = environment.reset()

    total_reward = 0
    episode_steps = 0

    terminated = False
    truncated = False

    print("\nInitial State:", state)
    environment.render()

    while not (terminated or truncated):

        # Agent selects an action
        action = agent.select_action(state)

        # Environment executes action
        next_state, reward, terminated, truncated = (
            environment.step(action)
        )

        # Track episode information
        total_reward += reward
        episode_steps += 1

        print("Current State:", state)
        print("Selected Action:", action)
        print("Next State:", next_state)
        print("Reward:", reward)

        environment.render()

        # Update current state
        state = next_state

    print("Episode Finished!")
    print("Total Reward:", total_reward)
    print("Total Steps:", episode_steps)

    if terminated:
        print("Termination: Goal or Pit reached")

    elif truncated:
        print("Termination: Maximum steps reached")

    return total_reward, episode_steps


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    np.random.seed(42)

    environment = GridWorld()

    agent = RandomAgent(environment.actions)

    print("REINFORCEMENT LEARNING")
    print("Agent-Environment Interaction")

    run_episode(environment, agent)
