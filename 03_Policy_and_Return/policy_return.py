
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

        self.max_steps = 30

        self.reset()

    def reset(self):

        self.agent_position = self.start_state
        self.steps = 0

        return self.agent_position

    def step(self, action):

        row, col = self.agent_position

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

        row = int(np.clip(row, 0, self.rows - 1))
        col = int(np.clip(col, 0, self.cols - 1))

        next_state = (row, col)

        self.agent_position = next_state
        self.steps += 1

        if next_state == self.goal_state:
            reward = 10
            terminated = True

        elif next_state == self.pit_state:
            reward = -10
            terminated = True

        else:
            reward = -1
            terminated = False

        truncated = (
            self.steps >= self.max_steps
            and not terminated
        )

        return next_state, reward, terminated, truncated


# ==========================================
# POLICY 1: RANDOM POLICY
# ==========================================

class RandomPolicy:

    def __init__(self, actions):
        self.actions = actions

    def select_action(self, state):

        return np.random.choice(self.actions)


# ==========================================
# POLICY 2: DETERMINISTIC POLICY
# ==========================================

class DeterministicPolicy:

    def __init__(self):

        self.policy = {

            (0, 0): "DOWN",
            (1, 0): "DOWN",
            (2, 0): "DOWN",

            (3, 0): "RIGHT",
            (3, 1): "RIGHT",
            (3, 2): "RIGHT"
        }

    def select_action(self, state):

        return self.policy.get(state, "UP")


# ==========================================
# POLICY 3: STOCHASTIC POLICY
# ==========================================

class StochasticPolicy:

    def __init__(self):

        self.policy = {

            (0, 0): [0.1, 0.7, 0.1, 0.1],
            (1, 0): [0.1, 0.7, 0.1, 0.1],
            (2, 0): [0.1, 0.7, 0.1, 0.1],

            (3, 0): [0.1, 0.1, 0.1, 0.7],
            (3, 1): [0.1, 0.1, 0.1, 0.7],
            (3, 2): [0.1, 0.1, 0.1, 0.7]
        }

        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]

    def select_action(self, state):

        probabilities = self.policy.get(
            state,
            [0.25, 0.25, 0.25, 0.25]
        )

        return np.random.choice(
            self.actions,
            p=probabilities
        )


# ==========================================
# RETURN CALCULATION
# ==========================================

def calculate_discounted_returns(rewards, gamma):

    returns = []

    G = 0

    for reward in reversed(rewards):

        G = reward + gamma * G

        returns.insert(0, G)

    return returns


# ==========================================
# EPISODE EXECUTION
# ==========================================

def run_episode(environment, policy, gamma):

    state = environment.reset()

    rewards = []
    actions_taken = []

    terminated = False
    truncated = False

    while not (terminated or truncated):

        action = policy.select_action(state)

        next_state, reward, terminated, truncated = (
            environment.step(action)
        )

        rewards.append(reward)
        actions_taken.append(action)

        state = next_state

    # Undiscounted episode reward
    total_reward = sum(rewards)

    # Discounted return at every timestep
    discounted_returns = calculate_discounted_returns(
        rewards,
        gamma
    )

    return {
        "actions": actions_taken,
        "rewards": rewards,
        "total_reward": total_reward,
        "discounted_returns": discounted_returns,
        "initial_return": discounted_returns[0],
        "terminated": terminated,
        "truncated": truncated
    }


# ==========================================
# EXPERIMENT
# ==========================================

if __name__ == "__main__":

    np.random.seed(42)

    GAMMA = 0.9

    environment = GridWorld()

    policies = {

        "Random Policy": RandomPolicy(environment.actions),

        "Deterministic Policy": DeterministicPolicy(),

        "Stochastic Policy": StochasticPolicy()
    }

    for name, policy in policies.items():

        result = run_episode(
            environment,
            policy,
            GAMMA
        )

        print("\n" + "=" * 45)
        print(name)
        print("=" * 45)

        print("Actions:", result["actions"])

        print("Rewards:", result["rewards"])

        print("Total Episode Reward:",
              result["total_reward"])

        print("Discounted Returns:",
              result["discounted_returns"])

        print("Initial State Return:",
              result["initial_return"])

        print("Terminated:",
              result["terminated"])

        print("Truncated:",
              result["truncated"])
