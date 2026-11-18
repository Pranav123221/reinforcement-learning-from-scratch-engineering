
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

        self.max_steps = 50

        self.reset()

    def reset(self, start_state=(0, 0)):

        self.agent_position = start_state
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
# POLICY
# ==========================================

class EpsilonSoftPolicy:

    def __init__(self, actions, epsilon=0.2):

        self.actions = actions
        self.epsilon = epsilon

    def preferred_action(self, state):

        row, col = state

        # Avoid the pit at (1, 3)
        if col == 3 and row < 3:
            return "LEFT"

        # Move towards the bottom row
        if row < 3:
            return "DOWN"

        # Move towards the goal
        if col < 3:
            return "RIGHT"

        return "UP"

    def select_action(self, state):

        if np.random.random() < self.epsilon:

            return np.random.choice(self.actions)

        return self.preferred_action(state)


# ==========================================
# MONTE CARLO STATE-VALUE ESTIMATION
# ==========================================

def estimate_state_values(
    env,
    policy,
    episodes=500,
    gamma=0.9
):

    state_values = {}

    all_states = [
        (r, c)
        for r in range(env.rows)
        for c in range(env.cols)
        if (r, c) not in [env.goal_state, env.pit_state]
    ]

    for start_state in all_states:

        returns = []

        for episode in range(episodes):

            state = env.reset(start_state)

            episode_rewards = []

            terminated = False
            truncated = False

            while not (terminated or truncated):

                action = policy.select_action(state)

                next_state, reward, terminated, truncated = (
                    env.step(action)
                )

                episode_rewards.append(reward)

                state = next_state

            # Calculate return from starting state
            G = 0

            for reward in reversed(episode_rewards):

                G = reward + gamma * G

            returns.append(G)

        # Average sampled returns
        state_values[start_state] = np.mean(returns)

    return state_values


# ==========================================
# VALUE TABLE
# ==========================================

def display_value_table(env, state_values):

    print("\nSTATE-VALUE FUNCTION V(s)")
    print("=" * 55)

    for row in range(env.rows):

        values = []

        for col in range(env.cols):

            state = (row, col)

            if state == env.goal_state:

                values.append(" GOAL ")

            elif state == env.pit_state:

                values.append("  PIT  ")

            else:

                value = state_values[state]

                values.append(f"{value:7.2f}")

        print(" | ".join(values))


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    np.random.seed(42)

    GAMMA = 0.9
    EPISODES = 500

    env = GridWorld()

    policy = EpsilonSoftPolicy(
        actions=env.actions,
        epsilon=0.2
    )

    print("STATE-VALUE FUNCTION EXPERIMENT")

    print("\nGamma:", GAMMA)
    print("Episodes per state:", EPISODES)
    print("Exploration probability:", policy.epsilon)

    state_values = estimate_state_values(
        env,
        policy,
        episodes=EPISODES,
        gamma=GAMMA
    )

    display_value_table(
        env,
        state_values
    )

    print("\nINDIVIDUAL STATE VALUES")

    for state, value in state_values.items():

        print(f"State {state}: V(s) = {value:.4f}")
