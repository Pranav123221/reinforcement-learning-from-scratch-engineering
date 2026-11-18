
import numpy as np
from collections import defaultdict

# ==========================================
# CONFIGURATION
# ==========================================

GRID_SIZE = 4

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

MOVES = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1)
}

GOAL = (3, 3)
PIT = (1, 3)

TERMINAL_STATES = {GOAL, PIT}

GAMMA = 0.9
EPSILON = 0.1

NUM_EPISODES = 30_000
MAX_STEPS = 100

SEED = 42


# ==========================================
# GRIDWORLD ENVIRONMENT
# ==========================================

class GridWorld:

    def __init__(self):
        self.state = (0, 0)

    def reset(self, start_state=None):

        if start_state is None:
            start_state = (0, 0)

        if start_state in TERMINAL_STATES:
            raise ValueError(
                "Starting state cannot be terminal."
            )

        self.state = start_state

        return self.state

    def step(self, action):

        if self.state in TERMINAL_STATES:
            raise RuntimeError("Episode has terminated.")

        row, col = self.state

        dr, dc = MOVES[action]

        next_row = row + dr
        next_col = col + dc

        if not (
            0 <= next_row < GRID_SIZE
            and 0 <= next_col < GRID_SIZE
        ):
            next_row, next_col = row, col

        next_state = (next_row, next_col)

        if next_state == GOAL:
            reward = 10
        elif next_state == PIT:
            reward = -10
        else:
            reward = -1

        terminated = next_state in TERMINAL_STATES

        self.state = next_state

        return next_state, reward, terminated


# ==========================================
# STATE AND ACTION HELPERS
# ==========================================

def get_states():

    return [
        (row, col)
        for row in range(GRID_SIZE)
        for col in range(GRID_SIZE)
        if (row, col) not in TERMINAL_STATES
    ]


def initialize_q_table():

    return {
        state: {
            action: 0.0
            for action in ACTIONS
        }
        for state in get_states()
    }


# ==========================================
# EPSILON-SOFT ACTION SELECTION
# ==========================================

def select_action(state, Q, rng):

    # Explore
    if rng.random() < EPSILON:
        return rng.choice(ACTIONS)

    # Exploit
    values = Q[state]

    max_value = max(values.values())

    best_actions = [
        action
        for action, value in values.items()
        if np.isclose(value, max_value)
    ]

    # Random tie-breaking avoids fixed action bias.
    return rng.choice(best_actions)


# ==========================================
# GENERATE EPISODE
# ==========================================

def generate_episode(env, Q, rng):

    # Exploring Starts:
    # Select a random non-terminal initial state.
    start_state = get_states()[
        rng.integers(len(get_states()))
    ]

    state = env.reset(start_state)

    episode = []

    for _ in range(MAX_STEPS):

        action = select_action(state, Q, rng)

        next_state, reward, terminated = env.step(action)

        episode.append((state, action, reward))

        state = next_state

        if terminated:
            break

    return episode


# ==========================================
# FIRST-VISIT MONTE CARLO CONTROL
# ==========================================

def monte_carlo_control():

    rng = np.random.default_rng(SEED)

    env = GridWorld()

    Q = initialize_q_table()

    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)

    episode_rewards = []

    for episode_number in range(NUM_EPISODES):

        episode = generate_episode(env, Q, rng)

        # Record total episode reward.
        total_reward = sum(
            reward for _, _, reward in episode
        )

        episode_rewards.append(total_reward)

        G = 0.0

        visited_state_actions = set()

        # Calculate returns backwards.
        for state, action, reward in reversed(episode):

            G = reward + GAMMA * G

            state_action = (state, action)

            # First-visit MC control.
            if state_action in visited_state_actions:
                continue

            visited_state_actions.add(state_action)

            returns_sum[state_action] += G
            returns_count[state_action] += 1

            Q[state][action] = (
                returns_sum[state_action]
                / returns_count[state_action]
            )

    return Q, episode_rewards, returns_count


# ==========================================
# EXTRACT GREEDY POLICY
# ==========================================

def extract_policy(Q):

    policy = {}

    for state in get_states():

        action_values = Q[state]

        max_value = max(action_values.values())

        best_actions = [
            action
            for action, value in action_values.items()
            if np.isclose(value, max_value)
        ]

        policy[state] = best_actions[0]

    return policy


# ==========================================
# DISPLAY Q TABLE
# ==========================================

def display_q_table(Q):

    print("\nLEARNED ACTION VALUES")
    print("=" * 55)

    for state in get_states():

        print(f"\nState: {state}")

        for action in ACTIONS:

            print(
                f"  {action:5s}: "
                f"{Q[state][action]:9.3f}"
            )


# ==========================================
# DISPLAY POLICY
# ==========================================

def display_policy(policy):

    symbols = {
        "UP": " ↑ ",
        "DOWN": " ↓ ",
        "LEFT": " ← ",
        "RIGHT": " → "
    }

    print("\nGREEDY POLICY")
    print("=" * 40)

    for row in range(GRID_SIZE):

        line = []

        for col in range(GRID_SIZE):

            state = (row, col)

            if state == GOAL:
                line.append(" G ")
            elif state == PIT:
                line.append(" P ")
            else:
                line.append(symbols[policy[state]])

        print(" | ".join(line))


# ==========================================
# DISPLAY LEARNING STATISTICS
# ==========================================

def display_statistics(rewards, counts):

    print("\nTRAINING STATISTICS")
    print("=" * 45)

    print("Total Episodes:", len(rewards))

    print(
        "Average Reward (last 1000 episodes):",
        round(np.mean(rewards[-1000:]), 3)
    )

    print(
        "State-Action Pairs Visited:",
        len(counts)
    )

    print(
        "Total Possible State-Action Pairs:",
        len(get_states()) * len(ACTIONS)
    )


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print("=" * 55)
    print("MONTE CARLO CONTROL")
    print("=" * 55)

    Q, rewards, counts = monte_carlo_control()

    policy = extract_policy(Q)

    display_statistics(rewards, counts)

    display_q_table(Q)

    display_policy(policy)

    print("\nRESULT")
    print("-" * 40)
    print("Monte Carlo Control training completed.")
    print("Greedy policy extracted from learned Q-values.")
